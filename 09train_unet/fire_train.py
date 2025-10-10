import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import segmentation_models_pytorch as smp
from utils.data_loading import FireData

# ==== 全局路径 ====
dir_input = Path('your_directory/dnstream/input')
dir_output = Path('your_directory/dnstream/hazard')
dir_checkpoint = Path('your_directory/result/fire/checkpoint')
metric_file = f'your_directory/dnstream/stats/statistics_summary.npz'
mask_file = f'your_directory/dnstream/stats/pmask.npz'
map_mask_file = f'your_directory/dnstream/stats/map_mask.npz'
log_path = f'your_directory/result/fire/train.log'

data_metric = np.load(metric_file, allow_pickle=True)
pmask = np.load(mask_file, allow_pickle=True)['pmask']
output_mask = np.load(map_mask_file, allow_pickle=True)['hazard_mask']
output_mask = output_mask[0]

year = ['2019', '2020', '2021']
input_metric, output_metric = data_metric['input_stats'].item(), data_metric['hazard_stats'].item()

class WeightedFocalMSELoss(torch.nn.Module):
    def __init__(self, threshold=2.0, alpha=5.0, beta=1.0, gamma=2.0):
        super().__init__()
        self.threshold = threshold  # 火灾温度阈值
        self.alpha = alpha          # 火灾点的权重
        self.beta = beta            # 背景点的权重
        self.gamma = gamma          # focal loss 指数项

    def forward(self, pred, target, omask):
        
        fire_mask = (target > self.threshold).float()
        weight_map = fire_mask * self.alpha + (1 - fire_mask) * self.beta
        
        loss = weight_map * (pred - target) ** 2
        loss = loss * omask

        return loss.sum()/omask.sum()

# ==== DDP 主训练函数 ====
def train_ddp(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    dataset = FireData(dir_input, dir_output, year, input_metric, output_metric)
    n_val = int(len(dataset) * args.val / 100)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank)

    loader_args = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val_set, sampler=val_sampler, drop_last=True, **loader_args)

    model = smp.Unet(encoder_name='resnet101', in_channels=9, classes=1)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    criterion = WeightedFocalMSELoss(threshold=2.0, alpha=1000.0, beta=1.0, gamma=2.0)
    omask = torch.tensor(output_mask, dtype=torch.float32).to(device)

    if rank == 0 and (not os.path.exists(log_path)):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            f.write('epoch,progress,avg_train_loss,val_score\n')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0

        num_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            images = batch['input_tensor'].to(device, non_blocking=True)
            true_masks = batch['output_tensor'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            try:
                with torch.autocast(device_type='cuda', enabled=False):
                    masks_pred = model(images)
                    weighted_loss = criterion(masks_pred, true_masks, omask)

                loss_value = weighted_loss.item()
                print(f"[Rank {rank}] Epoch {epoch} - Batch {batch_idx+1}/{num_batches} - Sample Loss: {loss_value:.6f}            ", end='\r')

                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss_value
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"[Rank {rank}] CUDA OOM - Skipping batch")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"[Rank {rank}] CUDA OOM - Skipping batch")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            # ==== 每训练 20% 的 batch，评估一次 ====
            progress = (batch_idx + 1) / num_batches
            eval_checkpoints = [1]
            for checkpoint in eval_checkpoints:
                if abs(progress - checkpoint) < 1e-6 or (
                    (progress > checkpoint) and (progress - 1.0/num_batches < checkpoint)
                ):
                    if rank == 0:
                        val_score = evaluate(model, output_mask, val_loader, device, amp=args.amp, rank=rank)
                        print(f"\n[Epoch {epoch} | {int(checkpoint*100)}%] Intermediate Validation Loss: {val_score:.4f}")
                        #
                        avg_train_loss = epoch_loss / (batch_idx + 1)  # 平均训练损失
                        with open(log_path, 'a') as f:
                            f.write(f"{epoch},{checkpoint:.2f},{avg_train_loss:.6f},{val_score:.6f}\n")
                        #
                    break  # 保证每个 checkpoint 只触发一次

        if rank == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.module.state_dict(), str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))

        scheduler.step()

    dist.destroy_process_group()

# ==== 评估函数 ====
def evaluate(model, output_mask, val_loader, device, amp=False, rank=0):
    model.eval()
    omask = torch.tensor(output_mask, dtype=torch.float32).to(device)
    criterion = WeightedFocalMSELoss(threshold=2.0, alpha=1000.0, beta=1.0, gamma=2.0)
    scores = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch['input_tensor'].to(device)
            true_masks = batch['output_tensor'].to(device)
    
            with torch.autocast(device_type='cuda', enabled=amp):
                masks_pred = model(images)
                weighted_loss = criterion(masks_pred, true_masks, omask)
                loss_value = weighted_loss.item()
                scores.append(loss_value)
                print(f"[Rank {rank}] Eval batch {batch_idx+1}/{len(val_loader)} - Sample Loss: {loss_value:.6f}            ", end='\r')

    return sum(scores) / len(scores)

# ==== 参数解析 ====
def get_args():
    parser = argparse.ArgumentParser(description='Train with DDP')
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--batch-size', '-b', type=int, default=6)
    parser.add_argument('--lr', '-l', type=float, default=1e-4)
    parser.add_argument('--val', type=float, default=20.0)
    parser.add_argument('--amp', action='store_false', default=False)
    return parser.parse_args(args=[])

# ==== 主入口 ====
if __name__ == '__main__':
    args = get_args()
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size, args), nprocs=world_size, join=True)
