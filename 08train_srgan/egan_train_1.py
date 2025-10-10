import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast
import torch.nn.functional as F
import numpy as np
import time
import os
from pathlib import Path
from tqdm.auto import tqdm
import gc

#####
from model.srgan import SRGAN_g,SRGAN_d
from utils.data_loading import WeatherData

##
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp

channel_group = 1  # 👈 Customize the channel group to use
year = ['2019','2020']
# ==== Global Paths ====
dir_input = Path('your_directory/upstream_low')
dir_output = Path('your_directory/dnstream/input')
checkpoint_dir = Path('.your_directory/result/weather')
metric_file = f'your_directory/dnstream/stats/statistics_summary.npz'
mask_file = f'your_directory/dnstream/stats/pmask.npz'
map_mask_file = f'your_directory/dnstream/stats/map_mask.npz'
log_pre_path = f'your_directory/result/weather{channel_group}/pre_train.log'
log_adv_path = f'your_directory/result/weather{channel_group}/adv_train.log'
os.makedirs(checkpoint_dir, exist_ok=True)
#########################################
data_metric = np.load(metric_file, allow_pickle=True)
pmask = np.load(mask_file, allow_pickle=True)['pmask']
output_mask = np.load(map_mask_file, allow_pickle=True)['output_mask']
input_metric, output_metric = data_metric['upstream_low_stats'].item(), data_metric['input_nopad_stats'].item()
input_mask = np.load(map_mask_file, allow_pickle=True)['upstream_mask']
##########################################
class RaGANLoss:
    def __init__(self):
        pass

    def discriminator_loss(self, real_logits, fake_logits):
        real_mean = torch.mean(fake_logits.detach())
        fake_mean = torch.mean(real_logits.detach())
        loss_real = F.mse_loss(real_logits - real_mean, torch.ones_like(real_logits))
        loss_fake = F.mse_loss(fake_logits - fake_mean, -torch.ones_like(fake_logits))
        return 0.5 * (loss_real + loss_fake)

    def generator_loss(self, real_logits, fake_logits):
        real_mean = torch.mean(fake_logits)
        fake_mean = torch.mean(real_logits)
        loss_real = F.mse_loss(real_logits - real_mean, -torch.ones_like(real_logits))
        loss_fake = F.mse_loss(fake_logits - fake_mean, torch.ones_like(fake_logits))
        return 0.5 * (loss_real + loss_fake)
#########################################
def gradient_loss(pred, target):
    grad_pred_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    grad_target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    grad_pred_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    grad_target_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(grad_pred_x, grad_target_x) + F.l1_loss(grad_pred_y, grad_target_y)

def tv_loss(x):
    return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

##########################################

def cleanup():
    dist.destroy_process_group()

################################################################
def train_srgan_ddp(
    G, D,
    rank, world_size,
    n_epoch_init=50,
    n_epoch=500,
    val_percent=0.2,
    batch_size=3,
    device="cuda",
    channel_group=1  # 👈 Added parameter
):
    os.environ['MASTER_ADDR'] = 'localhost'           # Or use the master node IP address
    os.environ['MASTER_PORT'] = '12355'               # An unused port number
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    dataset = WeatherData(
        dir_input, dir_output, year,
        input_metric, output_metric,
        pmask,
        input_mask=input_mask,
        channel_group=channel_group
    )
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val_set, sampler=val_sampler, drop_last=True, **loader_args)

    train_loader_adv = DataLoader(train_set, sampler=train_sampler, **loader_args)
    val_loader_adv = DataLoader(val_set, sampler=val_sampler, drop_last=True, **loader_args)

    # === Loss Functions & Optimizers ===
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    g_optimizer = optim.Adam(G.parameters(), lr=2e-4, betas=(0.9, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=1e-5, betas=(0.9, 0.999))
    g_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=n_epoch, eta_min=1e-6)
    d_scheduler = optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=n_epoch, eta_min=1e-6)

    # === Move models to GPU and wrap with DDP ===
    G = DDP(G.to(device), device_ids=[rank])
    D = DDP(D.to(device), device_ids=[rank])

    # === Logging (main process only) ===
    if rank == 0:
        os.makedirs(os.path.dirname(log_pre_path), exist_ok=True)
        with open(log_pre_path, 'w') as f:
            f.write('epoch,avg_train_loss,val_score\n')
        os.makedirs(os.path.dirname(log_adv_path), exist_ok=True)
        with open(log_adv_path, 'w') as f:
            f.write('epoch,avg_train_loss,eval_mse,eval_adv,eval_grad,total_val_loss\n')

    # === Stage 1: Adversarial Training ===
    print(f"[Rank {rank}] [Stage 2] Adversarial Training")
    ragan_loss = RaGANLoss()

    for epoch in range(n_epoch):
        G.train()
        D.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0
        d_epoch_loss = 0
        total = len(train_loader_adv.dataset)
        processed = 0

        for batch in train_loader_adv:
            lr_patch = batch['input_tensor'].to(device, non_blocking=True)
            hr_patch = batch['output_tensor'].to(device, non_blocking=True)

            # Discriminator step
            with torch.no_grad():
                fake_patch = G(lr_patch).detach()
            d_real_logits = D(hr_patch)
            d_fake_logits = D(fake_patch)
            d_loss = ragan_loss.discriminator_loss(d_real_logits, d_fake_logits)

            d_optimizer.zero_grad()
            d_loss.backward()
            clip_grad_norm_(D.parameters(), max_norm=1.0)
            d_optimizer.step()
            d_epoch_loss += d_loss.item()

            # Generator step
            fake_patch = G(lr_patch)
            d_fake_logits = D(fake_patch)
            d_real_logits = D(hr_patch.detach())
            adv_loss = ragan_loss.generator_loss(d_real_logits, d_fake_logits)
            pixel_loss = mse_loss(fake_patch, hr_patch)
            grad_loss = gradient_loss(fake_patch, hr_patch)
            shadow_loss = tv_loss(fake_patch)
            g_loss = pixel_loss + 1e-2 * adv_loss 

            g_optimizer.zero_grad()
            g_loss.backward()
            clip_grad_norm_(G.parameters(), max_norm=1.0)
            g_optimizer.step()
            epoch_loss += g_loss.item()

            processed += lr_patch.size(0)
            if rank == 0:
                print(f"[Epoch {epoch+1}/{n_epoch}] [Rank {rank}] {processed/total*2*100:.1f}%--d_loss: {d_loss:.6f}, pixel: {pixel_loss:.6f}, adv: {adv_loss:.6f}, grad: {grad_loss:.6f}, g_loss: {g_loss:.6f}",end="\r")

        # --- Aggregate G's loss ---
        epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        num_batches = torch.tensor(len(train_loader), device=device)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
        avg_train_loss = epoch_loss_tensor.item() / num_batches.item()

        # --- Aggregate D's loss ---
        d_epoch_loss_tensor = torch.tensor(d_epoch_loss, device=device)
        dist.all_reduce(d_epoch_loss_tensor, op=dist.ReduceOp.SUM)
        num_batches_adv = torch.tensor(len(train_loader_adv), device=device)
        dist.all_reduce(num_batches_adv, op=dist.ReduceOp.SUM)
        avg_d_loss = d_epoch_loss_tensor.item() / num_batches_adv.item()

        avg_mse, avg_adv, avg_grad, avg_shadow, total_val_loss = evaluate_adversarial(G, D, val_loader_adv, device, rank=0, amp=False)
        if rank == 0:
            with open(log_adv_path, 'a') as f:
                f.write(f"{epoch+1},{avg_d_loss:.6f},{avg_train_loss:.6f},{avg_mse:.6f},{avg_adv:.6f},{avg_grad:.6f},{avg_shadow:.6f},{total_val_loss:.6f}\n")
            torch.save(G.module.state_dict(), os.path.join(checkpoint_dir, f'G_epoch{epoch+1}.pth'))
            torch.save(D.module.state_dict(), os.path.join(checkpoint_dir, f'D_epoch{epoch+1}.pth'))

        g_scheduler.step()
        d_scheduler.step()


###########################################
def evaluate_adversarial(G, D, val_loader, device, rank=0, amp=False):
    G.eval()
    D.eval()

    # Initialize distributed tensors
    total_mse_loss = torch.tensor(0.0).to(device)
    total_adv_loss = torch.tensor(0.0).to(device)
    total_grad_loss = torch.tensor(0.0).to(device)
    total_tv_loss = torch.tensor(0.0).to(device)
    total_samples = torch.tensor(0.0).to(device)

    ragan_loss = RaGANLoss()

    with torch.no_grad():
        for batch in val_loader:
            lr = batch['input_tensor'].to(device, non_blocking=True)
            hr = batch['output_tensor'].to(device, non_blocking=True)
            batch_size = lr.shape[0]

            with torch.amp.autocast('cuda', enabled=amp):
                sr = G(lr)
                mse = F.mse_loss(sr, hr)
                d_fake_logits = D(sr)
                d_real_logits = D(hr)
                adv = ragan_loss.generator_loss(d_real_logits, d_fake_logits)
                grad = gradient_loss(sr, hr)
                shadow_loss = tv_loss(sr)

                total_mse_loss += mse * batch_size
                total_adv_loss += adv * batch_size
                total_grad_loss += grad * batch_size
                total_tv_loss += shadow_loss * batch_size
                total_samples += batch_size

    # Distributed synchronization
    for tensor in [total_mse_loss, total_adv_loss, total_grad_loss, total_tv_loss, total_samples]:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Average calculation
    avg_mse = total_mse_loss.item() / total_samples.item()
    avg_adv = total_adv_loss.item() / total_samples.item()
    avg_grad = total_grad_loss.item() / total_samples.item()
    avg_shadow = total_tv_loss.item() / total_samples.item()
    total_val_loss = avg_mse + 1e-2 * avg_adv 

    if rank == 0:
        print(f"\n[Eval-GAN] MSE: {avg_mse:.6f}, Adv: {avg_adv:.6f}, Grad: {avg_grad:.6f}, Shadow: {avg_shadow:.6f}, Total: {total_val_loss:.6f}")
    return avg_mse, avg_adv, avg_grad, avg_shadow, total_val_loss



def ddp_main(rank, world_size, channel_group):
    # Determine the number of output channels
    channel_map = {
        1: (2,),      # group 1 -> channels 1-2
        2: (4,),      # group 2 -> channels 3-6
        3: (2,),      # group 3 -> channels 7-8
    }
    if channel_group not in channel_map:
        raise ValueError(f"Invalid channel_group: {channel_group}, must be 1, 2, or 3")

    output_channels = channel_map[channel_group][0]
    input_channels = 7  # Assume your input is fixed (e.g., 7 meteorological variables)

    # Pass the number of channels when building the model
    G = SRGAN_g(in_channels=input_channels, out_channels=output_channels)
    D = SRGAN_d(in_channels=output_channels)

    train_srgan_ddp(G, D, rank, world_size, channel_group)
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(ddp_main, args=(world_size, channel_group), nprocs=world_size, join=True)