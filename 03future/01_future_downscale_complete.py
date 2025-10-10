import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from model.srgan import SRGAN_g
from utils.data_loading import FutureDataset
from utils.data_processing import inverse_normalize_weather, pad_to_multiple_of_32
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GaussianBlur(nn.Module):
    def __init__(self, channels, kernel_size=5, sigma=1.0):
        super().__init__()
        self.padding = kernel_size // 2
        self.groups = channels
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
        self.register_buffer('weight', kernel)
    def _create_gaussian_kernel(self, k, sigma):
        ax = torch.arange(-k // 2 + 1., k // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)
    def forward(self, x):
        return F.conv2d(x, weight=self.weight, padding=self.padding, groups=self.groups)

def load_model(checkpoint_path, out_channels):
    model = SRGAN_g(out_channels=out_channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get('state_dict', checkpoint))
    model.eval()
    return model

def process_sample(idx, dataset, model1, model2, model3, output_mask, output_metric, output_dir):
    try:
        sample = dataset[idx]
        input_tensor = sample['input_tensor'].clone().detach().unsqueeze(0).to(device)
        with torch.no_grad():
            out1 = model1(input_tensor)
            out2 = model2(input_tensor)
            out3 = model3(input_tensor)
            blur = GaussianBlur(out3.shape[1]).to(device)
            out3 = blur(out3)
        output_tensor = torch.cat([out1, out2, out3], dim=1).squeeze(0).cpu().numpy()
        output_tensor = inverse_normalize_weather(output_tensor, output_metric, 0)
        output_tensor, _ = pad_to_multiple_of_32(output_tensor)
        output_tensor[6:8] = np.maximum(output_tensor[6:8], 0)
        output_tensor[output_mask == 0] = -1
        output_tensor = output_tensor.astype(np.float16)
        filename = os.path.basename(sample['input_path']).replace("future_", "input_")
        np.savez_compressed(output_dir / filename, input=output_tensor)
    except Exception as e:
        print(f"[Error @ idx={idx}]: {e}")

def process_scenario(scenario_dir):
    scenario_name = os.path.basename(scenario_dir)
    print(f"\n Processing scenario: {scenario_name}")
    output_dir = Path(f"your_directory/{scenario_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # load stats/mask
    stats = np.load("your_directory/statistics_summary.npz", allow_pickle=True)
    input_metric = stats['upstream_low_stats'].item()
    output_metric = stats['input_nopad_stats'].item()
    mask = np.load("your_directory/map_mask.npz", allow_pickle=True)
    input_mask = mask['upstream_mask']
    output_mask = mask['input_mask']

    dataset = FutureDataset(Path(scenario_dir), ['2030', '2040', '2050'], input_metric, input_mask)
    print(f"Total samples: {len(dataset)}")

    model1 = load_model("your_directory/G_epoch30.pth", 2)
    model2 = load_model("your_directory/G_epoch70.pth", 4)
    model3 = load_model("your_directory/G_epoch49.pth", 2)

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
        futures = [executor.submit(
            process_sample, idx, dataset, model1, model2, model3,
            output_mask, output_metric, output_dir
        ) for idx in range(len(dataset))]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"{scenario_name}"):
            pass

def main():
    scenario_dirs = [
        "your_directory/future/rcp45cooler",
        "your_directory/future/rcp45hotter",
        "your_directory/future/rcp85cooler",
        "your_directory/future/rcp85hotter",
    ]
    for scenario in scenario_dirs:
        process_scenario(scenario)

if __name__ == "__main__":
    main()
