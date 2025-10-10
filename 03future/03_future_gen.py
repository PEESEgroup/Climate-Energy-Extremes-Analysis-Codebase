import os
import re
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import segmentation_models_pytorch as smp

from utils.data_loading import FutureEnergyData
from utils.data_processing import remove_padding, inverse_normalize_energy

def load_model_checkpoint(model, checkpoint_name):
    checkpoint_path = os.path.join("your_directory", checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found.")
    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model

def match_date(dataset, target_date):
    for index, file_name in enumerate(dataset.input_files):
        if isinstance(file_name, str) and re.search(f'input_{target_date}.npz$', file_name):
            return index
    return None

def get_future_output(model, dataset, target_date, device):
    model.to(device)
    index = match_date(dataset, target_date)
    if index is None:
        raise ValueError(f"No targeted file {target_date} found")
    sample = dataset[index]
    inputs = sample['input_tensor'].unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(inputs)
    return inputs.cpu(), outputs.cpu(), sample['input_path']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = smp.Unet(encoder_name='resnet152', in_channels=8, classes=2).to(memory_format=torch.channels_last).to(device)
model = load_model_checkpoint(model, "checkpoint_epoch100.pth")

metric_file = 'your_directory/bucket/dnstream/stats/statistics_summary.npz'
data_metric = np.load(metric_file, allow_pickle=True)
input_metric = data_metric['input_nopad_stats'].item()
output_metric = data_metric['output_stats'].item()

map_mask_file = 'your_directory/bucket/dnstream/stats/map_mask.npz'
map_mask = np.load(map_mask_file, allow_pickle=True)
input_mask = map_mask['input_mask']
output_mask = map_mask['output_mask']
omask = output_mask[:2]

pmask = np.load('your_directory/bucket/dnstream/stats/pmask.npz', allow_pickle=True)['pmask']

root_input = Path('your_directory/future_input')
root_output = Path('your_directory/future_gen/raw')
climate_scenarios = [d.name for d in root_input.iterdir() if d.is_dir()]

for scenario in climate_scenarios:
    for year in ['2030', '2040', '2050']:
        dir_input = root_input / scenario
        dataset = FutureEnergyData(dir_input, [year], input_metric, input_mask)
        print(f"[{scenario}-{year}] total files: {len(dataset)}")

        for idx in tqdm(range(len(dataset)), desc=f"Processing {scenario}-{year}"):
            sample = dataset[idx]
            input_path = sample['input_path']
            match = re.search(r'input_(\d{10})\.npz$', input_path)
            if not match:
                continue
            timestamp = match.group(1)
            try:
                image, pred, _ = get_future_output(model, dataset, timestamp, device)
                input_array = remove_padding(image.squeeze(0).numpy(), pmask)
                pred_array = pred * omask
                pred_array = inverse_normalize_energy(pred_array, output_metric)
                pred_array = pred_array * np.where(omask == 0, np.nan, omask)
                pred_array = remove_padding(pred_array.squeeze(0).numpy(), pmask)

                out_dir = root_output / scenario / year
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"output_{timestamp}.npz"
                np.savez_compressed(out_path, data=pred_array)
            except Exception as e:
                print(f"Failed processing {input_path}: {e}")