import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

mapping_files = {
    "solar": "your_directory/gen/virtual_plant_mapping_solar_with_ratio_and_future_capacity.csv",
    "wind": "your_directory/gen/virtual_plant_mapping_wind_with_ratio_and_future_capacity.csv"
}
input_dir = "your_directory/future_gen/raw"
output_dir = "your_directory/future_gen/unit"

cached_maps = {}
def load_virtual_plant_map(tech, year):
    key = (tech, year)
    if key in cached_maps:
        return cached_maps[key]
    df = pd.read_csv(mapping_files[tech])
    valid_cols = [col for col in df.columns if f"_{year}" in col]
    cached_maps[key] = (df, valid_cols)
    return df, valid_cols

def process_single_file(npz_file, climate, year, plant_map, scenario_cols):
    basename = os.path.basename(npz_file)
    time_str = basename.replace("output_", "").replace(".npz", "")
    raw_data = np.load(npz_file)['data']  # shape: (2, H, W)
    H, W = raw_data.shape[1:]

    for col in scenario_cols['solar']:  
        scenario = col.replace(f"_{year}", "")
        output_data = np.zeros_like(raw_data)  # shape: (2, H, W)

        for tech_idx, tech in enumerate(['solar', 'wind']):
            df = plant_map[tech]
            lat_idx = df["lat_idx"].values.astype(int)
            lon_idx = df["lon_idx"].values.astype(int)
            capacity = df[col].values

            norm_values = raw_data[tech_idx, lat_idx, lon_idx]
            actual_values = norm_values * capacity

            for i in range(len(lat_idx)):
                output_data[tech_idx, lat_idx[i], lon_idx[i]] += actual_values[i]

        output_data = np.maximum(output_data, 0)

        save_dir = os.path.join(output_dir, climate, str(year), scenario)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"output_{time_str}.npz")
        np.savez_compressed(save_path, data=output_data.astype(np.float32))

def process(climate_scenarios, years, max_workers=8):
    for climate in climate_scenarios:
        for year in years:
            input_path = os.path.join(input_dir, climate, str(year))
            npz_files = glob(os.path.join(input_path, "output_*.npz"))
            if not npz_files:
                continue

            plant_map = {}
            scenario_cols = {}
            for tech in ['solar', 'wind']:
                df, valid_cols = load_virtual_plant_map(tech, year)
                plant_map[tech] = df
                scenario_cols[tech] = valid_cols

            print(f"Processing {climate} - {year} with {len(npz_files)} files...")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                process_func = partial(
                    process_single_file,
                    climate=climate,
                    year=year,
                    plant_map=plant_map,
                    scenario_cols=scenario_cols
                )
                list(tqdm(executor.map(process_func, npz_files), total=len(npz_files)))

climate_scenarios = ['rcp45cooler', 'rcp45hotter', 'rcp85cooler', 'rcp85hotter']
years = [2030, 2040, 2050]
process(climate_scenarios, years)
