import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm.notebook import tqdm
import re

input_dir = "your_directory/future_gen/unit"  
solar_mapping_file = "your_directory/gen/virtual_plant_mapping_solar_with_ratio_and_future_capacity.csv"  
wind_mapping_file = "your_directory/gen/virtual_plant_mapping_wind_with_ratio_and_future_capacity.csv"  
output_dir = "your_directory/future_gen/subregion" 

def load_virtual_plant_map(mapping_file):
    df = pd.read_csv(mapping_file)
    return df

def get_planning_scenarios(climate, year):
    path = os.path.join(input_dir, climate, str(year))
    planning_scenarios = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return planning_scenarios

def sort_files_by_time(npz_files):
    def extract_time_from_filename(filename):
        match = re.search(r'output_(\d{8})(\d{2})\.npz', filename)
        if match:
            date_str = match.group(1)  # yyyymmdd
            hour_str = match.group(2)  # hh
            return f"{date_str}{hour_str}" 
        return ""
    return sorted(npz_files, key=extract_time_from_filename)

climate_scenarios = ["rcp45cooler", "rcp45hotter", "rcp85cooler", "rcp85hotter"]
years = [2030, 2040, 2050]

solar_virtual_plant_map = load_virtual_plant_map(solar_mapping_file)
wind_virtual_plant_map = load_virtual_plant_map(wind_mapping_file)

os.makedirs(output_dir, exist_ok=True)

for climate in climate_scenarios:
    for year in years:
        planning_scenarios = get_planning_scenarios(climate, year)

        for planning in planning_scenarios:
            input_path = os.path.join(input_dir, climate, str(year), planning)
            npz_files = glob(os.path.join(input_path, "output_*.npz"))

            npz_files = sort_files_by_time(npz_files)

            subregion_generation = {}

            hour = 0 
            for npz_file in npz_files:
                data = np.load(npz_file)['data']
                solar_gen = data[0]  
                wind_gen = data[1]   

                for idx, row in solar_virtual_plant_map.iterrows():
                    subregion_code = row['subregion']
                    plant_lat = row['lat_idx']
                    plant_lon = row['lon_idx']
                    solar_plant_gen = solar_gen[plant_lat][plant_lon]  
                    if subregion_code not in subregion_generation:
                        subregion_generation[subregion_code] = {'solar': np.zeros(8760), 'wind': np.zeros(8760)}
                    subregion_generation[subregion_code]['solar'][hour] += solar_plant_gen / 1000.0  

                for idx, row in wind_virtual_plant_map.iterrows():
                    subregion_code = row['subregion']
                    plant_lat = row['lat_idx']
                    plant_lon = row['lon_idx']
                    wind_plant_gen = wind_gen[plant_lat][plant_lon]  
                    if subregion_code not in subregion_generation:
                        subregion_generation[subregion_code] = {'solar': np.zeros(8760), 'wind': np.zeros(8760)}
                    subregion_generation[subregion_code]['wind'][hour] += wind_plant_gen / 1000.0  
                print(f"\rProcessing: {climate} {year} {planning} - Hour {hour + 1}", end="")

                hour += 1
                if hour >= 8760:
                    break

            for subregion_code, generation_data in subregion_generation.items():
                subregion_output_dir = os.path.join(output_dir, climate, str(year), planning)
                os.makedirs(subregion_output_dir, exist_ok=True)

                output_file = os.path.join(subregion_output_dir, f"subregion_{subregion_code}_generation.npz")
                np.savez_compressed(output_file, solar=generation_data['solar'], wind=generation_data['wind'])
                print(f"Saved generation data for subregion {subregion_code} to {output_file}")