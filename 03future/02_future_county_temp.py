import os
import numpy as np
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.data_processing import remove_padding

future_root = 'your_directory/future_input'
output_root = 'your_directory/future_temp'
pmask = np.load('your_directory/bucket/dnstream/stats/pmask.npz', allow_pickle=True)['pmask']
county_mask_dict = np.load('your_directory/loads/county_masks_conus.npz')
county_codes = list(county_mask_dict.keys())

county_indices = {
    code: np.nonzero(county_mask_dict[code].astype(bool))
    for code in county_codes
}

scenarios = ['rcp45cooler', 'rcp45hotter', 'rcp85cooler', 'rcp85hotter']
years = ['2030', '2040', '2050']

county_indices = {
    code: np.nonzero(county_mask_dict[code].astype(bool))
    for code in county_codes
}

for scenario in scenarios:
    scenario_dir = os.path.join(future_root, scenario)
    all_files = sorted(glob(os.path.join(scenario_dir, 'input_*.npz')))

    files_by_year = {year: [] for year in years}
    for filepath in all_files:
        fname = os.path.basename(filepath)
        for year in years:
            if f'input_{year}' in fname:
                files_by_year[year].append(filepath)

    for year in years:
        files = sorted(files_by_year[year])
        if not files:
            print(f"[Skip] No files found for {scenario} {year}")
            continue

        print(f" Processing {scenario} {year}, files: {len(files)}")
        results = {code: [] for code in county_codes}
        def process_one_file(filepath):
            try:
                data = np.load(filepath)['input']  # (C, H, W)
                data = remove_padding(data, pmask)
                sixth_channel = data[5] 
                result = {}
                for code in county_codes:
                    r_idx, c_idx = county_indices[code]
                    values = sixth_channel[r_idx, c_idx]
                    values = values[~np.isnan(values)]
                    result[code] = values.mean() if len(values) > 0 else np.nan
                return filepath, result
            except Exception as e:
                print(f"[Warning] Failed to process {filepath}: {e}")
                return filepath, None


        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_one_file, f) for f in files]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{scenario}-{year}"):
                filepath, file_result = future.result()
                if file_result is None:
                    continue
                for code in county_codes:
                    results[code].append(file_result[code])

        outdir = os.path.join(output_root, scenario, year)
        os.makedirs(outdir, exist_ok=True)
        for code in county_codes:
            arr = np.array(results[code])  # shape (T,)
            np.savez_compressed(os.path.join(outdir, f'weather_{code}.npz'), data=arr)