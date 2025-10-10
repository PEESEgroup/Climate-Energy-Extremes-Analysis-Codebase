#%%
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

base_load_dir = 'your_directory/loads/new'
rate_base_dir = 'your_directory/future_loads/rates'
temp_base_dir = 'your_directory/future_temp'
fitting_com_dir = 'your_directory/loads/com_fitting'
fitting_res_dir = 'your_directory/loads/res_fitting'
output_base_dir = 'your_directory/future_loads/new'

years = [2030, 2040, 2050]
climate_scenarios = os.listdir(temp_base_dir)
#%%
def apply_fit_hourly(params_24h, threshold_24h, temp_series, mode):

    hours = len(temp_series)
    days = hours // 24
    result = np.zeros(hours, dtype=np.float32)
    for d in range(days):
        for h in range(24):
            idx = d * 24 + h
            param = params_24h[h]
            thres = threshold_24h[h]
            t = temp_series[idx]


            if (mode == 'heating' and t >= thres) or (mode == 'cooling' and t <= thres):
                result[idx] = 0.0
            else:
                result[idx] = param[0] * t**2 + param[1] * t + param[2]
    return result

#%%
for year in years:
    for climate in climate_scenarios:
        climate_year_dir = os.path.join(temp_base_dir, climate, str(year))
        if not os.path.isdir(climate_year_dir):
            continue
        weather_files = [f for f in os.listdir(climate_year_dir) if f.startswith("weather_")]
        for weather_file in tqdm(weather_files, desc=f'{climate}-{year}'):
            county_code = weather_file.replace("weather_", "").replace(".npz", "")
            weather_path = os.path.join(climate_year_dir, weather_file)
            temperature = np.load(weather_path)['data'].astype('float32')
            temperature = temperature[:8760]

            load_path = os.path.join(base_load_dir, f'load_{county_code}.npz')
            if not os.path.exists(load_path):
                continue
            load_data = np.load(load_path)
            com_total_2018 = load_data['com_total'] / 1000.0
            com_heating_2018 = load_data['com_heating'] / 1000.0
            com_cooling_2018 = load_data['com_cooling'] / 1000.0
            res_total_2018 = load_data['res_total'] / 1000.0
            res_heating_2018 = load_data['res_heating'] / 1000.0
            res_cooling_2018 = load_data['res_cooling'] / 1000.0
            ind_trans_2018 = load_data['ind_trans'] / 1000.0

            fit_file_com = os.path.join(fitting_com_dir, f'fitting_{county_code}.npz')
            fit_file_res = os.path.join(fitting_res_dir, f'fitting_{county_code}.npz')
            if not os.path.exists(fit_file_com) or not os.path.exists(fit_file_res):
                continue
            com_fit = np.load(fit_file_com)
            res_fit = np.load(fit_file_res)

            for planning_scenario in os.listdir(os.path.join(rate_base_dir, str(year))):
                rate_file = os.path.join(rate_base_dir, str(year), planning_scenario, f'rate_{county_code}.npz')
                if not os.path.exists(rate_file):
                    continue
                growth_rate = np.load(rate_file)['rate'].item()

                # === heating / cooling ===
                com_heating_future = apply_fit_hourly(com_fit['heating'], com_fit['heating_threshold'], temperature, mode='heating')
                com_heating_future *= growth_rate

                com_cooling_future = apply_fit_hourly(com_fit['cooling'], com_fit['cooling_threshold'], temperature, mode='cooling')
                com_cooling_future *= growth_rate

                res_heating_future = apply_fit_hourly(res_fit['heating'], res_fit['heating_threshold'], temperature, mode='heating')
                res_heating_future *= growth_rate

                res_cooling_future = apply_fit_hourly(res_fit['cooling'], res_fit['cooling_threshold'], temperature, mode='cooling')
                res_cooling_future *= growth_rate

                # === other load ===
                com_other = com_total_2018 - com_heating_2018 - com_cooling_2018
                res_other = res_total_2018 - res_heating_2018 - res_cooling_2018
                com_total_future = com_other * growth_rate + com_heating_future + com_cooling_future
                res_total_future = res_other * growth_rate + res_heating_future + res_cooling_future

                ind_trans_future = ind_trans_2018 * growth_rate

                out_dir = os.path.join(output_base_dir, climate, str(year), planning_scenario)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f'load_{county_code}.npz')
                np.savez_compressed(out_path,
                                    com_total=com_total_future,
                                    com_heating=com_heating_future,
                                    com_cooling=com_cooling_future,
                                    res_total=res_total_future,
                                    res_heating=res_heating_future,
                                    res_cooling=res_cooling_future,
                                    ind_trans=ind_trans_future)
