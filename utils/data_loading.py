import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.data_processing import remove_padding


#### Dataset class for renewable #######
class EnergyData(Dataset):
    def __init__(self, input_folder, output_folder, year, input_metric, output_metric, transform=None):
        """
        Args:
            input_folder (str): The folder containing the input data, file names should be in the format input_YYYYMMDDHH.npz
            output_folder (str): The folder containing the output data, file names should be in the format output_YYYYMMDDHH.npz
            year (str or list): The years to filter, for example "2023" or ["2018", "2019"]
            input_metric (list or np.array): Normalization factors for input data, one for each channel
            output_metric (list or np.array): Normalization factors for output data, one for each channel
            transform (callable, optional): An optional preprocessing function for the input tensor
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.year = year if isinstance(year, list) else [year]  
        self.input_metric = input_metric
        self.output_metric = output_metric
        self.transform = transform

        self.input_files = []
        for f in os.listdir(input_folder):
            if f.startswith("input_") and f.endswith(".npz"):
                file_year = f[len("input_"):len("input_") + 4]  
                if file_year in self.year:
                    self.input_files.append(f)
        
        self.input_files.sort()  

        self.output_files = []
        for f in self.input_files:
            date_time_str = f[len("input_"):-len(".npz")]
            output_filename = f"output_{date_time_str}.npz"
            output_path = os.path.join(output_folder, output_filename)
            if os.path.exists(output_path):
                self.output_files.append(output_filename)
            else:
                raise FileNotFoundError(f"Corresponding output file {output_filename} does not exist")
        
        if len(self.input_files) != len(self.output_files):
            raise ValueError("The number of input and output files do not match!")
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        input_path = os.path.join(self.input_folder, self.input_files[idx])
        output_path = os.path.join(self.output_folder, self.output_files[idx])
        
        input_data = np.load(input_path)
        output_data = np.load(output_path)
        input_arr = list(input_data.values())[0]
        output_arr = list(output_data.values())[0]

        # Step 1: Replace -1 with NaN
        input_arr = np.where(input_arr == -1, np.nan, input_arr)
        output_arr = np.where(output_arr == -1, np.nan, output_arr)

        # Step 2: Normalize (preserving NaN)
        for ch in range(input_arr.shape[0]):
            input_arr[ch] = (input_arr[ch] - self.input_metric['mean'][ch]) / self.input_metric['std'][ch]
        for ch in range(output_arr.shape[0]):
            output_arr[ch] = (output_arr[ch] - self.output_metric['mean'][ch]) / self.output_metric['std'][ch]

        # Step 3: Replace normalized NaN with 0
        input_arr = np.nan_to_num(input_arr, nan=0.0)
        output_arr = np.nan_to_num(output_arr, nan=0.0)

        input_tensor = torch.tensor(input_arr, dtype=torch.float32)
        output_tensor = torch.tensor(output_arr, dtype=torch.float32)
        
        if self.transform:
            input_tensor = self.transform(input_tensor)
        
        return {
            'input_tensor': input_tensor,
            'output_tensor': output_tensor,
            'input_path': input_path
        }

####### Dataset for Fires ########
class FireData(Dataset):
    def __init__(self, input_folder, output_folder, year, input_metric, output_metric, transform=None):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.year = year if isinstance(year, list) else [year]  # Convert to list if not already a list
        self.input_metric = input_metric
        self.output_metric = output_metric
        self.transform = transform

        # Filter the npz files that correspond to the specified years
        self.input_files = []
        for f in os.listdir(input_folder):
            if f.startswith("input_") and f.endswith(".npz"):
                file_year = f[len("input_"):len("input_") + 4]  # Extract the year part
                if file_year in self.year:
                    self.input_files.append(f)
        
        self.input_files.sort()  # Ensure the files are in order

        # Construct corresponding output files based on input file names
        self.output_files = []
        for f in self.input_files:
            date_time_str = f[len("input_"):-len(".npz")]
            output_filename = f"hazard_{date_time_str}.npz"
            output_path = os.path.join(output_folder, output_filename)
            if os.path.exists(output_path):
                self.output_files.append(output_filename)
            else:
                raise FileNotFoundError(f"Corresponding output file {output_filename} does not exist")
        
        if len(self.input_files) != len(self.output_files):
            raise ValueError("The number of input and output files do not match!")
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        # Construct the full path for input and output files
        input_path = os.path.join(self.input_folder, self.input_files[idx])
        output_path = os.path.join(self.output_folder, self.output_files[idx])
        
        # Load the npz file data (assuming there's only one array, or take the first array)
        input_data = np.load(input_path)
        output_data = np.load(output_path)
        input_arr = list(input_data.values())[0]
        output_arr = list(output_data.values())[0]
        
        # Replace NaN with -1
        input_arr = np.nan_to_num(input_arr, nan=-1)
        output_arr = np.nan_to_num(output_arr, nan=-1)
        
        # Only take the first channel of the output data
        output_arr = output_arr[0:1]  # Take the first channel

        for ch in range(input_arr.shape[0]):
            input_arr[ch] = (input_arr[ch] - self.input_metric['mean'][ch]) / self.input_metric['std'][ch]
        # Normalize the first channel of the output data
        output_arr = (output_arr - self.output_metric['mean'][0]) / self.output_metric['std'][0]
        
        # Convert to torch tensor
        input_tensor = torch.tensor(input_arr, dtype=torch.float32)
        output_tensor = torch.tensor(output_arr, dtype=torch.float32)
        
        # Apply the optional transform function, if provided
        if self.transform:
            input_tensor = self.transform(input_tensor)
        
        return {
            'input_tensor': input_tensor,
            'output_tensor': output_tensor,
            'input_path': input_path
        }

####### Dataset for Hurricane ########
class HurricaneData(Dataset):
    def __init__(self, input_folder, output_folder, year, input_metric, output_metric, transform=None):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.year = year if isinstance(year, list) else [year]  # Convert to list if not already a list
        self.input_metric = input_metric
        self.output_metric = output_metric
        self.transform = transform

        # Filter the npz files that correspond to the specified years
        self.input_files = []
        for f in os.listdir(input_folder):
            if f.startswith("input_") and f.endswith(".npz"):
                file_year = f[len("input_"):len("input_") + 4]  # Extract the year part
                if file_year in self.year:
                    self.input_files.append(f)
        
        self.input_files.sort()  # Ensure the files are in order

        # Construct corresponding output files based on input file names
        self.output_files = []
        for f in self.input_files:
            date_time_str = f[len("input_"):-len(".npz")]
            output_filename = f"hazard_{date_time_str}.npz"
            output_path = os.path.join(output_folder, output_filename)
            if os.path.exists(output_path):
                self.output_files.append(output_filename)
            else:
                raise FileNotFoundError(f"Corresponding output file {output_filename} does not exist")
        
        if len(self.input_files) != len(self.output_files):
            raise ValueError("The number of input and output files do not match!")
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        # Construct the full path for input and output files
        input_path = os.path.join(self.input_folder, self.input_files[idx])
        output_path = os.path.join(self.output_folder, self.output_files[idx])
        
        # Load the npz file data (assuming there's only one array, or take the first array)
        input_data = np.load(input_path)
        output_data = np.load(output_path)
        input_arr = list(input_data.values())[0]
        output_arr = list(output_data.values())[0]
        
        # Replace NaN with -1
        input_arr = np.nan_to_num(input_arr, nan=-1)
        output_arr = np.nan_to_num(output_arr, nan=-1)

        for ch in range(input_arr.shape[0]):
            input_arr[ch] = (input_arr[ch] - self.input_metric['mean'][ch]) / self.input_metric['std'][ch]
        
        # Only take the second channel of the output data
        output_arr = output_arr[1:2]
        
        # Convert to torch tensor
        input_tensor = torch.tensor(input_arr, dtype=torch.float32)
        output_tensor = torch.tensor(output_arr, dtype=torch.float32)
        
        # Apply the optional transform function, if provided
        if self.transform:
            input_tensor = self.transform(input_tensor)
        
        return {
            'input_tensor': input_tensor,
            'output_tensor': output_tensor,
            'input_path': input_path
        }
    
class WeatherData(Dataset):
    def __init__(self, input_folder, output_folder, year, input_metric, output_metric, pmask, input_mask, transform=None, channel_group=1):
        """
        Args:
            input_folder (str): Folder containing the input data, file names like input_YYYYMMDDHH.npz
            output_folder (str): Folder containing the output data, file names like output_YYYYMMDDHH.npz
            year (str or list): Years to filter, e.g., "2023" or ["2018", "2019"]
            input_metric (dict): Normalization for input data, with keys 'mean' and 'std'
            output_metric (dict): Normalization for output data, with keys 'mean' and 'std'
            transform (callable, optional): Optional preprocessing function for the input tensor
            channel_group (int): Group index for selecting specific output channels (1: ch1-2, 2: ch3-6, 3: ch7-9)
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.year = year if isinstance(year, list) else [year]  
        self.input_metric = input_metric
        self.output_metric = output_metric
        self.transform = transform
        self.pmask = pmask
        self.input_mask = input_mask
        self.channel_group = channel_group

        self.input_files = []
        for f in os.listdir(input_folder):
            if f.startswith("initial_") and f.endswith(".npz"):
                datetime_str = f[len("initial_"):len("initial_") + 10]  
                if any(datetime_str.startswith(prefix) for prefix in self.year):
                    self.input_files.append(f)
        self.input_files.sort()

        self.output_files = []
        for f in self.input_files:
            date_time_str = f[len("initial_"):-len(".npz")]
            output_filename = f"input_{date_time_str}.npz"
            output_path = os.path.join(output_folder, output_filename)
            if os.path.exists(output_path):
                self.output_files.append(output_filename)
            else:
                raise FileNotFoundError(f"Corresponding output file {output_filename} does not exist")
        
        if len(self.input_files) != len(self.output_files):
            raise ValueError("The number of input and output files do not match!")

        # Set the output channel indices for each group
        self.channel_map = {
            1: list(range(0, 2)),   # Channels 1-2 (index 0,1)
            2: list(range(2, 6)),   # Channels 3-6 (index 2,3,4,5)
            3: list(range(6, 8)),   # Channels 7-9 (index 6,7)
        }
        if self.channel_group not in self.channel_map:
            raise ValueError(f"Invalid channel_group: {self.channel_group}, must be 1, 2, or 3")
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        input_path = os.path.join(self.input_folder, self.input_files[idx])
        output_path = os.path.join(self.output_folder, self.output_files[idx])
        
        input_data = np.load(input_path)
        output_data = np.load(output_path)
        input_arr = list(input_data.values())[0]
        output_arr = list(output_data.values())[0]

        # Step 1: Replace -1 with NaN (to preserve missing value information for subsequent normalization)
        input_arr = np.where(input_arr == -1, np.nan, input_arr)
        output_arr = np.where(output_arr == -1, np.nan, output_arr)
        input_arr[self.input_mask == 0] = np.nan

        # Step 2: Apply padding mask to remove padding area
        output_arr = remove_padding(output_arr, self.pmask)

        # Step 3: Normalize input
        for ch in range(input_arr.shape[0]):
            input_arr[ch] = (input_arr[ch] - self.input_metric['mean'][ch]) / self.input_metric['std'][ch]

        # Step 4: Select output channels and normalize
        selected_channels = self.channel_map[self.channel_group]
        output_arr = output_arr[selected_channels]
        for i, ch in enumerate(selected_channels):
            output_arr[i] = (output_arr[i] - self.output_metric['mean'][ch]) / self.output_metric['std'][ch]

        # Step 5: Replace normalized NaN with 0
        input_arr = np.nan_to_num(input_arr, nan=0.0)
        output_arr = np.nan_to_num(output_arr, nan=0.0)

        input_tensor = torch.tensor(input_arr, dtype=torch.float32)
        output_tensor = torch.tensor(output_arr, dtype=torch.float32)
        
        if self.transform:
            input_tensor = self.transform(input_tensor)
        
        return {
            'input_tensor': input_tensor,
            'output_tensor': output_tensor,
            'input_path': input_path
        }

##################################################################################
class FutureDataset(Dataset):
    def __init__(self, input_folder, year, input_metric, channel_mask, transform=None):
        """
        Args:
            input_folder (str): Folder containing input files like future_YYYYMMDDHH.npz
            year (str or list): Year(s) of data to include, e.g., "2023" or ["2020", "2021"]
            input_metric (dict): Dict with 'mean' and 'std' for normalization, each is list/array of length C
            channel_mask (ndarray): Array of shape (C, H, W), 0 means masked and value should be set to -1
            transform (callable, optional): Optional transform to apply to the input tensor
        """
        self.input_folder = input_folder
        self.year = year if isinstance(year, list) else [year]
        self.input_metric = input_metric
        self.channel_mask = channel_mask
        self.transform = transform

        self.input_files = []
        for f in os.listdir(input_folder):
            if f.startswith("future_") and f.endswith(".npz"):
                date_str = f[len("future_"):len("future_") + 10]
                if any(date_str.startswith(prefix) for prefix in self.year):
                    self.input_files.append(f)
        self.input_files.sort()

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_folder, self.input_files[idx])
        input_data = np.load(input_path)
        input_arr = list(input_data.values())[0]  # shape: (C, H, W)

        input_arr = np.nan_to_num(input_arr, nan=-1.0)
        # === convert units ===
        if input_arr.shape[0] > 4:
            input_arr[3] = input_arr[3] / 100.0
        if input_arr.shape[0] > 5:
            input_arr[4] = (input_arr[4] - 273.15) * 10.0
        # Normalize
        for ch in range(input_arr.shape[0]):
            input_arr[ch] = (input_arr[ch] - self.input_metric['mean'][ch]) / self.input_metric['std'][ch]
        # Apply channel mask: where mask==0, set corresponding input to -1
        masked_input = np.where(self.channel_mask == 0, 0, input_arr)
        input_tensor = torch.tensor(masked_input, dtype=torch.float32)
        if self.transform:
            input_tensor = self.transform(input_tensor)

        return {
            'input_tensor': input_tensor,
            'input_path': input_path
        }
    
#### Dataset class for future downscale data #######
class FutureEnergyData(Dataset):
    def __init__(self, input_folder, year, input_metric, input_mask, transform=None):
        """
        Args:
            input_folder (str): The folder containing the input data, file names should be in the format input_YYYYMMDDHH.npz
            year (str or list): The years to filter, for example "2023" or ["2018", "2019"]
            input_metric (list or np.array): Normalization factors for input data, one for each channel
            transform (callable, optional): An optional preprocessing function for the input tensor
        """
        self.input_folder = input_folder
        self.year = year if isinstance(year, list) else [year]  
        self.input_metric = input_metric
        self.transform = transform
        self.input_mask = input_mask

        self.input_files = []
        for f in os.listdir(input_folder):
            if f.startswith("input_") and f.endswith(".npz"):
                file_year = f[len("input_"):len("input_") + 4]  
                if file_year in self.year:
                    self.input_files.append(f)
        
        self.input_files.sort()  
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        input_path = os.path.join(self.input_folder, self.input_files[idx])
        
        input_data = np.load(input_path)
        input_arr = list(input_data.values())[0]

        # Step 2: Normalize (preserving NaN)
        for ch in range(input_arr.shape[0]):
            input_arr[ch] = (input_arr[ch] - self.input_metric['mean'][ch]) / self.input_metric['std'][ch]

        masked_input = np.where(self.input_mask == 0, 0, input_arr)
        input_tensor = torch.tensor(masked_input, dtype=torch.float32)

        if self.transform:
            input_tensor = self.transform(input_tensor)
        
        return {
            'input_tensor': input_tensor,
            'input_path': input_path
        }