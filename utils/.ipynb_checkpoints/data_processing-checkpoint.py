import os
import numpy as np
import re
import concurrent.futures
from tqdm import tqdm

mean_file = "/workspace/climate/water_mean.npz"

def normalize_water_data(input_data):
    """
    归一化输入数据中的 'lake' 和 'river' 键，
    仅对非 -1 的元素进行归一化，并直接修改输入数据
    """
    try:
        mean_data = np.load(mean_file)
        lake_mean = mean_data['lake']
        river_mean = mean_data['river']
    except Exception as e:
        print(f"Error loading mean file: {e}")
        return None

    output_lake = np.full_like(input_data['lake'], -1)
    output_river = np.full_like(input_data['river'], -1)
    output_data = dict(input_data)
    
    if 'lake' in input_data and 'river' in input_data:

        valid_lake_mask = np.where(np.logical_and(input_data['lake'] != -1, lake_mean != 0))
        valid_river_mask = np.where(np.logical_and(input_data['river'] != -1, river_mean != 0))
        
        # 只对非 -1 的元素进行归一化，确保索引匹配
        output_lake[valid_lake_mask] /= lake_mean[valid_lake_mask]
        output_river[valid_river_mask] /= river_mean[valid_river_mask]
        
        output_data['lake'] = output_lake
        output_data['river'] = output_river
    else:
        print("Input data missing required keys.")
        
    return output_data

def pad_to_multiple_of_32(data):
    """
    Pad the input 3D array (channel * height * width) so that height and width
    are multiples of 32. The added padding values are set to -1.
    
    Parameters:
        data (numpy.ndarray): Input array with shape (C, H, W)
    
    Returns:
        tuple: (padded array, padding mask)
    """
    if not isinstance(data, np.ndarray) or len(data.shape) != 3:
        raise ValueError("Input data must be a 3D numpy array with shape (C, H, W)")
    
    C, H, W = data.shape
    
    # Compute the new dimensions (multiples of 32)
    new_H = ((H + 31) // 32) * 32
    new_W = ((W + 31) // 32) * 32
    
    # Create a new array with -1 padding
    padded_data = np.full((C, new_H, new_W), fill_value=-1, dtype=data.dtype)
    
    # Create a mask where 1 indicates original data and 0 indicates padding
    padding_mask = np.zeros((new_H, new_W), dtype=np.uint8)
    padding_mask[:H, :W] = 1
    
    # Copy the original data into the padded array
    padded_data[:, :H, :W] = data
    
    return padded_data, padding_mask

def remove_padding(padded_data, padding_mask):
    """
    Remove padding from the input 3D array using the given padding mask.
    
    Parameters:
        padded_data (numpy.ndarray): Padded array with shape (C, H_padded, W_padded)
        padding_mask (numpy.ndarray): Mask with shape (H_padded, W_padded)
    
    Returns:
        numpy.ndarray: Trimmed array with original shape before padding.
    """
    if not isinstance(padded_data, np.ndarray) or len(padded_data.shape) != 3:
        raise ValueError("padded_data must be a 3D numpy array with shape (C, H, W)")
    if not isinstance(padding_mask, np.ndarray) or len(padding_mask.shape) != 2:
        raise ValueError("padding_mask must be a 2D numpy array")
    
    # Find the original dimensions
    H, W = np.where(padding_mask)
    min_H, max_H = H.min(), H.max()
    min_W, max_W = W.min(), W.max()
    
    return padded_data[:, min_H:max_H+1, min_W:max_W+1]

def inverse_normalize(tensor, metric):
    """
    将输入的tensor转换为numpy array，并根据给定的metric字典进行逆归一化。
    
    参数:
        tensor: 输入的tensor，形状为 (channels, ...)。
        metric: 字典，包含 'min' 和 'max' 两个key，值为每个通道对应的最小值和最大值列表。
                例如: {'min': [min_ch1, min_ch2, ...], 'max': [max_ch1, max_ch2, ...]}
                
    返回:
        逆归一化后的numpy数组。
    """
    # 如果是PyTorch的tensor，则转换为numpy数组
    arr = tensor
    
    # 对每个通道执行逆归一化操作：value = normalized_value * (max - min) + min
    for ch in range(arr.shape[0]):
        #arr[ch] = tensor[ch] * (metric['max'][ch] - metric['min'][ch]) + metric['min'][ch]
        arr[ch] = tensor[ch] * (metric['std'][ch]) + metric['mean'][ch]
        
    return arr