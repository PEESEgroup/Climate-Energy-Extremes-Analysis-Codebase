import os
import numpy as np
import re
import concurrent.futures
from tqdm import tqdm

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

def inverse_normalize_energy(tensor, metric):
    arr = tensor
    for ch in range(arr.shape[0]):
        arr[ch] = tensor[ch] * (metric['std'][ch]) + metric['mean'][ch]
    return arr

def inverse_normalize_fire(tensor, metric):
    arr = tensor[0:1]
    for ch in range(arr.shape[0]):
        arr[ch] = tensor[ch] * (metric['std'][ch]) + metric['mean'][ch]
    return arr

def inverse_normalize_weather(tensor, metric):
    arr = tensor
    for ch in range(arr.shape[0]):
        arr[ch] = (tensor[ch] + 1) / 2 * (metric['max'][ch] - metric['min'][ch]) + metric['min'][ch]
    return arr