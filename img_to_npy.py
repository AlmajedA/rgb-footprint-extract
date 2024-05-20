import numpy as np
from PIL import Image
import os

def jpg_to_npy(jpg_path, npy_path):
    """
    Converts a JPG image to a NumPy array and saves it as a .npy file.
    
    Args:
        jpg_path (str): The path to the input JPG image.
        npy_path (str): The path to the output .npy file.
    """
    # Open the JPG image and convert it to a NumPy array
    image = Image.open(jpg_path)
    image_array = np.array(image)
    
    # Save the NumPy array as a .npy file
    np.save(npy_path, image_array)

# Example usage
jpg_path = '.\\imgs\\kfupm.jpg'
npy_path = '.\\imgs\\133.npy'
jpg_to_npy(jpg_path, npy_path)
