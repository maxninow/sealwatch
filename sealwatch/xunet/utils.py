import os
import numpy as np
from glob import glob
from PIL import Image
import random

def load_images(cover_path, stego_path, image_size=(32, 32), size=None, random_seed=None):
    """
    Loads cover and stego images from disk and returns them as numpy arrays.

    Args:
        cover_path (str): Path to the directory containing cover images.
        stego_path (str): Path to the directory containing stego images.
        image_size (tuple): Desired size of the images (height, width).
        size (int, optional): Number of random cover-stego pairs to load. Defaults to None (load all pairs).
        random_seed (int, optional): Seed for random selection. Defaults to None.

    Returns:
        X (np.ndarray): Array of images (cover + stego).
        y (np.ndarray): Array of labels (0 for cover, 1 for stego).
    """
    # List all cover and stego image files
    cover_files = sorted(glob(os.path.join(cover_path, "*.png")))
    stego_files = sorted(glob(os.path.join(stego_path, "*.png")))

    # Ensure the number of cover and stego images match
    if len(cover_files) != len(stego_files):
        raise ValueError("Mismatch between the number of cover and stego images.")

    # Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)

    # Randomly select a subset of pairs if size is specified
    if size is not None:
        indices = random.sample(range(len(cover_files)), size)
        cover_files = [cover_files[i] for i in indices]
        stego_files = [stego_files[i] for i in indices]

    cover_images = []
    stego_images = []

    # Load cover images
    for fname in cover_files:
        img = Image.open(fname)
        cover_images.append(np.array(img))

    # Load stego images
    for fname in stego_files:
        img = Image.open(fname)
        stego_images.append(np.array(img))

    # Combine cover and stego images
    X = np.array(cover_images + stego_images, dtype=np.float32)
    y = np.array([0] * len(cover_images) + [1] * len(stego_images), dtype=np.int64)

    return X, y