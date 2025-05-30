"""
Script to split prepare bossbase

Author: Max Ninow
Affiliation: University of Innsbruck
"""
import os
import shutil
import pandas as pd
import random
from PIL import Image

def process_csv(
    csv_file: str,
    base_path: str,
    name: str,
    split_factor: int = None,
) -> None:
    """
    Processes a CSV file as provided from prepare_boss.py, splits the data into training and validation sets 
    (or test sets if no split factor is provided), and organizes the images into corresponding folders.

    :param csv_file: Name of the CSV file to process.
    :param base_path: Base directory where the CSV file and image files are located.
    :param name: A label used to name the output folders (e.g., 'lsbm' for 'split_tr_lsbm_stego' and 'split_tr_lsbm_cover').
    :param split_factor: Factor to split the data into training and validation sets. For example, a split_factor of 5 
                         results in an 80% train and 20% validation split. If None, the function assumes the data is for 
                         testing and does not create a validation set.
    :return: None
    """

    csv_file_path = os.path.join(base_path, csv_file)
    df = pd.read_csv(csv_file_path)
    print(f"Total rows in CSV: {len(df)}")
    
    # group cover and stego images as pairs
    pairs = []
    for _, row in df.iterrows():
        image_path = os.path.join(base_path, row['name'])
        
        if "stego" in image_path:
            filename = os.path.basename(image_path)
            # find corresponding cover image
            cover_path = os.path.join(base_path, "images", filename)
            pairs.append((cover_path, image_path))
    
    print(f"Total pairs: {len(pairs)}")
    
    random.seed(42)  # seed for reproducibility
    random.shuffle(pairs)
    
    # Split into train and validation sets based on the split factor
    if split_factor:
        train_size = int(len(pairs) * ((split_factor - 1) / split_factor))
        train_pairs = pairs[:train_size]
        val_pairs = pairs[train_size:]
        print(f"Train pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}")
        train_stego_folder = os.path.join(base_path, ("split_tr_" + name + "_stego"))
        train_cover_folder = os.path.join(base_path, ("split_tr_" + name + "_cover"))
    
    # no split, hence test set
    else:
        train_pairs = pairs
        val_pairs = []
        train_stego_folder = os.path.join(base_path, ("split_te_" + name + "_stego"))
        train_cover_folder = os.path.join(base_path, ("split_te_" + name + "_cover"))
    
    
    os.makedirs(train_stego_folder, exist_ok=True)
    os.makedirs(train_cover_folder, exist_ok=True)
    
    
    # copy train pairs
    train_cover_counter = 1
    train_stego_counter = 1
    for cover_path, stego_path in train_pairs:
        # cover
        if os.path.exists(cover_path):
            new_cover_name = f"{train_cover_counter:04d}.pgm"
            save_as_pgm(cover_path, os.path.join(train_cover_folder, new_cover_name))
            train_cover_counter += 1
        else:
            print(f"Cover file not found: {cover_path}")
        
        # stego
        if os.path.exists(stego_path):
            new_stego_name = f"{train_stego_counter:04d}.pgm"
            save_as_pgm(stego_path, os.path.join(train_stego_folder, new_stego_name))
            train_stego_counter += 1
        else:
            print(f"Stego file not found: {stego_path}")
    
    # copy validation pairs
    if val_pairs:
        val_stego_folder = os.path.join(base_path, ("split_val_" + name + "_stego"))
        val_cover_folder = os.path.join(base_path, ("split_val_" + name + "_cover"))
        os.makedirs(val_stego_folder, exist_ok=True)
        os.makedirs(val_cover_folder, exist_ok=True)
        val_cover_counter = 1
        val_stego_counter = 1
        for cover_path, stego_path in val_pairs:
            # cover
            if os.path.exists(cover_path):
                new_cover_name = f"{val_cover_counter:04d}.pgm"
                save_as_pgm(cover_path, os.path.join(val_cover_folder, new_cover_name))
                val_cover_counter += 1
            else:
                print(f"Cover file not found: {cover_path}")
            
            # stego
            if os.path.exists(stego_path):
                new_stego_name = f"{val_stego_counter:04d}.pgm"
                save_as_pgm(stego_path, os.path.join(val_stego_folder, new_stego_name))
                val_stego_counter += 1
            else:
                print(f"Stego file not found: {stego_path}")


def save_as_pgm(input_path: str, output_path: str) -> None:
    """
    Converts an image to .pgm format and saves it.

    :param input_path: Path to the input image.
    :param output_path: Path to save the .pgm image.
    :return: None
    """
    try:
        with Image.open(input_path) as img:
            # Ensure the image is in grayscale mode
            img = img.convert("L")
            img.save(output_path, format="PPM")  # Save as .pgm (Pillow uses "PPM" for .pgm files)
    except Exception as e:
        print(f"Error converting {input_path} to .pgm: {e}")