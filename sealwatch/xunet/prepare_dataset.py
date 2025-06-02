"""
Script to organize BOSSBase dataset into train and test splits.

Author: Max Ninow
Affiliation: University of Innsbruck
"""
import os
import pandas as pd
from shutil import copy2


def process_csvs(
    train_csv: str,
    test_csv: str,
    base_path: str,
    stego_method: str,
) -> None:
    """
    Processes training and testing CSV files, filters stego images by the specified stego method,
    and organizes the images into corresponding folders for cover and stego images.

    :param train_csv: Name of the CSV file for the training split.
    :param test_csv: Name of the CSV file for the testing split.
    :param base_path: Base directory where the CSV files and image files are located.
    :param stego_method: The stego method to filter stego images (e.g., 'hill').
    :return: None
    """
    def process_single_csv(csv_file, split_type):
        """
        Helper function to process a single CSV file and organize images into folders.

        :param csv_file: Name of the CSV file to process.
        :param split_type: Type of split ('tr' for training or 'te' for testing).
        """
        csv_file_path = os.path.join(base_path, csv_file)
        df = pd.read_csv(csv_file_path)
        print(f"Processing {split_type} split: {len(df)} rows in CSV.")

        # Filter stego images by the specified stego method
        stego_df = df[df["stego_method"] == stego_method]
        print(f"Filtered {len(stego_df)} stego images for method '{stego_method}'.")

        # Create output directories for cover and stego images
        cover_folder = os.path.join(base_path, f"split_{split_type}_cover")
        stego_folder = os.path.join(base_path, f"split_{split_type}_{stego_method}_stego")
        os.makedirs(stego_folder, exist_ok=True)

        # Check if cover folder exists
        cover_exists = os.path.exists(cover_folder)
        if not cover_exists:
            os.makedirs(cover_folder, exist_ok=True)

        # Process stego images
        stego_count = 0
        for _, row in stego_df.iterrows():
            stego_path = os.path.join(base_path, row["name"])
            if os.path.exists(stego_path):
                copy2(stego_path, stego_folder)
                stego_count += 1
            else:
                print(f"Stego file not found: {stego_path}")

        # Process cover images if the cover folder does not already exist
        if not cover_exists:
            cover_count = 0
            for _, row in df.iterrows():
                if "stego" not in row["name"]:  # Cover images do not contain "stego" in their name
                    cover_path = os.path.join(base_path, row["name"])
                    if os.path.exists(cover_path):
                        copy2(cover_path, cover_folder)
                        cover_count += 1
                    else:
                        print(f"Cover file not found: {cover_path}")
            print(f"Copied {cover_count} cover images to {cover_folder}.")
        else:
            print(f"Cover folder already exists: {cover_folder}. Skipping cover image copying.")

        print(f"Copied {stego_count} stego images to {stego_folder}.")

    # Process training and testing splits
    process_single_csv(train_csv, split_type="tr")
    process_single_csv(test_csv, split_type="te")