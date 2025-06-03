import os
from typing import Tuple, Dict
import torch
from torch import Tensor
from torch.utils.data import Dataset
import imageio.v2 as io
from pathlib import Path


class DatasetLoad(Dataset):
    """This class returns the data samples."""

    def __init__(
        self,
        cover_path: str,
        stego_path: str,
        size: int,
        transform: Tuple = None,
    ) -> None:
        """Constructor.

        Args:
            cover_path (str): Path to cover images.
            stego_path (str): Path to stego images.
            size (int): Number of images in the dataset.
            transform (Tuple, optional): Transformations to apply. Defaults to None.
        """
        self.cover = Path(cover_path)
        self.stego = Path(stego_path)
        self.transforms = transform

        # Validate paths
        if not self.cover.is_dir():
            raise ValueError(f"Cover path '{self.cover}' is not a valid directory.")
        if not self.stego.is_dir():
            raise ValueError(f"Stego path '{self.stego}' is not a valid directory.")

        # Dynamically list all valid image files in the directories
        valid_extensions = (".pgm", ".png")
        self.cover_files = sorted(
            [f for f in os.listdir(self.cover) if f.lower().endswith(valid_extensions)]
        )
        self.stego_files = sorted(
            [f for f in os.listdir(self.stego) if f.lower().endswith(valid_extensions)]
        )

        self.data_size = min(size, len(self.cover_files), len(self.stego_files))


        if len(self.cover_files) != len(self.stego_files):
            raise ValueError("Mismatch between cover and stego image counts.")

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return self.data_size

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Returns the (cover, stego) pairs for training.

        Args:
            index (int): Index of the sample.

        Returns:
            Dict[str, Tensor]: Dictionary containing cover image, stego image, and labels.
        """
        if index >= self.data_size:
            raise IndexError(f"Index {index} is out of range for dataset of size {self.data_size}.")
        cover_img_path = os.path.join(self.cover, self.cover_files[index])
        stego_img_path = os.path.join(self.stego, self.stego_files[index])

        # Load images
        cover_img = io.imread(cover_img_path)
        stego_img = io.imread(stego_img_path)

        # Apply transformations if provided
        if self.transforms:
            cover_img = self.transforms(cover_img)
            stego_img = self.transforms(stego_img)

        # Create labels
        label1 = torch.tensor(0, dtype=torch.long)
        label2 = torch.tensor(1, dtype=torch.long)

        # Create sample dictionary
        sample = {
            "cover": cover_img,
            "stego": stego_img,
            "label": [label1, label2],
        }

        return sample