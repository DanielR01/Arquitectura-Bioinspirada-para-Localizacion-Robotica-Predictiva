import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

from utils.image_utils import ImageUtils

class ImageTileDataset(Dataset):
    """
    A custom PyTorch Dataset for efficiently loading image tiles for visualization.
    It reads image paths from a CSV and loads images one by one, as needed.
    """
    def __init__(self, csv_file, data_root, transform=None, frame_skip=4, tile_params={'size': 16, 'stride': 8}):
        full_data_frame = pd.read_csv(csv_file)
        self.data_frame = full_data_frame.iloc[::frame_skip + 1, :].reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform
        self.tile_params = tile_params
        print(f"Dataset for visualization initialized with {len(self.data_frame)} images.")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        relative_img_path = self.data_frame.loc[idx, 'right_image_path']
        if relative_img_path.startswith('/'): relative_img_path = relative_img_path[1:]
        full_img_path = os.path.join(self.data_root, relative_img_path)
        try:
            image = Image.open(full_img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image not found at {full_img_path}. Skipping.")
            return None, None, None
        except OSError as e:
            print(f"Warning: OSError reading {full_img_path}: {e}. Skipping.")
            return None, None, None

        if self.transform: image_tensor = self.transform(image)

        # Tiling is now done on the CPU by the worker.
        tiles = ImageUtils.get_image_tiles(image_tensor, tile_size=self.tile_params['size'], stride=self.tile_params['stride'])
        tiles = tiles / 255.0 if tiles.max() > 1.0 else tiles
        return image_tensor, tiles, full_img_path