from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import random
from tifffile import imread
class RangeAnnotationDataset(Dataset):
    def __init__(self, annotations_dir, tiles_marker_dir, tiles_dapi_dir, augment=False):
        self.annotations_dir = Path(annotations_dir)
        self.tiles_marker_dir = Path(tiles_marker_dir)
        self.tiles_dapi_dir = Path(tiles_dapi_dir)
        self.augment = augment
        self.annotations = list(self.annotations_dir.glob("*.json"))

        # Define augmentations
        self.transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=(-15, 15)),  # ✅ Small-angle rotation
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),  # slight noise
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annot_path = self.annotations[idx]
        
        print(f"Loading annotation from {annot_path}")

        with open(annot_path) as f:
            annot = json.load(f)
        
        base_name = annot_path.stem  # e.g., MF151_CD4_x123_y456
        # Remove suffix _DAPI or _[marker] if present
        if base_name.endswith("_DAPI"):
            base_name = base_name[:-5]
        elif "_" in base_name:
            base_name = "_".join(base_name.split("_")[:-1])
            # Retrieve the suffix
            suffix = annot_path.stem.split("_")[-1]
             
        print(f"Base name for tile: {base_name}")
        print(f"Suffix for tile: {suffix}")

        marker_path = self.tiles_marker_dir / f"{base_name}_{suffix}.tiff"
        dapi_path = self.tiles_dapi_dir / f"{base_name}_DAPI.tiff"

        # Check if files exist
        if not marker_path.exists():
            raise FileNotFoundError(f"Marker image not found: {marker_path}")
        else:
            print(f"Found marker image: {marker_path}")
            
        # Scale images to [0, 1]
        marker_img = np.array(imread(marker_path)).astype(np.float32)
        dapi_img   = np.array(imread(dapi_path)).astype(np.float32)
        
        # Ensure images are between 0 and 1
        marker_img = np.clip(marker_img, 0, 1)
        dapi_img   = np.clip(dapi_img, 0, 1)

        # Stack marker and DAPI into 2 channels
        image = np.stack([marker_img, dapi_img], axis=0)
        image = torch.from_numpy(image)

        if self.augment:
            image = self.transforms(image)

        # Labels
        target = torch.tensor([annot["min"], annot["max"]], dtype=torch.float32)

        return image, target