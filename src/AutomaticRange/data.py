from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from tifffile import imread 
from scipy.ndimage import label
from scipy.signal import find_peaks
import cv2
from scipy.ndimage import gaussian_filter


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
        
        #print(f"Loading annotation from {annot_path}")

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
             
        #print(f"Base name for tile: {base_name}")
        #print(f"Suffix for tile: {suffix}")

        marker_path = self.tiles_marker_dir / f"{base_name}_{suffix}.tiff"
        dapi_path = self.tiles_dapi_dir / f"{base_name}_DAPI.tiff"

        # Check if files exist
        if not marker_path.exists():
            raise FileNotFoundError(f"Marker image not found: {marker_path}")
        # else:
            # print(f"Found marker image: {marker_path}")
            
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
    
    @staticmethod
    def preprocess_data(img, percentile=99.9):
        """
        Preprocess an image by clipping at a given percentile, normalizing to [0,1], and removing bright artifacts.

        Args:
            img (np.ndarray): Input image.
            percentile (float): Percentile for clipping.
            threshold_factor (float): Factor for bright artifact removal.
            remove_bright_artifacts (bool): Whether to remove bright artifacts.
        Returns:
            np.ndarray: Preprocessed image.
        """
        # Clip at the given percentile
        perc_val = np.percentile(img, percentile)
        img_clipped = np.copy(img)
        img_clipped[img_clipped > perc_val] = perc_val

        # Normalize to [0, 1]
        img_norm = img_clipped / perc_val

        # Ensure values are in [0, 1]
        img_norm = np.clip(img_norm, 0, 1)
        return img_norm
    
    @staticmethod
    def fast_dilate(mask, radius=50):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
            dilated = cv2.dilate(mask.astype(np.uint8), kernel)
            return dilated
    
    @staticmethod
    def find_lowest_intensity_patch(img, patch_size=100):
        """
        Find the patch_size x patch_size region with the lowest mean intensity,
        ensuring the patch does not overlap the borders of the image.
        """
        H, W = img.shape
        # Ensure the search area excludes borders
        valid_area = img[patch_size:-patch_size, patch_size:-patch_size]
        
        # Compute the integral image for fast sum computation
        integral = np.pad(valid_area, ((1, 0), (1, 0)), mode='constant').cumsum(0).cumsum(1)
        
        # Compute sum in each patch using the integral image
        sums = (
            integral[patch_size:, patch_size:]
            - integral[:-patch_size, patch_size:]
            - integral[patch_size:, :-patch_size]
            + integral[:-patch_size, :-patch_size]
        )
        means = sums / (patch_size * patch_size)
        y, x = np.unravel_index(np.argmin(means), means.shape)
        
        # Adjust coordinates to account for the valid area offset
        y += patch_size
        x += patch_size
        
        return y, x, img[y:y+patch_size, x:x+patch_size]
    
    @staticmethod
    def create_large_noise_background(patch, out_size=1000, smooth_sigma=1.5):
        """
        Tile the patch to create a large background, then smooth it slightly.
        """
        reps_y = int(np.ceil(out_size / patch.shape[0]))
        reps_x = int(np.ceil(out_size / patch.shape[1]))
        tiled = np.tile(patch, (reps_y, reps_x))
        large_bg = tiled[:out_size, :out_size]
        # Smooth a little for realism
        large_bg = gaussian_filter(large_bg, sigma=smooth_sigma)
        return large_bg

    @staticmethod
    def feather_blend(cleaned, noise_bg, mask, feather_radius=20):
        import cv2
        dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
        alpha = np.clip(dist / feather_radius, 0, 1)
        return cleaned * alpha + noise_bg * (1 - alpha)
    
    
    def remove_bright_aggregates(img, threshold_factor=10):
        """
        Detects and suppresses bright aggregates in a grayscale image.

        Args:
            img (np.ndarray): 2D grayscale image (float32 or uint16).
            threshold_factor (float): How many times brighter a peak must be than the 99th percentile to be considered artifact.

        Returns:
            cleaned_img (np.ndarray): Image with bright spots suppressed.
            mask (np.ndarray): Boolean mask of removed regions.
        """

        # Histogram
        hist, bin_edges = np.histogram(img.flatten(), bins=512)

        # Peak detection in histogram
        peaks, _ = find_peaks(hist, distance=20)


        # If no peaks found, return original image
        if len(peaks) == 0:
            print("No peaks found in histogram.")
            return img, np.zeros_like(img, dtype=bool)
        
        # Estimate "normal" high signal level
        signal_level = np.percentile(img, 99)

        # Check the rightmost peak
        brightest_bin_idx = peaks[np.argmax(bin_edges[peaks])]
        brightest_bin_val = bin_edges[brightest_bin_idx]

        # Check if the brightest bin value exceeds the threshold
        if brightest_bin_val > threshold_factor * signal_level:
            print(f"Artifact detected: Brightest bin value {brightest_bin_val:.4f} exceeds threshold {threshold_factor * signal_level:.4f}")
            
            # Artifact suspected: mask extreme high intensity
            thresh = threshold_factor * signal_level
            mask = img > thresh
            # Find objects in the mask
            labeled_mask, num_features = label(mask)
            
            # Discard spots with area < 50 pixels
            for lbl in range(1, num_features + 1):
                single_mask = (labeled_mask == lbl)
                if np.sum(single_mask) < 50:
                    labeled_mask[labeled_mask == lbl] = 0

            labeled_mask, num_features = label(labeled_mask > 0)  # Re-label after filtering small spots
            


            if num_features == 0:
                print("No significant artifacts detected after size filtering.")
                return img, np.zeros_like(img, dtype=bool)


            # Calculate radius proportional to the size of the spot
            radii = []
            for lbl in range(1, num_features + 1):
                single_mask = (labeled_mask == lbl)
                coords = np.argwhere(single_mask)
                size_y = np.max(coords[:, 0]) - np.min(coords[:, 0])
                size_x = np.max(coords[:, 1]) - np.min(coords[:, 1])
                size = size_y + size_x
                radii.append(max(5, min(100, int(0.25 * size))))

            # Create an extended mask with variable radii
            extended_mask = np.zeros_like(labeled_mask, dtype=bool)
            for lbl, radius in zip(range(1, num_features + 1), radii):
                single_mask = (labeled_mask == lbl)
                extended_single_mask = RangeAnnotationDataset.fast_dilate(single_mask, radius=radius).astype(bool)
                # Calculate average pixel intensity in the extended spot
                avg_intensity = np.mean(img[extended_single_mask])

                # Discard the spot if the average intensity is below the threshold
                if avg_intensity < 0.5 * thresh:
                    print(f"Discarding spot with average intensity {avg_intensity:.4f} below threshold {thresh:.4f}")
                    num_features = num_features -1
                else:
                    extended_mask |= extended_single_mask
                
            if num_features == 0:
                print("No significant artifacts detected after intensity filtering.")
                return img, np.zeros_like(img, dtype=bool)
                
            # Re-find objects
            relabeled_mask, num_features = label(extended_mask)
            # Directly use the relabeled mask as the final mask
            mask = relabeled_mask > 0

            cleaned = img.copy()

            # Create a fake background with random noise around the median value
            y, x, patch = RangeAnnotationDataset.find_lowest_intensity_patch(img, patch_size=100)
            out_size = int(np.ceil(np.max(img.shape) / 100.0)) * 100
            noise_bg = RangeAnnotationDataset.create_large_noise_background(patch, out_size=out_size, smooth_sigma=1.5)
            noise_bg = noise_bg[:mask.shape[0], :mask.shape[1]:]  # Ensure it matches the mask size

            # Fill the cleaned image values of the spot with the background values
            cleaned[mask] = noise_bg[mask]

            # Smooth only the regions around the detected spots (mask + ~50 pixel radius around)
            smooth_radius = 50
            spot_mask = RangeAnnotationDataset.fast_dilate(mask, radius=smooth_radius)  # Create a mask for the regions to smooth

            # Apply smoothing only to the regions defined by the spot_mask
            smoothed_regions = RangeAnnotationDataset.feather_blend(cleaned, noise_bg, mask, feather_radius=50)

            print(f"Detected {num_features} large & bright enough spots in the image.")

            cleaned[spot_mask.astype(bool)] = smoothed_regions[spot_mask.astype(bool)]
            
        else:
            # No artifact detected
            mask = np.zeros_like(img, dtype=bool)
            cleaned = img
            
        return cleaned, mask