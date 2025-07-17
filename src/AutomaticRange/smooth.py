import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize



@staticmethod
def infer_grid(img_marker, img_dapi, model, patch_size=200, stride=100, device='cpu'):
    model.eval()
    h, w = img_marker.shape
    min_preds = []
    max_preds = []
    coords = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            marker_patch = img_marker[y:y+patch_size, x:x+patch_size]
            dapi_patch = img_dapi[y:y+patch_size, x:x+patch_size]
            patch = torch.from_numpy(np.stack([marker_patch, dapi_patch])[None, :, :, :]).float().to(device)

            with torch.no_grad():
                pred_min, pred_max = model(patch).squeeze().tolist()
            
            min_preds.append(pred_min)
            max_preds.append(pred_max)
            coords.append((y, x))
    
    grid_shape = (
        (h - patch_size) // stride + 1,
        (w - patch_size) // stride + 1
    )
    min_grid = np.array(min_preds).reshape(grid_shape)
    max_grid = np.array(max_preds).reshape(grid_shape)
    return min_grid, max_grid, coords

@staticmethod
def smooth_predictions(min_grid, max_grid, target_shape, method="gaussian", sigma=1.0):
    if method == "gaussian":
        min_smoothed = gaussian_filter(min_grid, sigma=sigma)
        max_smoothed = gaussian_filter(max_grid, sigma=sigma)
        min_interp = resize(min_smoothed, target_shape, order=1)  # bilinear
        max_interp = resize(max_smoothed, target_shape, order=1)
    elif method == "bicubic":
        min_interp = resize(min_grid, target_shape, order=3)  # bicubic
        max_interp = resize(max_grid, target_shape, order=3)
    else:
        raise ValueError("Method must be 'gaussian' or 'bicubic'")
    return min_interp, max_interp

@staticmethod
def normalize_by_range(img, vmin_map, vmax_map):
    norm = (img - vmin_map) / (vmax_map - vmin_map + 1e-8)
    norm = np.clip(norm, 0, 1)
    return norm

@staticmethod
def predict_range(marker, dapi, model, device = 'cpu', patch_size=200, stride=100):
    
    # Infer the grid
    min_grid, max_grid, coords = infer_grid(marker, dapi, model, patch_size=patch_size, stride=stride, device=device)
    
    # Smooth the predictions
    min_interp, max_interp = smooth_predictions(min_grid, max_grid, marker.shape, method="gaussian", sigma=1.0)
    
    # Normalize the image using the smoothed min and max grids
    marker_norm = normalize_by_range(marker, min_interp, max_interp)

    return marker_norm, min_interp, max_interp