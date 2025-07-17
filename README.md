# AutomaticRange
An open‑source, reproducible intensity‑scaling method that leverages the nuclear signal as an anchor to calibrate per‑channel contrast. 

# Prerequisites

* Python 3.8 (3.7 should work)
* Python libraries:
    - pathlib 
    - napari
    - numpy
    - tifffile
    - torch
    - scipy
    - scikit-image

# Setup

* Download the github repository (> Code > Download ZIP)
* Unzip it in the given location
* In the AutomaticRange folder, create a data/ and a annotation/ folder
* Place the tiles to annotate into data/ folder (e.g. data/CD4/ which contains tiles_marker/ and tiles_dapi/)
* Open VisualStudio IDE and File > Open Folder and open AutomaticRanges
* Open the 2.0_Annotate_ranges_tiles.ipynb jupyter notebook
* Click on Run All. This will open Napari. Then, manually adjust the range of the first image, and then press **Shift + B** to save annotation and go to next image. There are a few key shortcuts that can help you adjust the range faster.



# Key shortcuts

* Shift + A = reduce the minimum contrast by 10%  
* Shift + S = increase the minimum contrast by 10%  
* Shift + Q = reduce the maximum contrast by 10%  
* Shift + W = increase the maximum contrast by 10%  

* **Shift + B : Save manually set range and go to next image available.** If there are no more images available, this will not do anything. It means you are finished !

# Annotation Strategy

* Go as low as possible to see if there is signal enrichment in the nucleus or cell membrane. 
* If you find signal : adjust so that the noise is as low as possible but the cell signal is as clear as possible.
* As much as possible, regions devoid of cells should be black.
* Avoid saturating signal in cells, but privilege seeing positive signal. 

# Introduction
Multiplex fluorescence histochemistry (MxIF) generates large, high-dimensional TIFF images of tissue sections stained for multiple markers (e.g. CD4, nuclear stains), serving as foundational data for modern single-cell analyses. However, raw MxIF images are plagued by variable intensity ranges, heterogeneous signal-to-noise ratios, cell‑density fluctuations, and frequent artifacts such as bright spots, blurs, or folds. Under standard visualization tools (e.g. napari / FIJI), the built‑in “automatic” intensity scaling often fails to reveal biologically relevant patterns—CD4 around nuclei, for instance—forcing users to spend considerable time manually adjusting contrast ranges on each image or region to achieve clarity.
This ad hoc manual adjustment is subjective, non‑reproducible, and hard to scale. Notably:
    1. Human variability: Different users produce diverging contrast settings depending on their tolerance or strategy—some may avoid bright artifacts at the expense of dim but valid signals, while others may favor sensitivity but introduce spurious noise.
    2. Training/annotation bias: Data used for AI model training (e.g. cell segmentation, classification, positive‑marker detection) becomes inconsistent. Varying input contrast can dramatically alter what appears as a signal or background, potentially skewing both annotations and downstream model performance.
    3. Spatial heterogeneity: Large tissue scans often contain both dense and sparse regions—tone‑mapping tuned to a dense region typically underexposes signal in sparse areas, and vice versa. This intra‑image inconsistency further undermines reproducibility.	
Prior work has long acknowledged that arbitrary auto‑contrast settings can distort quantitative interpretation. For instance, converting 16‑bit images to 8‑bit with independent autoscaling leads to intensity-shift artifacts—two tissues with the same raw content might be visually scaled to identical brightness despite underlying differences embopress.org. Similarly, fixed global thresholds are known to fail in heterogeneous settings, and mitigation strategies like histogram equalization (e.g. CLAHE) offer partial solutions, but lack adaptability across complex MxIF scans .
Feature-level normalization pipelines such as z‑score standardization, ComBat, MxNorm, and FLINO address batch effects based on summary features, but are unable to fully reconcile the pixel-wise heterogeneity seen in MxIF images pmc.ncbi.nlm.nih.gov. While methods such as UniFORM aim for unified normalization at the pixel and feature levels, they are tailored for whole‑image batch correction rather than interactive, per-region visualization, and do not directly inform the display scaling needed during image annotation.
Thus, without automated, objective contrast scaling, researchers:
    • Waste hours on manual tuning.
    • Introduce population bias into annotation datasets.
    • Risk missing low-intensity bona fide signals—or accepting false positives.
To overcome these limitations, we introduce AutomaticRange, an open‑source, reproducible intensity‑scaling method that leverages the nuclear signal as an anchor to calibrate per‑channel contrast. AutomaticRange dynamically computes intensity limits at the region-of-interest and global-image levels—excluding artifacts and factoring cell density—and adapts contrast scaling in a statistically consistent manner. We demonstrate that AutomaticRange:
    1. Reduces manual setup time from minutes per image to seconds.
    2. Improves uniform visibility across heterogeneous tissue regions.
    3. Increases reproducibility across annotators and imaging sessions.
    4. Preserves biological features critical for downstream AI models (segmentation, marker detection, cell-type classification).
In the following, we describe the algorithmic design of AutomaticRange, validate its performance relative to standard autoscaling in napari and FIJI, and assess its impact on human annotations and downstream model accuracy.


# napari-automatic-range

## Overview
The `napari-automatic-range` plugin provides a simple interface for performing automatic range predictions on images using a trained model. It integrates seamlessly with the Napari viewer, allowing users to easily apply the `predict_range` function to their images.

## Features
- Button labeled "AutomaticRange" to execute the prediction on the current image.
- Automatically finds the first DAPI image in the viewer for processing.
- Displays the normalized image in the Napari viewer.

## Installation
To install the plugin, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/napari-automatic-range.git
cd napari-automatic-range
pip install -r requirements.txt
```

## Usage
1. Launch Napari.
2. Load your images into the viewer.
3. Click the "AutomaticRange" button in the plugin interface.
4. The normalized image will be displayed in the viewer.

## Requirements
- Napari
- PyTorch
- NumPy
- SciPy
- scikit-image

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

