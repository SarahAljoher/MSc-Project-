# MSc-Project-
### Synthetic Lung Tumor Generation and U-Net Segmentation
images and corresponding segmentation masks, developed to support deep learning model training in limited-data settings. The project also includes Google Colab notebooks for 2D U-Net segmentation experiments using real and synthetic data.

## Project Overview

This project is divided into two main components:

1. Synthetic Tumor Generation Pipeline
2. U-Net Segmentation Training (4 Experimental Trials)

## Code Files
**MSc project tools file.zip** has the Synthetic Tumor Generation Pipeline code files

- **synthetic_poisson_enhanced.py**
Main script for generating synthetic tumor patches using Poisson blending and 2D Tukey windowing. Masks are generated via SAM and post-processed.

- **extract_real_tumor_slices_and_masks.py**
Extracts real tumor slices and creates binary masks from the Lung-PET-CT-Dx dataset using SAM.

- **run_viewer.py**
Tool for visualizing bounding boxes from XML annotations on top of DICOM images.

- **requirements.txt**
Contains the Python environment dependencies (Anaconda environment).

- **Two utility functions** —  `getUID`, and `get_gt`  — were used in this project. These scripts were sourced from the official Python code provided with the publicly available [Lung-PET-CT-Dx](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/) dataset by The Cancer Imaging Archive (TCIA).


##  Data Sources

This project uses the following datasets:

- **Tumor Data**  
  Cropped tumor regions were extracted from the publicly available [Lung-PET-CT-Dx](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/) dataset, provided by The Cancer Imaging Archive (TCIA). Tumor annotations are stored as XML files in PASCAL VOC format.

- **Healthy CT Scans**  
  Healthy lung CT images were obtained from the [ICASSP 2024 SPGC-3D CBCT dataset](https://sites.google.com/view/icassp2024-spgc-3dcbct/data?authuser=0). Access to this dataset may require a request to the authors.

- **Segmentation Model (SAM)**  
  The Segment Anything Model (SAM) for lung CT segmentation was adapted from the open-source repository [rekalantar/MedSegmentAnything_SAM_LungCT](https://github.com/rekalantar/MedSegmentAnything_SAM_LungCT).

- **Annotation Viewer Tool**  
  Bounding box annotations were visualized using the TCIA-provided toolkit, available [here](https://www.cancerimagingarchive.net/wp-content/uploads/VisualizationTools.zip).

## U-Net Training
- Training, validation, and testing were carried out across four trials to compare the performance of real vs. synthetic data.
- The U-Net architecture is based on the 2D implementation proposed by Ronneberger et al.
- All training experiments are provided as Google Colab notebooks in this repository.

## 🚀 Google Colab Notebooks
Four notebooks ( Trial#.ipynb) for U-Net training, validation, and testing under different experimental conditions.
Click the badges below to open the training notebooks in Google Colab:

[![Open Trial 1 in Colab](https://drive.google.com/file/d/199nOQGTIj9nILdAsfkH4EUY_6imxyCEt/view?usp=sharing)

[![Open Trial 2 in Colab](https://drive.google.com/file/d/1mFzYeoCPGkR8Q7XqmSUeuq6nDSZofWfs/view?usp=sharing)

[![Open Trial 3 in Colab](https://drive.google.com/file/d/1OqW-jlm4M44LsGYgKwq7uu1zdjenEi_k/view?usp=sharing)

[![Open Trial 4 in Colab](https://drive.google.com/file/d/16LHqykL0fGPLlZu1At0g4MCHo-XwOUy3/view?usp=sharing)

