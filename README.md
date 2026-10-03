> **This is an early version of the project and is no longer maintained.**
> The current code, trained models (TensorFlow/Keras and PyTorch), results and live demo are in
> **[ecampus2c/lightweight-3d-unet-dlpfc](https://github.com/ecampus2c/lightweight-3d-unet-dlpfc)**.

3D DLPFC Segmentation for TMS Navigation

A lightweight, automated Deep Learning pipeline for segmenting the Dorsolateral Prefrontal Cortex (DLPFC) from structural MRI scans. Designed for small datasets and consumer-grade hardware.

 Features
- Custom Lightweight 3D U-Net: ~1.4M parameters, trained from scratch.
- Robust Preprocessing: Isotropic resampling and Z-score normalization via SimpleITK.
- Small Data Strategy: Patch-based training with aggressive 3D augmentation.
- Validation: Leave-One-Subject-Out (LOSO) cross-validation.
 Structure
- `src/`: Source code for model, training, and inference.
- `notebooks/`: Demo notebooks.
- `dataset/`: Place your raw NIfTI files here.

Usage

1. Installation
```bash
pip install -r requirements.txt
