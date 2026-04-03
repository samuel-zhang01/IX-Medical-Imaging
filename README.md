# Uncertainty-Aware Deep MRI Reconstruction: A Multi-Faceted Trustworthiness Analysis

Individual Mini-Project Report for the TAIMI 2024-2025 module at Imperial College London.

## Overview

This project implements and evaluates a trustworthy deep learning framework for accelerated cardiac MRI reconstruction. A U-Net with cascaded learnable data-consistency (DC) layers is trained on the MM-WHS cardiac dataset, and its trustworthiness is assessed across four dimensions:

1. **Explainability** -- Saliency maps, Grad-CAM, and integrated gradients adapted to reconstruction loss
2. **Uncertainty Quantification** -- MC Dropout (T=30) and deep ensemble (M=3) comparison with calibration diagnostics
3. **Robustness** -- K-space noise, FGSM/PGD adversarial attacks, MR-to-CT cross-domain shift, and DC ablation
4. **Downstream Clinical Impact** -- Cardiac segmentation Dice scores on reconstructed images

The accompanying paper is formatted for LNCS (Springer Lecture Notes in Computer Science) and is located in `latex/main.tex`.

## Repository Structure

```
IX-Medical-Imaging/
|-- latex/                        # LNCS paper and figures
|   |-- main.tex                  # Main paper source
|   |-- main.pdf                  # Compiled paper (8 pages)
|   |-- figures/                  # All publication-quality figures (PDF + PNG)
|   |-- llncs.cls                 # Springer LNCS document class
|   `-- splncs04.bst              # Bibliography style
|
|-- notebooks/                    # Reproducible experiment notebooks
|   |-- 01_data_exploration_and_reconstruction.ipynb
|   |-- 02_trustworthy_xai_analysis.ipynb
|   `-- 03_ensemble_and_calibration.ipynb
|
|-- src/                          # Core Python modules
|   |-- models.py                 # ReconUNet, SegmentationUNet, DataConsistencyLayer
|   |-- data.py                   # MRIReconDataset, k-space simulation pipeline
|   |-- losses.py                 # Combined L1 + SSIM loss, PSNR/SSIM/NMSE metrics
|   |-- train.py                  # Training loop with Optuna hyperparameter search
|   |-- train_final.py            # Final training with best hyperparameters
|   |-- run_all_experiments.py    # End-to-end experiment orchestration
|   |-- generate_xai_figure.py    # GradCAM and attribution figure generation
|   `-- update_latex_results.py   # Auto-populate LaTeX tables from JSON results
|
|-- requirements/                 # Project documentation and planning
|   |-- instructions.md           # Assignment brief
|   |-- lit_review.md             # Literature review notes
|   |-- experiments.md            # Experiment design specification
|   |-- report.md                 # Paper outline and planning
|   `-- research_direction.md     # Research direction analysis
|
|-- literatures/                  # Reference papers
|   |-- 2501.14158v1.pdf          # Safari et al. (2025) - DL+CS MRI review
|   `-- nihms-2102372.pdf         # Trustworthy AI methods survey
|
|-- data/                         # Dataset (not tracked in git)
|   `-- processed_data/
|       |-- mr_256/{train,val,test}/npz/   # MR cardiac slices (256x256)
|       `-- ct_256/{train,val,test}/npz/   # CT cardiac slices (256x256)
|
`-- checkpoints/                  # Trained models and results (not tracked)
    |-- final_R4/                 # Best U-Net+DC model at R=4x
    |-- final_R8/                 # Best U-Net+DC model at R=8x
    |-- ensemble_{0,1,2}_R4/     # Three ensemble members
    |-- seg_model.pth             # Segmentation model
    `-- *.json                    # Experiment result files
```

## Setup

### Dependencies

The project requires Python 3.10+ and the following packages:

- PyTorch 2.0+ with CUDA support
- NumPy, SciPy
- matplotlib
- tqdm
- Optuna (hyperparameter optimisation)
- jupytext (notebook conversion)

Install with:

```bash
pip install torch torchvision numpy scipy matplotlib tqdm optuna jupytext
```

### Data

The MM-WHS cardiac dataset should be placed in `data/processed_data/` with the directory structure shown above. Each `.npz` file contains:
- `image`: Ground-truth magnitude image (256 x 256, float64)
- `label`: Segmentation mask (256 x 256, uint8, classes 0-7)

K-space data and undersampling masks are generated on the fly by the data loader.


## Running the Notebooks

Execute the notebooks in order, as later notebooks depend on checkpoints produced by earlier ones.

### Notebook 01: Data Exploration and Reconstruction Baseline

`notebooks/01_data_exploration_and_reconstruction.ipynb`

Explores the dataset structure, demonstrates k-space undersampling at various acceleration factors, and trains the ReconUNet with Optuna-optimised hyperparameters (dropout rate, learning rate, number of DC cascades). Saves model checkpoints and generates dataset overview figures (Figures 1--8).

### Notebook 02: Trustworthy AI Analysis

`notebooks/02_trustworthy_xai_analysis.ipynb`

Runs all trustworthiness experiments on the trained model: GradCAM and occlusion-based explainability analysis, MC Dropout uncertainty quantification, adversarial robustness evaluation (FGSM and PGD attacks), k-space noise perturbation, and cross-domain (MR-to-CT) generalisation. Trains the segmentation model and computes downstream Dice scores on reconstructed images. Produces Figures 9--12.

### Notebook 03: Deep Ensemble Comparison and Calibration

`notebooks/03_ensemble_and_calibration.ipynb`

Compares MC Dropout and Deep Ensemble uncertainty methods on identical test data. Computes calibration metrics (Expected Calibration Error, Area Under Sparsification Error, Pearson r), runs the data consistency ablation study, and analyses spatial correlation between reconstruction uncertainty and downstream segmentation failures. Produces Figures 13--15 and saves structured results to JSON.


## Compiling the Paper

The LaTeX paper uses the LNCS template. To compile:

```bash
cd latex
pdflatex main.tex
pdflatex main.tex   # second pass for cross-references
```

Or with latexmk:

```bash
cd latex
latexmk -pdf main.tex
```


## Source Modules

| Module | Description |
|--------|-------------|
| `src/data.py` | `MRIReconDataset` class and `get_dataloaders` function for loading cardiac MRI data with on-the-fly k-space simulation, undersampling masks, and segmentation labels |
| `src/models.py` | `ReconUNet` (reconstruction U-Net with MC Dropout and cascaded data consistency layers) and `SegmentationUNet` (tissue segmentation U-Net) |
| `src/losses.py` | Combined L1 + perceptual loss function, plus `compute_psnr`, `compute_ssim`, and `compute_nmse` evaluation metrics |
| `src/train.py` | Optuna-based hyperparameter search over learning rate, dropout rate, base features, and number of DC cascades |
| `src/train_final.py` | Training script using the best hyperparameters found by Optuna, with learning rate scheduling and early stopping |
| `src/run_all_experiments.py` | Orchestrates the full experimental pipeline from training through evaluation |
| `src/generate_xai_figure.py` | Generates GradCAM heatmaps, saliency maps, and attribution overlay figures |
| `src/update_latex_results.py` | Reads result JSONs from `checkpoints/` and programmatically updates LaTeX tables and inline values |


## Key Results

| Metric | Value |
|--------|-------|
| Reconstruction PSNR (R=4x) | 31.9 dB |
| DC ablation PSNR drop | -8.1 dB |
| MC Dropout Pearson r (uncertainty vs error) | 0.59 |
| Ensemble ECE | 0.017 |
| Segmentation Dice preservation (R=4x) | 92% |
| Cross-domain uncertainty increase (MR to CT) | +77% |


## References

Key works referenced in the paper:
- Safari et al. (2025) -- Comprehensive DL+CS MRI reconstruction review
- Atalik et al. (2026) -- Trust-Guided Variational Network (TGVN)
- Gal and Ghahramani (2016) -- MC Dropout as Bayesian approximation
- Lakshminarayanan et al. (2017) -- Deep ensembles for uncertainty
- Schlemper et al. (2018) -- Cascaded DC-CNN for MRI reconstruction
