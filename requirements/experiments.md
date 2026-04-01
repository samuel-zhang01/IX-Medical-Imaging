# Experiment Plan: Trustworthy AI in MRI Reconstruction

> **Project**: Imperial College London TAIMI 2024-2025 Mini-Project
> **Topic**: Trustworthy AI in MRI Reconstruction
> **Deadline**: 3 April 2026, 12:00 midday
> **Format**: 8-page Springer LNCS paper (max 1 page references)
> **Dataset**: MM-WHS cardiac MRI (and CT for cross-domain evaluation)

---

## Global Configuration and Constants

All experiments share these paths and settings. The implementing agent should create a
single `config.py` file that every notebook and script imports.

```python
# config.py -- saved at /root/IX-Medical-Imaging/config.py

import os

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = "/root/IX-Medical-Imaging"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "processed_data")

MR_TRAIN_DIR = os.path.join(DATA_ROOT, "mr_256", "train", "npz")
MR_VAL_DIR   = os.path.join(DATA_ROOT, "mr_256", "val", "npz")
MR_TEST_DIR  = os.path.join(DATA_ROOT, "mr_256", "test", "npz")

CT_TRAIN_DIR = os.path.join(DATA_ROOT, "ct_256", "train", "npz")
CT_VAL_DIR   = os.path.join(DATA_ROOT, "ct_256", "val", "npz")
CT_TEST_DIR  = os.path.join(DATA_ROOT, "ct_256", "test", "npz")

FIGURES_DIR   = os.path.join(PROJECT_ROOT, "latex", "figures")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results")

# ============================================================
# Dataset sizes (verified)
# ============================================================
# MR: train=1738, val=254, test=236  (256x256, image=float64, label=uint8, classes 0-7)
# CT: train=3389, val=382, test=484  (256x256, same format)
# File naming: mr_XXXX_slice_YYY.npz / ct_XXXX_slice_YYY.npz

# ============================================================
# Global hyperparameters
# ============================================================
IMAGE_SIZE = 256
NUM_CLASSES = 8  # label classes 0-7

DEVICE = "cuda"  # fall back to "cpu" if unavailable
SEED = 42
NUM_WORKERS = 4
```

---

## Experiment 1: Baseline U-Net MRI Reconstruction

### 1.1 Objective

Train a standard U-Net to reconstruct undersampled MRI from zero-filled inputs.
This serves as the baseline for all trustworthiness experiments that follow.

### 1.2 Data Pipeline

**Source files**: `/root/IX-Medical-Imaging/data/processed_data/mr_256/{train,val,test}/npz/*.npz`

Each NPZ contains:
- `image`: shape (256, 256), dtype float64 -- ground-truth magnitude image
- `label`: shape (256, 256), dtype uint8 -- segmentation mask (classes 0-7; used later in Experiment 5)

**Pipeline pseudocode:**

```python
# data_pipeline.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob

class MRIReconDataset(Dataset):
    """
    Loads NPZ slices, simulates k-space undersampling, returns
    (zero_filled_input, ground_truth_target, mask, label).
    """

    def __init__(self, npz_dir, acceleration=4, transform=None,
                 return_label=False):
        """
        Args:
            npz_dir: path to directory containing .npz files
            acceleration: R factor (2, 4, 6, 8, or 10)
            transform: optional augmentation (not needed for MRI recon)
            return_label: if True, also return the segmentation label
        """
        self.file_list = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        self.acceleration = acceleration
        self.transform = transform
        self.return_label = return_label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        image = data['image'].astype(np.float32)  # (256, 256)
        label = data['label']                       # (256, 256) uint8

        # Normalize image to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Simulate k-space acquisition via 2D FFT
        kspace = np.fft.fft2(image)                 # complex (256, 256)
        kspace = np.fft.fftshift(kspace)

        # Generate random Cartesian undersampling mask
        mask = generate_cartesian_mask(
            shape=(256, 256),
            acceleration=self.acceleration,
            center_fraction=0.08
        )  # binary (256, 256)

        # Apply mask
        undersampled_kspace = kspace * mask

        # Zero-filled reconstruction via inverse FFT
        zero_filled = np.abs(np.fft.ifft2(
            np.fft.ifftshift(undersampled_kspace)
        ))

        # Convert to tensors with channel dimension: (1, 256, 256)
        zf_tensor = torch.from_numpy(zero_filled).unsqueeze(0).float()
        gt_tensor = torch.from_numpy(image).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).float()

        if self.return_label:
            label_tensor = torch.from_numpy(label.astype(np.int64))
            return zf_tensor, gt_tensor, mask_tensor, label_tensor
        return zf_tensor, gt_tensor, mask_tensor


def generate_cartesian_mask(shape, acceleration, center_fraction=0.08):
    """
    Generate a random Cartesian undersampling mask.

    Fully samples a central band of phase-encode lines (ACS region),
    then randomly selects additional lines to reach the target acceleration.

    Args:
        shape: (H, W) image dimensions
        acceleration: target acceleration factor R
        center_fraction: fraction of central k-space lines to always acquire

    Returns:
        mask: binary numpy array (H, W), 1 = sampled, 0 = not sampled
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)

    # Central ACS lines (always sampled)
    num_center = int(W * center_fraction)
    center_start = W // 2 - num_center // 2
    center_end = center_start + num_center
    mask[:, center_start:center_end] = 1.0

    # Number of additional lines to sample
    total_lines_needed = int(W / acceleration)
    additional_lines = max(0, total_lines_needed - num_center)

    # Randomly select from non-center columns
    available = list(set(range(W)) - set(range(center_start, center_end)))
    if additional_lines > 0 and len(available) > 0:
        chosen = np.random.choice(
            available,
            size=min(additional_lines, len(available)),
            replace=False
        )
        mask[:, chosen] = 1.0

    return mask
```

**DataLoader construction:**

```python
# In training script / notebook

train_dataset = MRIReconDataset(MR_TRAIN_DIR, acceleration=4)
val_dataset   = MRIReconDataset(MR_VAL_DIR,   acceleration=4)
test_dataset  = MRIReconDataset(MR_TEST_DIR,  acceleration=4)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False,
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False,
                          num_workers=4, pin_memory=True)
```

### 1.3 Model Architecture

```python
# models/unet.py

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Two 3x3 conv layers, each followed by InstanceNorm and LeakyReLU."""

    def __init__(self, in_ch, out_ch, dropout_rate=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        layers += [
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    4-level U-Net for single-channel MRI reconstruction.

    Architecture summary:
      Encoder: 1->64->128->256->512 (4 down-sampling steps)
      Bottleneck: 512->1024->512
      Decoder: 512->256->128->64 (4 up-sampling steps with skip connections)
      Output: 64->1 (sigmoid activation for [0,1] output)

    Total parameters: ~31M
    """

    def __init__(self, in_ch=1, out_ch=1, base_ch=64, dropout_rate=0.0):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch,      base_ch,     dropout_rate)
        self.enc2 = ConvBlock(base_ch,     base_ch * 2, dropout_rate)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, dropout_rate)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8, dropout_rate)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16, dropout_rate)

        # Decoder (input channels = skip + upsampled)
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 8, dropout_rate)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4, dropout_rate)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2, dropout_rate)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch, dropout_rate)

        # Output
        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)                          # (B, 64, 256, 256)
        e2 = self.enc2(self.pool(e1))               # (B, 128, 128, 128)
        e3 = self.enc3(self.pool(e2))               # (B, 256, 64, 64)
        e4 = self.enc4(self.pool(e3))               # (B, 512, 32, 32)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))          # (B, 1024, 16, 16)

        # Decoder path with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))   # (B, 512, 32, 32)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # (B, 256, 64, 64)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 128, 128, 128)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 64, 256, 256)

        return torch.sigmoid(self.out_conv(d1))     # (B, 1, 256, 256)
```

### 1.4 Loss Function

```python
# losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    """Differentiable SSIM loss (1 - SSIM)."""

    def __init__(self, window_size=11, channel=1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        # Create Gaussian window
        self.register_buffer('window', self._create_window(window_size, channel))

    def _gaussian(self, window_size, sigma=1.5):
        import math
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        window = _2D_window.float().unsqueeze(0).unsqueeze(0)
        return window.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = F.conv2d(pred, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(target, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, self.window,
                             padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window,
                             padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window,
                           padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1.0 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """L1 + alpha * SSIM loss."""

    def __init__(self, alpha=0.84):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        return (1 - self.alpha) * self.l1(pred, target) + self.alpha * self.ssim(pred, target)
```

### 1.5 Evaluation Metrics

```python
# metrics.py

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_psnr(gt, pred, data_range=1.0):
    """Peak Signal-to-Noise Ratio."""
    return psnr(gt, pred, data_range=data_range)

def compute_ssim(gt, pred, data_range=1.0):
    """Structural Similarity Index."""
    return ssim(gt, pred, data_range=data_range)

def compute_nmse(gt, pred):
    """Normalized Mean Squared Error."""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2
```

### 1.6 Training Procedure

```python
# train_baseline.py  (pseudocode for training loop)

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_baseline(acceleration=4):
    # Hyperparameters
    EPOCHS = 50
    BATCH_SIZE = 4
    LR = 1e-3
    SEED = 42

    torch.manual_seed(SEED)

    # Model, loss, optimizer
    model = UNet(in_ch=1, out_ch=1, base_ch=64, dropout_rate=0.0).to(DEVICE)
    criterion = CombinedLoss(alpha=0.84)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Data loaders
    train_loader = DataLoader(
        MRIReconDataset(MR_TRAIN_DIR, acceleration=acceleration),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        MRIReconDataset(MR_VAL_DIR, acceleration=acceleration),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        for zf, gt, mask in train_loader:
            zf, gt = zf.to(DEVICE), gt.to(DEVICE)
            pred = model(zf)
            loss = criterion(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for zf, gt, mask in val_loader:
                zf, gt = zf.to(DEVICE), gt.to(DEVICE)
                pred = model(zf)
                val_loss += criterion(pred, gt).item()

        val_losses.append(val_loss / len(val_loader))

        # Save best model
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            torch.save(model.state_dict(),
                       f"{CHECKPOINTS_DIR}/unet_baseline_R{acceleration}.pth")

        print(f"Epoch {epoch+1}/{EPOCHS}  "
              f"Train Loss: {train_losses[-1]:.6f}  "
              f"Val Loss: {val_losses[-1]:.6f}")

    return model, train_losses, val_losses
```

### 1.7 Hyperparameters Summary

| Parameter | Value | Justification |
|---|---|---|
| Base channels | 64 | Standard for 256x256 input; ~31M params |
| Levels | 4 | Sufficient receptive field for 256x256 |
| Batch size | 4 | Fits in ~8 GB VRAM with 256x256 single-channel |
| Learning rate | 1e-3 | Standard for Adam on reconstruction tasks |
| LR schedule | Cosine annealing | Smooth decay, no manual tuning |
| Epochs | 50 | 1738 train samples / 4 = 435 iters/epoch, ~21750 total |
| Loss weights | alpha=0.84 for SSIM, 0.16 for L1 | Per Zhao et al. (2017) |
| Acceleration R | 4, 8 | Standard factors; R=4 is moderate, R=8 is aggressive |
| Center fraction | 0.08 | ~20 ACS lines; standard for Cartesian sampling |
| Normalization | InstanceNorm2d | Better than BatchNorm for small-batch MRI |
| Activation | LeakyReLU(0.2) | Avoids dead neurons |
| Output activation | Sigmoid | Constrains output to [0,1] range |

### 1.8 Expected Runtime

- Training: ~15-25 min per epoch on single GPU (V100/A100), total ~15-20 hours for 50 epochs
- If using a smaller GPU (e.g., T4), ~30-40 min per epoch; reduce to 30 epochs if time-constrained
- Inference (full test set, 236 images): ~30 seconds

### 1.9 Output Files

| File | Path |
|---|---|
| Trained model (R=4) | `checkpoints/unet_baseline_R4.pth` |
| Trained model (R=8) | `checkpoints/unet_baseline_R8.pth` |
| Training curves | `results/exp1_training_curves_R4.json`, `results/exp1_training_curves_R8.json` |
| Test metrics | `results/exp1_test_metrics_R4.json`, `results/exp1_test_metrics_R8.json` |

### 1.10 Figures Produced

- **Fig: Training convergence** -- Train/val loss curves for R=4 and R=8 (save to `latex/figures/exp1_training_curves.pdf`)

### 1.11 Test Evaluation Script

```python
# evaluate_baseline.py

def evaluate_on_test(model, test_loader, device):
    """
    Returns dict with lists of per-sample PSNR, SSIM, NMSE
    and the overall means.
    """
    model.eval()
    psnrs, ssims, nmses = [], [], []

    with torch.no_grad():
        for zf, gt, mask in test_loader:
            zf, gt = zf.to(device), gt.to(device)
            pred = model(zf)

            # Convert to numpy for metrics
            pred_np = pred.squeeze().cpu().numpy()
            gt_np = gt.squeeze().cpu().numpy()

            psnrs.append(compute_psnr(gt_np, pred_np))
            ssims.append(compute_ssim(gt_np, pred_np))
            nmses.append(compute_nmse(gt_np, pred_np))

    return {
        'psnr_mean': np.mean(psnrs), 'psnr_std': np.std(psnrs),
        'ssim_mean': np.mean(ssims), 'ssim_std': np.std(ssims),
        'nmse_mean': np.mean(nmses), 'nmse_std': np.std(nmses),
        'psnr_list': psnrs, 'ssim_list': ssims, 'nmse_list': nmses,
    }
```

---

## Experiment 2: MC Dropout Uncertainty Quantification

### 2.1 Objective

Quantify predictive uncertainty by enabling dropout at test time and performing
multiple stochastic forward passes (Monte Carlo Dropout). This is the primary
trustworthiness method and the core contribution of the paper.

### 2.2 Model Modification

Re-use the exact same U-Net architecture from Experiment 1, but with `dropout_rate=0.1`.
The dropout is already integrated in `ConvBlock` above. Train a NEW model with dropout enabled.

**Key point**: The model is trained WITH dropout (p=0.1), which acts as regularization
during training. At test time, dropout remains ON to produce stochastic predictions.

```python
# Train MC Dropout model
model_mcd = UNet(in_ch=1, out_ch=1, base_ch=64, dropout_rate=0.1).to(DEVICE)
# Use same training procedure as Experiment 1
```

### 2.3 MC Dropout Inference

```python
# mc_dropout_inference.py

def enable_dropout(model):
    """Enable dropout layers during evaluation for MC Dropout."""
    for module in model.modules():
        if isinstance(module, nn.Dropout2d):
            module.train()

def mc_dropout_predict(model, x, T=20):
    """
    Perform T stochastic forward passes with dropout enabled.

    Args:
        model: trained U-Net with dropout
        x: input tensor (B, 1, 256, 256)
        T: number of forward passes

    Returns:
        mean_pred: (B, 1, 256, 256) -- mean reconstruction
        var_map:   (B, 1, 256, 256) -- per-pixel variance (uncertainty)
        all_preds: (T, B, 1, 256, 256) -- all individual predictions
    """
    model.eval()
    enable_dropout(model)

    predictions = []
    with torch.no_grad():
        for t in range(T):
            pred = model(x)
            predictions.append(pred.unsqueeze(0))

    all_preds = torch.cat(predictions, dim=0)  # (T, B, 1, H, W)
    mean_pred = all_preds.mean(dim=0)           # (B, 1, H, W)
    var_map   = all_preds.var(dim=0)            # (B, 1, H, W)

    return mean_pred, var_map, all_preds
```

### 2.4 Uncertainty Evaluation Metrics

```python
# uncertainty_metrics.py

def expected_calibration_error(uncertainty_map, error_map, n_bins=15):
    """
    Compute Expected Calibration Error (ECE) for regression uncertainty.

    Bins pixels by predicted uncertainty, computes mean actual error per bin,
    and measures the gap between predicted and actual uncertainty.

    Args:
        uncertainty_map: (H, W) array of predicted variance values
        error_map: (H, W) array of squared errors |gt - pred|^2

    Returns:
        ece: scalar ECE value (lower is better)
        bin_uncertainties: per-bin mean predicted uncertainty
        bin_errors: per-bin mean actual error
    """
    unc_flat = uncertainty_map.flatten()
    err_flat = error_map.flatten()

    # Bin by uncertainty level
    bin_edges = np.linspace(unc_flat.min(), unc_flat.max() + 1e-10, n_bins + 1)
    bin_uncertainties = []
    bin_errors = []
    bin_counts = []

    for i in range(n_bins):
        mask = (unc_flat >= bin_edges[i]) & (unc_flat < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_uncertainties.append(unc_flat[mask].mean())
            bin_errors.append(err_flat[mask].mean())
            bin_counts.append(mask.sum())

    bin_uncertainties = np.array(bin_uncertainties)
    bin_errors = np.array(bin_errors)
    bin_counts = np.array(bin_counts)

    # Weighted absolute difference
    total = bin_counts.sum()
    ece = np.sum(bin_counts / total * np.abs(bin_uncertainties - bin_errors))

    return ece, bin_uncertainties, bin_errors


def sparsification_analysis(uncertainty_map, error_map, n_steps=20):
    """
    Sparsification plot: progressively remove highest-uncertainty pixels
    and measure how the remaining error drops.

    A well-calibrated model should show steep initial decline (removing
    uncertain pixels removes the most erroneous ones).

    Args:
        uncertainty_map: (H, W) predicted variance
        error_map: (H, W) squared error

    Returns:
        fractions_removed: array of fraction thresholds
        remaining_mse: MSE of remaining pixels at each threshold
        oracle_mse: same but removing by actual error (oracle baseline)
    """
    unc_flat = uncertainty_map.flatten()
    err_flat = error_map.flatten()

    fractions = np.linspace(0, 0.95, n_steps)
    remaining_mse = []
    oracle_mse = []

    for frac in fractions:
        # Remove top-frac highest uncertainty pixels
        n_remove = int(frac * len(unc_flat))
        if n_remove == 0:
            remaining_mse.append(err_flat.mean())
            oracle_mse.append(err_flat.mean())
            continue

        # Uncertainty-based removal
        unc_threshold = np.sort(unc_flat)[-n_remove]
        keep_mask = unc_flat < unc_threshold
        if keep_mask.sum() > 0:
            remaining_mse.append(err_flat[keep_mask].mean())
        else:
            remaining_mse.append(0.0)

        # Oracle removal (best possible)
        err_threshold = np.sort(err_flat)[-n_remove]
        oracle_keep = err_flat < err_threshold
        if oracle_keep.sum() > 0:
            oracle_mse.append(err_flat[oracle_keep].mean())
        else:
            oracle_mse.append(0.0)

    return np.array(fractions), np.array(remaining_mse), np.array(oracle_mse)


def uncertainty_error_correlation(uncertainty_map, error_map):
    """
    Compute Pearson and Spearman correlation between uncertainty and error.
    Higher correlation means uncertainty is more informative.
    """
    from scipy.stats import pearsonr, spearmanr

    unc_flat = uncertainty_map.flatten()
    err_flat = error_map.flatten()

    pearson_r, pearson_p = pearsonr(unc_flat, err_flat)
    spearman_r, spearman_p = spearmanr(unc_flat, err_flat)

    return {
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'spearman_r': spearman_r, 'spearman_p': spearman_p,
    }
```

### 2.5 Full Evaluation Pipeline

```python
# evaluate_mc_dropout.py

def evaluate_mc_dropout(model, test_loader, T=20, device='cuda'):
    """
    Run MC Dropout inference on test set, compute all uncertainty metrics.
    """
    model.eval()
    enable_dropout(model)

    all_results = []

    for idx, (zf, gt, mask) in enumerate(test_loader):
        zf, gt = zf.to(device), gt.to(device)

        mean_pred, var_map, _ = mc_dropout_predict(model, zf, T=T)

        pred_np = mean_pred.squeeze().cpu().numpy()
        gt_np = gt.squeeze().cpu().numpy()
        var_np = var_map.squeeze().cpu().numpy()

        # Error map: squared pixel-wise error
        error_map = (gt_np - pred_np) ** 2

        # Metrics
        result = {
            'psnr': compute_psnr(gt_np, pred_np),
            'ssim': compute_ssim(gt_np, pred_np),
            'nmse': compute_nmse(gt_np, pred_np),
            'ece': expected_calibration_error(var_np, error_map)[0],
            'correlation': uncertainty_error_correlation(var_np, error_map),
        }
        all_results.append(result)

        # Save a few example images for visualization
        if idx < 10:
            np.savez(
                f"{RESULTS_DIR}/exp2_mcd_sample_{idx}.npz",
                gt=gt_np, pred=pred_np, var=var_np, error=error_map
            )

    return all_results
```

### 2.6 Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| Dropout rate (p) | 0.1 | Light dropout; heavier rates (0.2+) degrade MRI reconstruction quality |
| MC samples (T) | 20 | Good balance of estimate quality vs compute; diminishing returns past ~30 |
| Dropout placement | After each conv in ConvBlock | Spatial dropout (Dropout2d) preserves spatial correlations |
| Training | Same as Exp 1 | Fair comparison requires identical training setup |

### 2.7 Expected Runtime

- Training: Same as Experiment 1 (dropout adds negligible overhead)
- MC Dropout inference (T=20, 236 test images): ~10-12 minutes on GPU
- Metric computation: ~2-3 minutes

### 2.8 Output Files

| File | Path |
|---|---|
| Trained model (R=4) | `checkpoints/unet_mcdropout_R4.pth` |
| Trained model (R=8) | `checkpoints/unet_mcdropout_R8.pth` |
| MC Dropout test results | `results/exp2_mcd_metrics_R4.json`, `results/exp2_mcd_metrics_R8.json` |
| Sample predictions (NPZ) | `results/exp2_mcd_sample_{0..9}.npz` |

### 2.9 Figures Produced

- **Fig: Uncertainty maps** -- Side-by-side: ground truth, zero-filled, MC mean reconstruction, variance map, absolute error map (save to `latex/figures/exp2_uncertainty_maps.pdf`)
- **Fig: Calibration plot** -- Predicted uncertainty vs actual error per bin (save to `latex/figures/exp2_calibration.pdf`)
- **Fig: Sparsification plot** -- Remaining MSE vs fraction of pixels removed (with oracle) (save to `latex/figures/exp2_sparsification.pdf`)

---

## Experiment 3: Deep Ensemble Uncertainty

### 3.1 Objective

Train N=5 independently initialized U-Net models and combine predictions to estimate
uncertainty. Compare ensemble uncertainty quality against MC Dropout from Experiment 2.

### 3.2 Training Protocol

```python
# train_ensemble.py

def train_ensemble(N=5, acceleration=4):
    """
    Train N models with different random seeds.
    Each model is an identical U-Net (no dropout) trained independently.
    """
    ensemble_models = []

    for i in range(N):
        seed = 42 + i * 1000  # seeds: 42, 1042, 2042, 3042, 4042
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = UNet(in_ch=1, out_ch=1, base_ch=64, dropout_rate=0.0).to(DEVICE)
        # Train with SAME procedure as Experiment 1
        model, train_losses, val_losses = train_baseline_model(
            model, acceleration=acceleration, seed=seed
        )

        checkpoint_path = f"{CHECKPOINTS_DIR}/unet_ensemble_{i}_R{acceleration}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        ensemble_models.append(model)

        print(f"Ensemble member {i+1}/{N} trained. Seed={seed}")

    return ensemble_models
```

### 3.3 Ensemble Inference

```python
# ensemble_inference.py

def ensemble_predict(models, x):
    """
    Forward pass through all ensemble members.

    Args:
        models: list of N trained UNet models
        x: input tensor (B, 1, 256, 256)

    Returns:
        mean_pred: (B, 1, 256, 256) ensemble mean
        var_map:   (B, 1, 256, 256) ensemble variance
        all_preds: (N, B, 1, 256, 256) individual predictions
    """
    predictions = []

    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(x)
            predictions.append(pred.unsqueeze(0))

    all_preds = torch.cat(predictions, dim=0)  # (N, B, 1, H, W)
    mean_pred = all_preds.mean(dim=0)
    var_map   = all_preds.var(dim=0)

    return mean_pred, var_map, all_preds
```

### 3.4 Comparative Evaluation

```python
# compare_uncertainty.py

def compare_mcd_vs_ensemble(mcd_results, ensemble_results):
    """
    Produce a comparison table and plots for:
    - Reconstruction quality (PSNR, SSIM, NMSE)
    - Uncertainty quality (ECE, sparsification AUC, correlation)

    Results formatted as a LaTeX table for the paper.
    """
    comparison = {
        'method': ['MC Dropout (T=20)', 'Deep Ensemble (N=5)'],
        'psnr':   [mcd_results['psnr_mean'], ens_results['psnr_mean']],
        'ssim':   [mcd_results['ssim_mean'], ens_results['ssim_mean']],
        'nmse':   [mcd_results['nmse_mean'], ens_results['nmse_mean']],
        'ece':    [mcd_results['ece_mean'],  ens_results['ece_mean']],
        'pearson':[mcd_results['corr_mean'], ens_results['corr_mean']],
    }
    return comparison
```

### 3.5 Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| Ensemble size N | 5 | Standard in deep ensemble literature (Lakshminarayanan 2017); diminishing returns past 5 |
| Seeds | 42, 1042, 2042, 3042, 4042 | Sufficiently separated for diverse initialization |
| Architecture | Identical U-Net (no dropout) | Isolate effect of ensemble diversity |
| Training | Same procedure per member as Exp 1 | Fair comparison |

### 3.6 Expected Runtime

- Training: 5x the cost of Experiment 1 = ~75-100 hours total on single GPU
- **Mitigation**: Train members in parallel if multiple GPUs available, or reduce to 30 epochs per member (~45-60 hours)
- Ensemble inference (236 test images, 5 members): ~2-3 minutes

### 3.7 Output Files

| File | Path |
|---|---|
| Ensemble member models | `checkpoints/unet_ensemble_{0..4}_R4.pth`, `checkpoints/unet_ensemble_{0..4}_R8.pth` |
| Ensemble test results | `results/exp3_ensemble_metrics_R4.json`, `results/exp3_ensemble_metrics_R8.json` |
| Comparison table | `results/exp3_mcd_vs_ensemble_comparison.json` |

### 3.8 Figures Produced

- **Fig: Ensemble vs MCD uncertainty maps** -- Same test image: GT, recon (MCD), recon (Ensemble), var (MCD), var (Ensemble), error (save to `latex/figures/exp3_ensemble_vs_mcd.pdf`)
- **Fig: Calibration comparison** -- Overlaid calibration plots for both methods (save to `latex/figures/exp3_calibration_comparison.pdf`)
- **Fig: Sparsification comparison** -- Overlaid sparsification curves (save to `latex/figures/exp3_sparsification_comparison.pdf`)

---

## Experiment 4: Robustness Analysis

### 4.1 Objective

Evaluate how reconstruction quality and uncertainty estimates degrade under increasingly
challenging conditions: higher acceleration, additive noise, and cross-domain data.
This directly addresses the "trustworthiness" narrative: a trustworthy model should
produce higher uncertainty when conditions are more challenging.

### 4.2 Sub-experiment 4a: Acceleration Factor Sweep

```python
# robustness_acceleration.py

def acceleration_sweep(model, test_dir, accelerations=[2, 4, 6, 8, 10], T=20):
    """
    Evaluate MC Dropout model at multiple acceleration factors.
    The model was trained at R=4; test at R=2,4,6,8,10 to see degradation.

    Returns:
        results: dict mapping R -> {psnr, ssim, nmse, mean_uncertainty}
    """
    results = {}

    for R in accelerations:
        test_dataset = MRIReconDataset(test_dir, acceleration=R)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        psnrs, ssims, nmses, mean_uncs = [], [], [], []

        for zf, gt, mask in test_loader:
            zf, gt = zf.to(DEVICE), gt.to(DEVICE)
            mean_pred, var_map, _ = mc_dropout_predict(model, zf, T=T)

            pred_np = mean_pred.squeeze().cpu().numpy()
            gt_np = gt.squeeze().cpu().numpy()
            var_np = var_map.squeeze().cpu().numpy()

            psnrs.append(compute_psnr(gt_np, pred_np))
            ssims.append(compute_ssim(gt_np, pred_np))
            nmses.append(compute_nmse(gt_np, pred_np))
            mean_uncs.append(var_np.mean())

        results[R] = {
            'psnr': np.mean(psnrs), 'ssim': np.mean(ssims),
            'nmse': np.mean(nmses), 'uncertainty': np.mean(mean_uncs),
        }
        print(f"R={R}: PSNR={results[R]['psnr']:.2f}, "
              f"SSIM={results[R]['ssim']:.4f}, "
              f"Unc={results[R]['uncertainty']:.6f}")

    return results
```

### 4.3 Sub-experiment 4b: Noise Robustness

```python
# robustness_noise.py

def noise_robustness(model, test_dir, snr_levels=[40, 30, 20, 15, 10],
                     acceleration=4, T=20):
    """
    Add Gaussian noise to the k-space data at various SNR levels BEFORE
    undersampling, to simulate acquisition noise.

    SNR (dB) = 10 * log10(signal_power / noise_power)
    """
    results = {}

    for snr_db in snr_levels:
        psnrs, ssims, mean_uncs = [], [], []

        test_dataset = MRIReconDataset(test_dir, acceleration=acceleration)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        for zf, gt, mask in test_loader:
            gt_np = gt.squeeze().numpy()

            # Recompute k-space from ground truth and add noise
            kspace = np.fft.fftshift(np.fft.fft2(gt_np))
            signal_power = np.mean(np.abs(kspace) ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(*kspace.shape) + 1j * np.random.randn(*kspace.shape)
            )
            noisy_kspace = kspace + noise

            # Apply undersampling mask
            mask_np = generate_cartesian_mask((256, 256), acceleration, 0.08)
            undersampled = noisy_kspace * mask_np
            zf_noisy = np.abs(np.fft.ifft2(np.fft.ifftshift(undersampled)))
            zf_noisy = (zf_noisy - zf_noisy.min()) / (zf_noisy.max() - zf_noisy.min() + 1e-8)

            zf_tensor = torch.from_numpy(zf_noisy).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            mean_pred, var_map, _ = mc_dropout_predict(model, zf_tensor, T=T)

            pred_np = mean_pred.squeeze().cpu().numpy()
            var_np = var_map.squeeze().cpu().numpy()

            psnrs.append(compute_psnr(gt_np, pred_np))
            ssims.append(compute_ssim(gt_np, pred_np))
            mean_uncs.append(var_np.mean())

        results[snr_db] = {
            'psnr': np.mean(psnrs), 'ssim': np.mean(ssims),
            'uncertainty': np.mean(mean_uncs),
        }

    return results
```

### 4.4 Sub-experiment 4c: Cross-Domain Evaluation (MR to CT)

```python
# robustness_crossdomain.py

def cross_domain_evaluation(model, ct_test_dir, acceleration=4, T=20):
    """
    Apply the MR-trained model to CT data.
    CT images are normalized to [0,1] using per-slice min-max (same as MR).
    Expect: degraded reconstruction quality AND higher uncertainty.
    This tests whether the model "knows what it doesn't know."
    """
    ct_dataset = MRIReconDataset(ct_test_dir, acceleration=acceleration)
    ct_loader = DataLoader(ct_dataset, batch_size=1, shuffle=False)

    psnrs, ssims, mean_uncs = [], [], []

    for zf, gt, mask in ct_loader:
        zf, gt = zf.to(DEVICE), gt.to(DEVICE)
        mean_pred, var_map, _ = mc_dropout_predict(model, zf, T=T)

        pred_np = mean_pred.squeeze().cpu().numpy()
        gt_np = gt.squeeze().cpu().numpy()
        var_np = var_map.squeeze().cpu().numpy()

        psnrs.append(compute_psnr(gt_np, pred_np))
        ssims.append(compute_ssim(gt_np, pred_np))
        mean_uncs.append(var_np.mean())

    results = {
        'psnr': np.mean(psnrs), 'ssim': np.mean(ssims),
        'uncertainty': np.mean(mean_uncs),
    }

    print(f"Cross-domain (CT): PSNR={results['psnr']:.2f}, "
          f"SSIM={results['ssim']:.4f}, Unc={results['uncertainty']:.6f}")

    return results
```

### 4.5 Key Analysis: Uncertainty Tracking Degradation

The central argument is:

**Trustworthy AI should produce uncertainty that increases monotonically as actual
performance degrades.**

Produce a combined analysis:

```python
# robustness_combined.py

def plot_uncertainty_vs_quality(accel_results, noise_results, cross_domain_results):
    """
    Create a multi-panel figure showing:
      Panel A: PSNR and mean uncertainty vs acceleration R
      Panel B: PSNR and mean uncertainty vs noise SNR
      Panel C: Bar chart comparing in-domain MR vs cross-domain CT
               (PSNR + uncertainty side by side)

    Save to: latex/figures/exp4_robustness.pdf
    """
    # Implementation: matplotlib, 3 subplots (1 row, 3 columns)
    pass
```

### 4.6 Expected Runtime

- Acceleration sweep (5 R values, 236 test images, T=20 each): ~50-60 min
- Noise sweep (5 SNR levels, 236 test images, T=20 each): ~50-60 min
- Cross-domain CT (484 test images, T=20): ~25-30 min
- Total: ~2.5 hours

### 4.7 Output Files

| File | Path |
|---|---|
| Acceleration sweep results | `results/exp4a_acceleration_sweep.json` |
| Noise sweep results | `results/exp4b_noise_sweep.json` |
| Cross-domain results | `results/exp4c_cross_domain.json` |

### 4.8 Figures Produced

- **Fig: PSNR/SSIM vs R curve** -- Line plot with dual y-axis: left = PSNR/SSIM, right = mean uncertainty (save to `latex/figures/exp4_psnr_vs_acceleration.pdf`)
- **Fig: Noise robustness** -- Similar dual-axis plot vs SNR (save to `latex/figures/exp4_noise_robustness.pdf`)
- **Fig: Cross-domain comparison** -- Bar chart: MR in-domain vs CT cross-domain, showing both quality and uncertainty (save to `latex/figures/exp4_cross_domain.pdf`)

---

## Experiment 5: Downstream Segmentation Impact

### 5.1 Objective

Demonstrate that reconstruction artifacts impact downstream clinical tasks (segmentation),
and that uncertainty maps from reconstruction can predict WHERE segmentation will fail.
This provides a concrete clinical motivation for trustworthy reconstruction.

### 5.2 Segmentation Model

Train a simple U-Net segmenter on ground-truth (fully sampled) MR images.

```python
# models/segmentation_unet.py

class SegmentationUNet(nn.Module):
    """
    Smaller U-Net for 8-class cardiac segmentation.
    Input: (B, 1, 256, 256) magnitude image
    Output: (B, 8, 256, 256) class logits
    """

    def __init__(self, in_ch=1, num_classes=8, base_ch=32):
        super().__init__()
        # Same architecture as reconstruction U-Net but:
        #   - base_ch=32 (smaller, segmentation is simpler)
        #   - output channels = num_classes (8)
        #   - NO sigmoid; use softmax/cross-entropy
        #   - 3 levels instead of 4 (sufficient for segmentation)

        self.enc1 = ConvBlock(in_ch,      base_ch,     0.0)
        self.enc2 = ConvBlock(base_ch,     base_ch * 2, 0.0)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, 0.0)

        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8, 0.0)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4, 0.0)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2, 0.0)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch, 0.0)

        self.out_conv = nn.Conv2d(base_ch, num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)  # (B, 8, 256, 256) raw logits
```

### 5.3 Segmentation Training

```python
# train_segmentation.py

def train_segmentation():
    """
    Train segmentation model on ground-truth MR images.
    """
    # Hyperparameters
    EPOCHS = 30
    BATCH_SIZE = 8
    LR = 1e-3

    model = SegmentationUNet(in_ch=1, num_classes=8, base_ch=32).to(DEVICE)
    criterion = nn.CrossEntropyLoss()  # handles class imbalance reasonably
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Custom dataset that returns (image, label) for segmentation
    # image: (1, 256, 256) float, label: (256, 256) long
    train_dataset = MRISegDataset(MR_TRAIN_DIR)  # loads gt images + labels
    val_dataset = MRISegDataset(MR_VAL_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4)

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    torch.save(model.state_dict(), f"{CHECKPOINTS_DIR}/seg_unet.pth")
    return model


class MRISegDataset(Dataset):
    """Dataset that loads ground-truth images and segmentation labels."""

    def __init__(self, npz_dir):
        self.file_list = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        image = data['image'].astype(np.float32)
        label = data['label'].astype(np.int64)

        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        label_tensor = torch.from_numpy(label).long()

        return image_tensor, label_tensor
```

### 5.4 Downstream Evaluation Pipeline

```python
# evaluate_downstream.py

def compute_dice_per_class(pred_seg, gt_seg, num_classes=8):
    """Compute Dice score per class."""
    dice_scores = {}
    for c in range(num_classes):
        pred_c = (pred_seg == c)
        gt_c = (gt_seg == c)
        intersection = (pred_c & gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        if union == 0:
            dice_scores[c] = 1.0  # both empty
        else:
            dice_scores[c] = 2 * intersection / union
    return dice_scores


def downstream_segmentation_analysis(recon_model, seg_model,
                                     test_dir, accelerations=[2, 4, 6, 8, 10],
                                     T=20):
    """
    For each acceleration:
      1. Reconstruct test images using MCD U-Net
      2. Apply segmentation model to reconstructed images
      3. Compare segmentation to GT segmentation labels
      4. Compute Dice scores per class

    Also compute baseline: segmentation on ground truth images.
    """
    results = {}

    test_dataset = MRIReconDataset(test_dir, acceleration=2,
                                   return_label=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Baseline: segmentation on GT images
    gt_dices = []
    for zf, gt, mask, label in test_loader:
        gt = gt.to(DEVICE)
        label_np = label.squeeze().numpy()

        seg_logits = seg_model(gt)
        seg_pred = seg_logits.argmax(dim=1).squeeze().cpu().numpy()
        gt_dices.append(compute_dice_per_class(seg_pred, label_np))
    results['gt'] = aggregate_dice_scores(gt_dices)

    # For each acceleration factor
    for R in accelerations:
        test_dataset_R = MRIReconDataset(test_dir, acceleration=R,
                                         return_label=True)
        test_loader_R = DataLoader(test_dataset_R, batch_size=1, shuffle=False)

        recon_dices = []
        for zf, gt, mask, label in test_loader_R:
            zf = zf.to(DEVICE)
            label_np = label.squeeze().numpy()

            # Reconstruct
            mean_pred, var_map, _ = mc_dropout_predict(recon_model, zf, T=T)

            # Segment the reconstruction
            seg_logits = seg_model(mean_pred)
            seg_pred = seg_logits.argmax(dim=1).squeeze().cpu().numpy()
            recon_dices.append(compute_dice_per_class(seg_pred, label_np))

        results[f'R={R}'] = aggregate_dice_scores(recon_dices)

    return results


def aggregate_dice_scores(dice_list):
    """Aggregate per-image Dice scores into mean +/- std per class."""
    aggregated = {}
    for c in range(8):
        scores = [d[c] for d in dice_list]
        aggregated[c] = {'mean': np.mean(scores), 'std': np.std(scores)}
    # Mean Dice across all classes (excluding background class 0)
    all_scores = [d[c] for d in dice_list for c in range(1, 8)]
    aggregated['mean_dice'] = np.mean(all_scores)
    return aggregated
```

### 5.5 Uncertainty-Segmentation Failure Correlation

```python
# uncertainty_seg_correlation.py

def correlate_uncertainty_and_seg_failure(recon_model, seg_model,
                                         test_dir, acceleration=4, T=20):
    """
    For each test image:
      1. Compute reconstruction uncertainty map (MC Dropout variance)
      2. Compute segmentation failure map (pixels where seg(recon) != seg(GT))
      3. Measure spatial correlation between uncertainty and failure regions

    This is a key figure for the paper: does high reconstruction uncertainty
    predict downstream task failure?
    """
    test_dataset = MRIReconDataset(test_dir, acceleration=acceleration,
                                   return_label=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    correlations = []
    sample_maps = []

    for idx, (zf, gt, mask, label) in enumerate(test_loader):
        zf, gt = zf.to(DEVICE), gt.to(DEVICE)
        label_np = label.squeeze().numpy()

        # Reconstruction + uncertainty
        mean_pred, var_map, _ = mc_dropout_predict(recon_model, zf, T=T)
        var_np = var_map.squeeze().cpu().numpy()

        # Segmentation on GT
        seg_gt = seg_model(gt).argmax(dim=1).squeeze().cpu().numpy()

        # Segmentation on reconstruction
        seg_recon = seg_model(mean_pred).argmax(dim=1).squeeze().cpu().numpy()

        # Failure map: binary, 1 where segmentation disagrees
        failure_map = (seg_gt != seg_recon).astype(np.float32)

        # Correlation
        corr = uncertainty_error_correlation(var_np, failure_map)
        correlations.append(corr)

        # Save a few examples for visualization
        if idx < 5:
            sample_maps.append({
                'gt': gt.squeeze().cpu().numpy(),
                'recon': mean_pred.squeeze().cpu().numpy(),
                'var': var_np,
                'seg_gt': seg_gt,
                'seg_recon': seg_recon,
                'failure': failure_map,
            })

    return correlations, sample_maps
```

### 5.6 Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| Seg model base channels | 32 | Smaller than recon U-Net; segmentation is easier |
| Seg model levels | 3 | Sufficient for 256x256 segmentation |
| Seg training epochs | 30 | Segmentation converges faster |
| Seg batch size | 8 | Smaller model fits larger batches |
| Loss | CrossEntropyLoss | Standard for multi-class segmentation |
| Classes | 8 (0-7) | 0=background, 1-7 cardiac structures |

### 5.7 Expected Runtime

- Segmentation training: ~3-5 hours (30 epochs, smaller model)
- Downstream evaluation (5 accelerations, 236 images): ~2-3 hours
- Correlation analysis: ~1 hour
- Total: ~7-9 hours

### 5.8 Output Files

| File | Path |
|---|---|
| Segmentation model | `checkpoints/seg_unet.pth` |
| Dice vs acceleration results | `results/exp5_dice_vs_acceleration.json` |
| Correlation results | `results/exp5_uncertainty_seg_correlation.json` |
| Sample visualization data | `results/exp5_sample_maps_{0..4}.npz` |

### 5.9 Figures Produced

- **Fig: Dice vs acceleration** -- Line plot of mean Dice (classes 1-7) vs R, with error bars (save to `latex/figures/exp5_dice_vs_acceleration.pdf`)
- **Fig: Uncertainty predicts failure** -- Multi-panel: GT, recon, uncertainty map, GT segmentation, recon segmentation, failure map (save to `latex/figures/exp5_uncertainty_seg_failure.pdf`)

---

## Experiment 6: Publication Figures

### 6.1 Objective

Generate all publication-quality figures needed for the 8-page LNCS paper.
All figures use matplotlib with consistent styling for a professional appearance.

### 6.2 Global Figure Style

```python
# figure_style.py

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

def set_publication_style():
    """Set consistent style for all figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': False,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
    })

# LNCS column width: ~122mm (~4.8in), text width: ~122mm single col
COLUMN_WIDTH = 4.8   # inches
TEXT_WIDTH = 4.8      # inches (single column LNCS)
FULL_WIDTH = 6.5      # inches (for full-width figures)
```

### 6.3 Figure 1: Architecture Diagram

**Description**: Schematic of the U-Net architecture with MC Dropout placement indicated, plus the overall pipeline (FFT -> mask -> IFFT -> U-Net -> reconstruction).

```python
# figures/fig1_architecture.py

def create_architecture_diagram():
    """
    Create pipeline and architecture diagram.

    Layout (top-to-bottom):
      Row 1: Pipeline flow diagram
        [GT Image] -> [FFT] -> [k-space] -> [Mask x] -> [Undersampled k-space]
        -> [IFFT] -> [Zero-filled] -> [U-Net] -> [Reconstruction]

      Row 2: U-Net internal architecture
        Encoder blocks (with dropout markers) | Skip connections | Decoder blocks
        Show: Conv-IN-LeakyReLU-Dropout at each level

    Use matplotlib patches, arrows, and annotations.
    Alternative: create with TikZ in LaTeX if time permits.

    Save to: latex/figures/fig1_architecture.pdf
    Size: FULL_WIDTH x 3.0 inches
    """
    fig, axes = plt.subplots(2, 1, figsize=(FULL_WIDTH, 5.0),
                             gridspec_kw={'height_ratios': [1, 2]})
    # ... implementation with patches and arrows ...
    fig.savefig(f"{FIGURES_DIR}/fig1_architecture.pdf")
    plt.close()
```

### 6.4 Figure 2: Visual Comparison of Reconstructions

**Description**: Grid showing reconstruction quality at R=4 and R=8 for 2-3 example slices.

```python
# figures/fig2_recon_comparison.py

def create_recon_comparison_figure(samples_R4, samples_R8):
    """
    Grid layout (rows = samples, cols = views):

    | Ground Truth | Zero-Filled (R=4) | Recon (R=4) | Zero-Filled (R=8) | Recon (R=8) |
    |    Slice A   |        ...         |     ...     |        ...         |     ...     |
    |    Slice B   |        ...         |     ...     |        ...         |     ...     |

    Below each reconstructed image: PSNR / SSIM annotation.
    All images displayed with same grayscale colormap and windowing.

    Save to: latex/figures/fig2_recon_comparison.pdf
    Size: FULL_WIDTH x 3.5 inches (for 2 example rows)
    """
    num_samples = 2
    fig, axes = plt.subplots(num_samples, 5, figsize=(FULL_WIDTH, 3.5))

    for i in range(num_samples):
        # Column 0: Ground truth
        axes[i, 0].imshow(samples_R4[i]['gt'], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('GT' if i == 0 else '')
        axes[i, 0].axis('off')

        # Column 1: Zero-filled R=4
        axes[i, 1].imshow(samples_R4[i]['zf'], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('ZF (R=4)' if i == 0 else '')
        axes[i, 1].axis('off')

        # Column 2: Reconstruction R=4
        axes[i, 2].imshow(samples_R4[i]['recon'], cmap='gray', vmin=0, vmax=1)
        p4 = samples_R4[i]['psnr']
        s4 = samples_R4[i]['ssim']
        axes[i, 2].set_title(f'Recon (R=4)\n{p4:.1f}dB / {s4:.3f}' if i == 0
                             else f'{p4:.1f}dB / {s4:.3f}')
        axes[i, 2].axis('off')

        # Column 3: Zero-filled R=8
        axes[i, 3].imshow(samples_R8[i]['zf'], cmap='gray', vmin=0, vmax=1)
        axes[i, 3].set_title('ZF (R=8)' if i == 0 else '')
        axes[i, 3].axis('off')

        # Column 4: Reconstruction R=8
        axes[i, 4].imshow(samples_R8[i]['recon'], cmap='gray', vmin=0, vmax=1)
        p8 = samples_R8[i]['psnr']
        s8 = samples_R8[i]['ssim']
        axes[i, 4].set_title(f'Recon (R=8)\n{p8:.1f}dB / {s8:.3f}' if i == 0
                             else f'{p8:.1f}dB / {s8:.3f}')
        axes[i, 4].axis('off')

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig2_recon_comparison.pdf")
    plt.close()
```

### 6.5 Figure 3: Uncertainty Maps Overlaid on Reconstructions

**Description**: Core figure for the paper. Shows uncertainty maps alongside error maps to visually demonstrate calibration.

```python
# figures/fig3_uncertainty_maps.py

def create_uncertainty_maps_figure(samples):
    """
    Grid layout for 2 example slices:

    | GT | Recon | Abs Error | MC Dropout Var | Ensemble Var |
    |  A |  ...  |    ...    |      ...       |     ...      |
    |  B |  ...  |    ...    |      ...       |     ...      |

    Error maps: 'hot' colormap
    Uncertainty maps: 'hot' colormap (same scale as error for visual comparison)
    GT and recon: 'gray' colormap

    Save to: latex/figures/fig3_uncertainty_maps.pdf
    Size: FULL_WIDTH x 3.0 inches
    """
    num_samples = 2
    fig, axes = plt.subplots(num_samples, 5, figsize=(FULL_WIDTH, 3.0))

    for i in range(num_samples):
        s = samples[i]

        axes[i, 0].imshow(s['gt'], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('GT' if i == 0 else '')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(s['recon'], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Reconstruction' if i == 0 else '')
        axes[i, 1].axis('off')

        im_err = axes[i, 2].imshow(s['abs_error'], cmap='hot', vmin=0)
        axes[i, 2].set_title('|Error|' if i == 0 else '')
        axes[i, 2].axis('off')

        im_mcd = axes[i, 3].imshow(s['mcd_var'], cmap='hot', vmin=0)
        axes[i, 3].set_title('MC Dropout Var' if i == 0 else '')
        axes[i, 3].axis('off')

        im_ens = axes[i, 4].imshow(s['ens_var'], cmap='hot', vmin=0)
        axes[i, 4].set_title('Ensemble Var' if i == 0 else '')
        axes[i, 4].axis('off')

    # Add colorbars
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig3_uncertainty_maps.pdf")
    plt.close()
```

### 6.6 Figure 4: Calibration Plots

**Description**: Quantitative uncertainty calibration. Expected uncertainty vs observed error.

```python
# figures/fig4_calibration.py

def create_calibration_figure(mcd_cal, ens_cal):
    """
    Plot: predicted uncertainty (x) vs actual error (y) for binned pixels.
    Perfect calibration = diagonal line.

    Two curves: MC Dropout and Deep Ensemble
    Include shaded region showing gap from diagonal.
    Report ECE value in legend.

    Save to: latex/figures/fig4_calibration.pdf
    Size: COLUMN_WIDTH x 3.0 inches
    """
    fig, ax = plt.subplots(1, 1, figsize=(COLUMN_WIDTH, 3.0))

    ax.plot(mcd_cal['bin_unc'], mcd_cal['bin_err'], 'o-',
            label=f'MC Dropout (ECE={mcd_cal["ece"]:.4f})', color='#1f77b4')
    ax.plot(ens_cal['bin_unc'], ens_cal['bin_err'], 's-',
            label=f'Deep Ensemble (ECE={ens_cal["ece"]:.4f})', color='#ff7f0e')

    # Perfect calibration line
    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect calibration')

    ax.set_xlabel('Predicted Uncertainty')
    ax.set_ylabel('Observed Error')
    ax.legend()
    ax.set_title('Uncertainty Calibration')

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig4_calibration.pdf")
    plt.close()
```

### 6.7 Figure 5: PSNR/SSIM vs Acceleration Factor

**Description**: Performance degradation curves from Experiment 4a.

```python
# figures/fig5_psnr_vs_accel.py

def create_psnr_vs_acceleration_figure(accel_results_mcd, accel_results_ens):
    """
    Two subplots side by side:
      Left: PSNR vs R (lines for MC Dropout and Ensemble)
      Right: SSIM vs R

    Both include mean uncertainty on secondary y-axis (dashed line).

    Save to: latex/figures/fig5_psnr_vs_acceleration.pdf
    Size: FULL_WIDTH x 2.5 inches
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.5))

    Rs = [2, 4, 6, 8, 10]

    # PSNR subplot
    psnr_mcd = [accel_results_mcd[R]['psnr'] for R in Rs]
    psnr_ens = [accel_results_ens[R]['psnr'] for R in Rs]
    ax1.plot(Rs, psnr_mcd, 'o-', label='MC Dropout', color='#1f77b4')
    ax1.plot(Rs, psnr_ens, 's-', label='Deep Ensemble', color='#ff7f0e')
    ax1.set_xlabel('Acceleration Factor R')
    ax1.set_ylabel('PSNR (dB)')
    ax1.legend()

    # Uncertainty on secondary axis
    ax1b = ax1.twinx()
    unc_mcd = [accel_results_mcd[R]['uncertainty'] for R in Rs]
    ax1b.plot(Rs, unc_mcd, '--', color='#1f77b4', alpha=0.5)
    ax1b.set_ylabel('Mean Uncertainty', color='gray')

    # SSIM subplot
    ssim_mcd = [accel_results_mcd[R]['ssim'] for R in Rs]
    ssim_ens = [accel_results_ens[R]['ssim'] for R in Rs]
    ax2.plot(Rs, ssim_mcd, 'o-', label='MC Dropout', color='#1f77b4')
    ax2.plot(Rs, ssim_ens, 's-', label='Deep Ensemble', color='#ff7f0e')
    ax2.set_xlabel('Acceleration Factor R')
    ax2.set_ylabel('SSIM')
    ax2.legend()

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig5_psnr_vs_acceleration.pdf")
    plt.close()
```

### 6.8 Figure 6: Downstream Segmentation Dice vs Acceleration

```python
# figures/fig6_dice_vs_accel.py

def create_dice_vs_acceleration_figure(dice_results):
    """
    Line plot: Mean Dice (classes 1-7) vs acceleration R.
    Horizontal dashed line = Dice on ground truth images (upper bound).
    Error bars = +/- 1 std.

    Save to: latex/figures/fig6_dice_vs_acceleration.pdf
    Size: COLUMN_WIDTH x 3.0 inches
    """
    fig, ax = plt.subplots(1, 1, figsize=(COLUMN_WIDTH, 3.0))

    Rs = [2, 4, 6, 8, 10]
    mean_dice = [dice_results[f'R={R}']['mean_dice'] for R in Rs]
    gt_dice = dice_results['gt']['mean_dice']

    ax.plot(Rs, mean_dice, 'o-', label='Reconstructed', color='#2ca02c')
    ax.axhline(y=gt_dice, color='k', linestyle='--', alpha=0.7,
               label=f'GT (Dice={gt_dice:.3f})')
    ax.set_xlabel('Acceleration Factor R')
    ax.set_ylabel('Mean Dice Score')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.set_title('Segmentation Quality vs Acceleration')

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig6_dice_vs_acceleration.pdf")
    plt.close()
```

### 6.9 Figure 7: Error Map vs Uncertainty Map Correlation

```python
# figures/fig7_error_uncertainty_scatter.py

def create_error_uncertainty_scatter(var_maps, error_maps):
    """
    Scatter/hexbin plot of pixel-level uncertainty vs absolute error.
    Aggregate across multiple test images for density.

    Include Pearson/Spearman r in legend.

    Save to: latex/figures/fig7_error_vs_uncertainty.pdf
    Size: COLUMN_WIDTH x 3.0 inches
    """
    fig, ax = plt.subplots(1, 1, figsize=(COLUMN_WIDTH, 3.0))

    # Subsample pixels for plotting (full set is 256*256*N = too many)
    all_unc = np.concatenate([v.flatten() for v in var_maps])
    all_err = np.concatenate([e.flatten() for e in error_maps])

    # Random subsample for hexbin
    idx = np.random.choice(len(all_unc), size=min(100000, len(all_unc)),
                           replace=False)
    ax.hexbin(all_unc[idx], all_err[idx], gridsize=50, cmap='Blues',
              mincnt=1)
    ax.set_xlabel('Predicted Uncertainty (Variance)')
    ax.set_ylabel('Absolute Error')

    # Correlation annotation
    from scipy.stats import pearsonr
    r, p = pearsonr(all_unc, all_err)
    ax.annotate(f'Pearson r = {r:.3f}', xy=(0.05, 0.95),
                xycoords='axes fraction', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig7_error_vs_uncertainty.pdf")
    plt.close()
```

### 6.10 Figure 8: Cross-Domain Robustness Comparison

```python
# figures/fig8_cross_domain.py

def create_cross_domain_figure(in_domain_results, cross_domain_results):
    """
    Grouped bar chart:
      Groups: [PSNR, SSIM (scaled), Mean Uncertainty (scaled)]
      Within each group: [In-domain MR, Cross-domain CT]

    Also show example reconstructions:
      Row 1: MR test sample (GT, ZF, Recon, Uncertainty)
      Row 2: CT test sample (GT, ZF, Recon, Uncertainty)

    Save to: latex/figures/fig8_cross_domain.pdf
    Size: FULL_WIDTH x 4.0 inches (2 rows: bar chart + examples)
    """
    fig = plt.figure(figsize=(FULL_WIDTH, 4.0))

    # Top row: bar chart comparison
    ax_bar = fig.add_subplot(2, 1, 1)
    # ... grouped bar chart implementation ...

    # Bottom row: example images (4 columns, 2 rows for MR and CT)
    # ... grid of example images ...

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig8_cross_domain.pdf")
    plt.close()
```

### 6.11 Complete Figure Inventory

| Figure # | Description | Source Experiment | File Path | Size |
|---|---|---|---|---|
| 1 | Architecture + pipeline diagram | N/A (design) | `latex/figures/fig1_architecture.pdf` | full-width |
| 2 | Reconstruction comparison R=4,8 | Exp 1 | `latex/figures/fig2_recon_comparison.pdf` | full-width |
| 3 | Uncertainty maps (MCD + Ensemble) | Exp 2, 3 | `latex/figures/fig3_uncertainty_maps.pdf` | full-width |
| 4 | Calibration plots | Exp 2, 3 | `latex/figures/fig4_calibration.pdf` | column-width |
| 5 | PSNR/SSIM vs acceleration R | Exp 4a | `latex/figures/fig5_psnr_vs_acceleration.pdf` | full-width |
| 6 | Dice vs acceleration | Exp 5 | `latex/figures/fig6_dice_vs_acceleration.pdf` | column-width |
| 7 | Error vs uncertainty scatter | Exp 2 | `latex/figures/fig7_error_vs_uncertainty.pdf` | column-width |
| 8 | Cross-domain robustness | Exp 4c | `latex/figures/fig8_cross_domain.pdf` | full-width |

### 6.12 Tables for the Paper

| Table # | Description | Content |
|---|---|---|
| 1 | Reconstruction quality comparison | Methods (Baseline, MCD, Ensemble) x Metrics (PSNR, SSIM, NMSE) x R (4, 8) |
| 2 | Uncertainty quality comparison | Methods (MCD, Ensemble) x Metrics (ECE, Sparsification AUC, Pearson r) |
| 3 | Robustness under noise | SNR levels x PSNR x SSIM x Mean Uncertainty |
| 4 | Downstream segmentation Dice | R values x Mean Dice x Per-class Dice for key structures |

---

## Implementation Order and Dependencies

```
Experiment 1 (Baseline U-Net)
    |
    +---> Experiment 2 (MC Dropout)
    |         |
    |         +---> Experiment 4 (Robustness -- uses MCD model)
    |         |         |
    |         |         +---> Figure 5, Figure 8
    |         |
    |         +---> Figure 3, Figure 4, Figure 7
    |
    +---> Experiment 3 (Deep Ensemble)
    |         |
    |         +---> Figure 3, Figure 4
    |
    +---> Experiment 5 (Downstream Segmentation -- uses recon model)
    |         |
    |         +---> Figure 6
    |
    +---> Experiment 6 (All Figures -- depends on results from 1-5)
              |
              +---> Figure 1 (architecture, can be done anytime)
              +---> Figure 2 (needs Exp 1 results)
```

### Suggested Implementation Schedule (3 days remaining to deadline)

| Day | Task | Priority |
|---|---|---|
| Day 1 (Mar 31) | Implement data pipeline + Exp 1 baseline training (R=4, R=8) | CRITICAL |
| Day 1 (Mar 31) | Start Exp 2 MC Dropout training in parallel | CRITICAL |
| Day 1 (Mar 31) | Start Exp 3 ensemble member 1-2 training | HIGH |
| Day 2 (Apr 1) | Complete Exp 2 evaluation + uncertainty metrics | CRITICAL |
| Day 2 (Apr 1) | Complete Exp 3 ensemble members 3-5 + evaluation | HIGH |
| Day 2 (Apr 1) | Exp 5 segmentation training + downstream evaluation | HIGH |
| Day 2 (Apr 1) | Exp 4 robustness analysis (uses trained MCD model) | MEDIUM |
| Day 3 (Apr 2) | Generate ALL figures (Exp 6) | CRITICAL |
| Day 3 (Apr 2) | Write/finalize paper | CRITICAL |

### Time Budget Estimates (Total)

| Experiment | GPU Hours | Wall-Clock (sequential) |
|---|---|---|
| Exp 1: Baseline (2 models) | ~30-40 hrs | 30-40 hrs |
| Exp 2: MC Dropout (2 models) | ~30-40 hrs | 30-40 hrs |
| Exp 3: Ensemble (10 models) | ~150-200 hrs | 150-200 hrs |
| Exp 4: Robustness (inference only) | ~3-4 hrs | 3-4 hrs |
| Exp 5: Segmentation | ~7-9 hrs | 7-9 hrs |
| Exp 6: Figures | ~0.5 hrs | 0.5 hrs |
| **Total** | **~220-290 hrs** | |

**CRITICAL NOTE ON TIME**: The above estimates assume a single V100/A100 GPU. Given only 3 days:

1. **Reduce epochs to 30** for all models (saves ~40% training time)
2. **Train ensemble with N=3** instead of N=5 if time-constrained
3. **Prioritize Exp 1 + Exp 2** -- these are the core contributions
4. **Exp 3 is nice-to-have** -- can present with fewer ensemble members
5. **If using multi-GPU**: train Exp 1, 2, 3 members simultaneously

---

## Code File Organization

```
/root/IX-Medical-Imaging/
|-- config.py                       # Global configuration
|-- data/
|   |-- processed_data/
|       |-- mr_256/{train,val,test}/npz/   # MR data
|       |-- ct_256/{train,val,test}/npz/   # CT data
|-- models/
|   |-- __init__.py
|   |-- unet.py                     # UNet, ConvBlock
|   |-- segmentation_unet.py        # SegmentationUNet
|-- losses.py                       # CombinedLoss, SSIMLoss
|-- metrics.py                      # PSNR, SSIM, NMSE
|-- data_pipeline.py                # MRIReconDataset, MRISegDataset, mask generation
|-- uncertainty_metrics.py          # ECE, sparsification, correlation
|-- mc_dropout_inference.py         # MC Dropout utilities
|-- ensemble_inference.py           # Ensemble utilities
|-- notebooks/
|   |-- exp1_baseline.ipynb         # Experiment 1 full pipeline
|   |-- exp2_mc_dropout.ipynb       # Experiment 2 full pipeline
|   |-- exp3_ensemble.ipynb         # Experiment 3 full pipeline
|   |-- exp4_robustness.ipynb       # Experiment 4 full pipeline
|   |-- exp5_segmentation.ipynb     # Experiment 5 full pipeline
|   |-- exp6_figures.ipynb          # ALL publication figures
|-- checkpoints/                    # Saved model weights
|-- results/                        # JSON metrics, NPZ intermediate results
|-- latex/
|   |-- figures/                    # PDF/PNG figures for paper
|   |-- TAIMI2025 paper template.tex
|-- requirements/
    |-- experiments.md              # This file
```

---

## Dependencies (pip install)

```
torch>=2.0
torchvision
numpy
scipy
scikit-image
matplotlib
tqdm
json
```

---

## Expected Results Summary (Rough Estimates)

These are ballpark numbers to sanity-check experimental results:

| Metric | ZF (R=4) | Baseline (R=4) | MCD (R=4) | Ensemble (R=4) |
|---|---|---|---|---|
| PSNR (dB) | ~22-25 | ~30-34 | ~30-34 | ~31-35 |
| SSIM | ~0.65-0.75 | ~0.90-0.95 | ~0.90-0.95 | ~0.91-0.96 |
| NMSE | ~0.05-0.10 | ~0.005-0.02 | ~0.005-0.02 | ~0.004-0.015 |

| Metric | ZF (R=8) | Baseline (R=8) | MCD (R=8) | Ensemble (R=8) |
|---|---|---|---|---|
| PSNR (dB) | ~18-22 | ~27-31 | ~27-31 | ~28-32 |
| SSIM | ~0.50-0.65 | ~0.82-0.90 | ~0.82-0.90 | ~0.83-0.91 |

These numbers will vary based on the specific data characteristics. If results
differ substantially from these ranges, investigate data normalization and mask
generation first.
