#!/usr/bin/env python3
"""
Generate Figure 6: XAI Explainability Analysis for MRI Reconstruction.

Produces saliency maps, Grad-CAM at multiple layers, and integrated gradients
for a ReconUNet model, saving a publication-quality figure.
"""
import sys
import os

os.chdir('/root/IX-Medical-Imaging')
sys.path.insert(0, 'src')

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr

from data import MRIReconDataset
from models import ReconUNet
from losses import compute_psnr, CombinedLoss


def load_model_and_sample(device):
    """Load trained model and a single test sample."""
    # Load checkpoint
    ckpt_path = 'checkpoints/final_R4/best_model_R4.pth'
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']

    # Build model
    model = ReconUNet(
        in_channels=1,
        out_channels=1,
        base_features=config['base_features'],
        dropout_rate=config['dropout_rate'],
        use_dc=config.get('use_dc', True),
        num_dc_cascades=config.get('num_dc_cascades', 1),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load test dataset and pick a sample
    test_ds = MRIReconDataset(
        'data/processed_data/mr_256/test/npz',
        acceleration=4,
        center_fraction=config.get('center_fraction', 0.08),
        fixed_masks=True,
    )

    # Pick a sample with good anatomical content (try a few, pick highest-variance one)
    best_idx, best_var = 0, 0.0
    for i in range(min(len(test_ds), 30)):
        s = test_ds[i]
        v = s['target'].var().item()
        if v > best_var:
            best_var = v
            best_idx = i

    sample = test_ds[best_idx]
    print(f"Using test sample index {best_idx}: {sample['filename']}")
    return model, sample, config


# --------------- XAI Methods ---------------

def compute_saliency_loss(model, sample, device):
    """Saliency map: gradient of reconstruction loss w.r.t. input."""
    model.eval()
    undersampled = sample['undersampled'].unsqueeze(0).to(device).requires_grad_(True)
    target = sample['target'].unsqueeze(0).to(device)
    kspace = sample['kspace'].unsqueeze(0).to(device)
    mask = sample['mask'].unsqueeze(0).to(device)

    recon = model(undersampled, kspace, mask)
    loss = F.l1_loss(recon, target)
    loss.backward()

    saliency = undersampled.grad.data.abs().squeeze().cpu().numpy()
    return saliency


def compute_saliency_output(model, sample, device):
    """Saliency map: gradient of summed output intensity w.r.t. input."""
    model.eval()
    undersampled = sample['undersampled'].unsqueeze(0).to(device).requires_grad_(True)
    kspace = sample['kspace'].unsqueeze(0).to(device)
    mask = sample['mask'].unsqueeze(0).to(device)

    recon = model(undersampled, kspace, mask)
    score = recon.sum()
    score.backward()

    saliency = undersampled.grad.data.abs().squeeze().cpu().numpy()
    return saliency


def compute_gradcam(model, sample, device, target_layer_name):
    """Compute Grad-CAM for a specified layer of the U-Net."""
    model.eval()
    activations = {}
    gradients = {}

    # Resolve the target module
    target_module = dict(model.named_modules())[target_layer_name]

    def forward_hook(module, inp, out):
        activations['value'] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    fh = target_module.register_forward_hook(forward_hook)
    bh = target_module.register_full_backward_hook(backward_hook)

    undersampled = sample['undersampled'].unsqueeze(0).to(device).requires_grad_(True)
    kspace = sample['kspace'].unsqueeze(0).to(device)
    mask = sample['mask'].unsqueeze(0).to(device)

    recon = model(undersampled, kspace, mask)
    score = recon.sum()
    score.backward()

    fh.remove()
    bh.remove()

    # Grad-CAM computation
    act = activations['value']       # (1, C, H', W')
    grad = gradients['value']        # (1, C, H', W')
    weights = grad.mean(dim=(2, 3), keepdim=True)  # GAP of gradients
    cam = (weights * act).sum(dim=1, keepdim=True)  # weighted combination
    cam = F.relu(cam)
    # Upsample to input size
    cam = F.interpolate(cam, size=(256, 256), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    # Normalize to [0, 1]
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def compute_integrated_gradients(model, sample, device, n_steps=50):
    """Compute Integrated Gradients w.r.t. a zero baseline."""
    model.eval()
    undersampled = sample['undersampled'].unsqueeze(0).to(device)
    kspace = sample['kspace'].unsqueeze(0).to(device)
    mask = sample['mask'].unsqueeze(0).to(device)
    baseline = torch.zeros_like(undersampled)

    integrated_grads = torch.zeros_like(undersampled)

    for step in range(n_steps + 1):
        alpha = step / n_steps
        interp = baseline + alpha * (undersampled - baseline)
        interp = interp.clone().detach().requires_grad_(True)
        recon = model(interp, kspace, mask)
        score = recon.sum()
        score.backward()
        if interp.grad is not None:
            integrated_grads += interp.grad.detach()

    # Average and scale
    integrated_grads = (undersampled - baseline) * integrated_grads / (n_steps + 1)
    ig = integrated_grads.abs().squeeze().cpu().numpy()
    return ig


# --------------- Figure Generation ---------------

def generate_figure(model, sample, device):
    """Generate the full XAI explainability figure."""
    print("Computing reconstruction...")
    model.eval()
    with torch.no_grad():
        undersampled_t = sample['undersampled'].unsqueeze(0).to(device)
        kspace_t = sample['kspace'].unsqueeze(0).to(device)
        mask_t = sample['mask'].unsqueeze(0).to(device)
        recon = model(undersampled_t, kspace_t, mask_t)

    input_img = sample['undersampled'].squeeze().cpu().numpy()
    target_img = sample['target'].squeeze().cpu().numpy()
    recon_img = recon.squeeze().cpu().numpy()
    error_map = np.abs(recon_img - target_img)

    print("Computing saliency (loss-based)...")
    saliency_loss = compute_saliency_loss(model, sample, device)

    print("Computing saliency (output-based)...")
    saliency_output = compute_saliency_output(model, sample, device)

    print("Computing Grad-CAM at multiple layers...")
    gradcam_layers = {
        'Encoder L1': 'enc1',
        'Encoder L2': 'enc2',
        'Bottleneck': 'bottleneck',
        'Decoder L2': 'dec2',
        'Decoder L1': 'dec1',
    }
    gradcam_maps = {}
    for label, layer_name in gradcam_layers.items():
        gradcam_maps[label] = compute_gradcam(model, sample, device, layer_name)
        print(f"  {label} ({layer_name}) done")

    print("Computing Integrated Gradients...")
    ig_map = compute_integrated_gradients(model, sample, device, n_steps=50)

    # Smooth the saliency/IG maps for visual clarity
    saliency_loss_smooth = gaussian_filter(saliency_loss, sigma=2)
    saliency_output_smooth = gaussian_filter(saliency_output, sigma=2)
    ig_smooth = gaussian_filter(ig_map, sigma=2)

    # --------------- Correlations between XAI methods and error ---------------
    error_flat = error_map.flatten()
    methods = {
        'Saliency\n(loss)': saliency_loss_smooth.flatten(),
        'Saliency\n(output)': saliency_output_smooth.flatten(),
        'GradCAM\nEnc L1': gradcam_maps['Encoder L1'].flatten(),
        'GradCAM\nBottleneck': gradcam_maps['Bottleneck'].flatten(),
        'GradCAM\nDec L1': gradcam_maps['Decoder L1'].flatten(),
        'Integrated\nGradients': ig_smooth.flatten(),
    }
    correlations = {}
    for name, vals in methods.items():
        r, _ = pearsonr(vals, error_flat)
        correlations[name] = r

    # --------------- Build Figure ---------------
    print("Building figure...")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
        'axes.labelsize': 9,
    })

    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    fig.suptitle('Figure 6: XAI Explainability Analysis for MRI Reconstruction',
                 fontsize=13, fontweight='bold', y=0.98)

    # ---------- Row 1: Input, Reconstruction, Error, Saliency (loss), Saliency (output) ----------
    row1_data = [
        (input_img, 'Input (Zero-filled)', 'gray', None),
        (recon_img, 'Reconstruction', 'gray', None),
        (error_map, 'Error Map', 'hot', None),
        (saliency_loss_smooth, 'Saliency (Loss-based)', 'inferno', None),
        (saliency_output_smooth, 'Saliency (Output-based)', 'inferno', None),
    ]
    for j, (img, title, cmap, _) in enumerate(row1_data):
        ax = axes[0, j]
        vmax = np.percentile(img, 99) if cmap != 'gray' else None
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')
        if cmap != 'gray':
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=7)

    # ---------- Row 2: Grad-CAM at 5 layers ----------
    for j, (label, cam) in enumerate(gradcam_maps.items()):
        ax = axes[1, j]
        # Overlay Grad-CAM on the reconstruction
        ax.imshow(recon_img, cmap='gray', vmin=0, vmax=1)
        im = ax.imshow(cam, cmap='jet', alpha=0.45, vmin=0, vmax=1)
        ax.set_title(f'Grad-CAM: {label}')
        ax.axis('off')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)

    # ---------- Row 3: IG, IG overlay, correlation bar chart (spans 3 cols) ----------
    # IG heatmap
    ax = axes[2, 0]
    vmax_ig = np.percentile(ig_smooth, 99)
    im = ax.imshow(ig_smooth, cmap='inferno', vmin=0, vmax=vmax_ig)
    ax.set_title('Integrated Gradients')
    ax.axis('off')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)

    # IG overlay on reconstruction
    ax = axes[2, 1]
    ax.imshow(recon_img, cmap='gray', vmin=0, vmax=1)
    ig_norm = ig_smooth / (ig_smooth.max() + 1e-8)
    im = ax.imshow(ig_norm, cmap='magma', alpha=0.5, vmin=0, vmax=1)
    ax.set_title('IG Overlay')
    ax.axis('off')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)

    # Correlation bar chart spanning last 3 columns
    # Remove individual axes and create a merged axis
    for k in range(2, 5):
        axes[2, k].remove()
    gs = axes[2, 2].get_gridspec()
    ax_bar = fig.add_subplot(gs[2, 2:5])

    names = list(correlations.keys())
    vals = [correlations[n] for n in names]
    # Color bars: positive=warm, negative=cool
    val_min, val_max = min(vals), max(vals)
    norm_vals = [(v - val_min) / (val_max - val_min + 1e-8) for v in vals]
    colors = plt.cm.RdYlGn_r(norm_vals)
    bars = ax_bar.bar(names, vals, color=colors, edgecolor='black', linewidth=0.7)
    ax_bar.set_ylabel('Pearson Correlation with Error', fontsize=9)
    ax_bar.set_title('XAI Method vs. Reconstruction Error Correlation', fontweight='bold', fontsize=10)
    y_lo = min(0, val_min * 1.3)
    y_hi = max(0, val_max * 1.25)
    ax_bar.set_ylim(y_lo, y_hi)
    ax_bar.axhline(0, color='black', linewidth=0.5, linestyle='-')
    ax_bar.tick_params(axis='x', labelsize=8)
    ax_bar.tick_params(axis='y', labelsize=8)
    # Add value labels on bars
    for bar, v in zip(bars, vals):
        offset = 0.008 if v >= 0 else -0.008
        va = 'bottom' if v >= 0 else 'top'
        y_pos = bar.get_height() + offset if v >= 0 else bar.get_y() + bar.get_height() - 0.008
        ax_bar.text(bar.get_x() + bar.get_width() / 2, v + offset,
                    f'{v:.3f}', ha='center', va=va, fontsize=7.5, fontweight='bold')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    # Row labels
    row_labels = [
        '(a) Reconstruction Quality & Saliency Analysis',
        '(b) Grad-CAM Activation Maps Across Network Depth',
        '(c) Integrated Gradients & XAI-Error Correlation',
    ]
    for i, label in enumerate(row_labels):
        fig.text(0.02, 0.88 - i * 0.32, label, fontsize=10, fontweight='bold',
                 rotation=90, va='center', ha='center')

    plt.tight_layout(rect=[0.03, 0.01, 1.0, 0.95])

    # Save
    os.makedirs('latex/figures', exist_ok=True)
    png_path = 'latex/figures/fig6_xai_explainability.png'
    pdf_path = 'latex/figures/fig6_xai_explainability.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, sample, config = load_model_and_sample(device)
    psnr_val = compute_psnr(
        model(sample['undersampled'].unsqueeze(0).to(device),
              sample['kspace'].unsqueeze(0).to(device),
              sample['mask'].unsqueeze(0).to(device)),
        sample['target'].unsqueeze(0).to(device),
    ).item()
    print(f"Sample PSNR: {psnr_val:.2f} dB")

    generate_figure(model, sample, device)
    print("Done.")


if __name__ == '__main__':
    main()
