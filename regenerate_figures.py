#!/usr/bin/env python3
"""
Regenerate ALL figures for the MRI reconstruction paper with tight subplot spacing.
Saves each figure as both PDF and PNG to latex/figures/.
"""
import os, sys, json
os.chdir('/root/IX-Medical-Imaging')
sys.path.insert(0, 'src')

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr

from data import get_dataloaders, MRIReconDataset
from models import ReconUNet, SegmentationUNet
from losses import compute_psnr, compute_ssim

# ============================================================
# Global config
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = 'data/processed_data'
CKPT_DIR = 'checkpoints'
FIG_DIR  = 'latex/figures'
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
})

CB_KW = dict(fraction=0.04, pad=0.02)  # thin colorbars

def save_fig(fig, name):
    """Save figure as both PDF and PNG."""
    fig.savefig(f'{FIG_DIR}/{name}.pdf', bbox_inches='tight', dpi=300, facecolor='white')
    fig.savefig(f'{FIG_DIR}/{name}.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    print(f'  Saved {name}.pdf + .png')


def load_model(acceleration=4):
    path = os.path.join(CKPT_DIR, f'final_R{acceleration}', f'best_model_R{acceleration}.pth')
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    config = ckpt['config']
    m = ReconUNet(
        base_features=config['base_features'],
        dropout_rate=config['dropout_rate'],
        use_dc=True,
        num_dc_cascades=config.get('num_dc_cascades', 1),
    ).to(DEVICE)
    m.load_state_dict(ckpt['model_state_dict'])
    return m, config


def load_ensemble(n=3, accel=4):
    models = []
    for i in range(n):
        path = os.path.join(CKPT_DIR, f'ensemble_{i}_R{accel}', f'best_model_R{accel}.pth')
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        config = ckpt['config']
        m = ReconUNet(
            base_features=config['base_features'],
            dropout_rate=config['dropout_rate'],
            use_dc=True,
            num_dc_cascades=config.get('num_dc_cascades', 1),
        ).to(DEVICE)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        models.append(m)
    return models


# ============================================================
# Load models and data once
# ============================================================
print("Loading models...")
model_R4, cfg4 = load_model(4)
model_R8, _    = load_model(8)
ensemble_models = load_ensemble(3, 4)
seg_model = SegmentationUNet(in_channels=1, num_classes=8, base_features=32).to(DEVICE)
seg_model.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'seg_model.pth'),
                                      map_location=DEVICE, weights_only=True))
seg_model.eval()

print("Loading data...")
_, _, test_loader_R4 = get_dataloaders(DATA_ROOT, 'mr', acceleration=4, batch_size=1, num_workers=2)
_, _, test_loader_R8 = get_dataloaders(DATA_ROOT, 'mr', acceleration=8, batch_size=1, num_workers=2)

# Pick a good sample (skip 5)
test_iter4 = iter(test_loader_R4)
test_iter8 = iter(test_loader_R8)
for _ in range(5):
    batch4 = next(test_iter4)
    batch8 = next(test_iter8)

# Load JSON results
def load_json(name):
    with open(os.path.join(CKPT_DIR, name)) as f:
        return json.load(f)

multi_accel   = load_json('multi_accel_results.json')
adv_results   = load_json('adversarial_results.json')
cross_domain  = load_json('cross_domain_results.json')
dice_results  = load_json('dice_results.json')
ens_results   = load_json('ensemble_comparison_results.json')
perturb       = load_json('perturbation_results.json')
test_results  = load_json('test_results.json')


# ============================================================
# Helper: MC Dropout inference
# ============================================================
def mc_dropout_predict(model, us, ks, msk, T=20):
    model.train()  # enable dropout
    preds = []
    with torch.no_grad():
        for _ in range(T):
            p = model(us, ks, msk)
            preds.append(p)
    preds = torch.stack(preds)
    mean = preds.mean(0)
    std  = preds.std(0)
    model.eval()
    return mean, std


def ensemble_predict(models, us, ks, msk):
    preds = []
    with torch.no_grad():
        for m in models:
            m.eval()
            preds.append(m(us, ks, msk))
    preds = torch.stack(preds)
    return preds.mean(0), preds.std(0)


# ============================================================
# Run inference for figures that need it
# ============================================================
print("Running inference for fig4...")
model_R4.eval(); model_R8.eval()
with torch.no_grad():
    pred4 = model_R4(batch4['undersampled'].to(DEVICE), batch4['kspace'].to(DEVICE), batch4['mask'].to(DEVICE))
    pred8 = model_R8(batch8['undersampled'].to(DEVICE), batch8['kspace'].to(DEVICE), batch8['mask'].to(DEVICE))

gt4  = batch4['target'][0,0].numpy()
zf4  = batch4['undersampled'][0,0].numpy()
rec4 = pred4[0,0].cpu().numpy()
gt8  = batch8['target'][0,0].numpy()  # same dataset, different mask
zf8  = batch8['undersampled'][0,0].numpy()
rec8 = pred8[0,0].cpu().numpy()

print("Running MC Dropout + Ensemble inference for fig13/14/15...")
# Collect over multiple test samples
N_SAMPLES = 20
mc_means_list, mc_stds_list = [], []
ens_means_list, ens_stds_list = [], []
targets_list, errors_mc_list, errors_ens_list = [], [], []
seg_error_maps_list = []

test_iter = iter(test_loader_R4)
for i in range(N_SAMPLES):
    b = next(test_iter)
    us_i = b['undersampled'].to(DEVICE)
    ks_i = b['kspace'].to(DEVICE)
    msk_i = b['mask'].to(DEVICE)
    tgt_i = b['target'].to(DEVICE)
    lbl_i = b['label'].to(DEVICE)

    mc_mean, mc_std = mc_dropout_predict(model_R4, us_i, ks_i, msk_i, T=20)
    ens_mean, ens_std = ensemble_predict(ensemble_models, us_i, ks_i, msk_i)

    mc_means_list.append(mc_mean.cpu())
    mc_stds_list.append(mc_std.cpu())
    ens_means_list.append(ens_mean.cpu())
    ens_stds_list.append(ens_std.cpu())
    targets_list.append(tgt_i.cpu())
    errors_mc_list.append((mc_mean - tgt_i).abs().cpu())
    errors_ens_list.append((ens_mean - tgt_i).abs().cpu())

    # Seg error
    with torch.no_grad():
        seg_gt = seg_model(tgt_i).argmax(1)
        seg_mc = seg_model(mc_mean.float()).argmax(1)
    seg_err = (seg_mc != lbl_i).float().cpu().numpy().squeeze()
    seg_error_maps_list.append(seg_err)

mc_stds_all   = torch.cat(mc_stds_list).squeeze().numpy()
ens_stds_all  = torch.cat(ens_stds_list).squeeze().numpy()
errors_mc_all = torch.cat(errors_mc_list).squeeze().numpy()
errors_ens_all= torch.cat(errors_ens_list).squeeze().numpy()
targets_all   = torch.cat(targets_list).squeeze().numpy()
mc_means_all  = torch.cat(mc_means_list).squeeze().numpy()
ens_means_all = torch.cat(ens_means_list).squeeze().numpy()

# Pick sample_idx with visible features
sample_idx = 5

# Compute calibration metrics
def compute_ece(uncertainty, error, n_bins=15):
    unc_flat = uncertainty.flatten()
    err_flat = error.flatten()
    unc_norm = unc_flat / (unc_flat.max() + 1e-10)
    err_norm = err_flat / (err_flat.max() + 1e-10)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(unc_norm)
    for i in range(n_bins):
        mask = (unc_norm >= bins[i]) & (unc_norm < bins[i+1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / total) * abs(unc_norm[mask].mean() - err_norm[mask].mean())
    return ece

def compute_ause(uncertainty, error, n_fractions=50):
    unc_flat = uncertainty.flatten()
    err_flat = (error.flatten()) ** 2
    unc_sorted_idx = np.argsort(-unc_flat)
    oracle_sorted_idx = np.argsort(-err_flat)
    fractions = np.linspace(0, 0.95, n_fractions)
    unc_mse, oracle_mse = [], []
    for f in fractions:
        n_remove = int(f * len(unc_flat))
        if n_remove == 0:
            unc_mse.append(err_flat.mean())
            oracle_mse.append(err_flat.mean())
        else:
            remaining_unc = np.delete(err_flat, unc_sorted_idx[:n_remove])
            remaining_oracle = np.delete(err_flat, oracle_sorted_idx[:n_remove])
            unc_mse.append(remaining_unc.mean() if len(remaining_unc) > 0 else 0)
            oracle_mse.append(remaining_oracle.mean() if len(remaining_oracle) > 0 else 0)
    ause = np.trapezoid(unc_mse, fractions) - np.trapezoid(oracle_mse, fractions)
    return ause, fractions, unc_mse, oracle_mse

mc_ece = compute_ece(mc_stds_all, errors_mc_all)
ens_ece = compute_ece(ens_stds_all, errors_ens_all)
mc_ause, mc_fracs, mc_spars, mc_oracle = compute_ause(mc_stds_all, errors_mc_all)
ens_ause, ens_fracs, ens_spars, ens_oracle = compute_ause(ens_stds_all, errors_ens_all)
mc_pearson = np.corrcoef(mc_stds_all.flatten(), errors_mc_all.flatten())[0, 1]
ens_pearson = np.corrcoef(ens_stds_all.flatten(), errors_ens_all.flatten())[0, 1]

# Random baseline for sparsification
err_flat_all = (errors_mc_all.flatten()) ** 2
random_spars = [err_flat_all.mean()] * 50

# seg correlation
seg_correlations_mc = []
for i in range(N_SAMPLES):
    mc_unc = mc_stds_all[i]
    seg_err = seg_error_maps_list[i]
    if seg_err.std() > 0 and mc_unc.std() > 0:
        r = np.corrcoef(mc_unc.flatten(), seg_err.flatten())[0, 1]
        seg_correlations_mc.append(r)
mean_seg_corr_mc = np.mean(seg_correlations_mc) if seg_correlations_mc else 0.0

print("Inference done. Generating figures...\n")


# ============================================================
# FIG 4: Reconstruction Comparison (main paper)
# ============================================================
print("Fig 4: reconstruction comparison")
fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
plt.subplots_adjust(hspace=0.12, wspace=0.08)

psnr_zf4 = 10*np.log10(1/(np.mean((zf4-gt4)**2)+1e-10))
psnr_r4  = 10*np.log10(1/(np.mean((rec4-gt4)**2)+1e-10))
psnr_zf8 = 10*np.log10(1/(np.mean((zf8-gt8)**2)+1e-10))
psnr_r8  = 10*np.log10(1/(np.mean((rec8-gt8)**2)+1e-10))

# Row 1: images
axes[0,0].imshow(gt4, cmap='gray'); axes[0,0].set_title('Ground Truth')
axes[0,1].imshow(zf4, cmap='gray'); axes[0,1].set_title(f'ZF R=4 ({psnr_zf4:.1f} dB)')
axes[0,2].imshow(rec4, cmap='gray'); axes[0,2].set_title(f'Ours R=4 ({psnr_r4:.1f} dB)')
axes[0,3].imshow(zf8, cmap='gray'); axes[0,3].set_title(f'ZF R=8 ({psnr_zf8:.1f} dB)')
axes[0,4].imshow(rec8, cmap='gray'); axes[0,4].set_title(f'Ours R=8 ({psnr_r8:.1f} dB)')

# Row 2: error maps
vmax = 0.15
axes[1,0].axis('off')
for j, (err, lbl) in enumerate([
    (np.abs(zf4-gt4),  'Error ZF R=4'),
    (np.abs(rec4-gt4), 'Error Ours R=4'),
    (np.abs(zf8-gt8),  'Error ZF R=8'),
    (np.abs(rec8-gt8), 'Error Ours R=8'),
], start=1):
    im = axes[1,j].imshow(err, cmap='hot', vmin=0, vmax=vmax)
    axes[1,j].set_title(lbl)

for ax in axes.flat:
    ax.axis('off')

# Single thin colorbar for error row
cbar_ax = fig.add_axes([0.92, 0.05, 0.012, 0.4])
fig.colorbar(im, cax=cbar_ax)

save_fig(fig, 'fig4_reconstruction_comparison')


# ============================================================
# FIG 13: Ensemble vs MC Dropout (main paper)
# ============================================================
print("Fig 13: ensemble vs MC dropout")
fig = plt.figure(figsize=(10, 6.5))
gs = gridspec.GridSpec(3, 4, hspace=0.18, wspace=0.12, height_ratios=[1, 1, 1.2])

s = sample_idx
gt_s = targets_all[s]
mc_rec_s = mc_means_all[s]
mc_unc_s = mc_stds_all[s]
mc_err_s = errors_mc_all[s]
ens_rec_s = ens_means_all[s]
ens_unc_s = ens_stds_all[s]
ens_err_s = errors_ens_all[s]

# Row 1: MC Dropout
titles1 = ['Ground Truth', 'MC Mean', 'MC Uncertainty', 'Abs. Error']
imgs1   = [gt_s, mc_rec_s, mc_unc_s, mc_err_s]
cmaps1  = ['gray', 'gray', 'magma', 'hot']
for j, (img, title, cmap) in enumerate(zip(imgs1, titles1, cmaps1)):
    ax = fig.add_subplot(gs[0, j])
    im = ax.imshow(img, cmap=cmap)
    ax.set_title(title); ax.axis('off')
    if j >= 2:
        plt.colorbar(im, ax=ax, **CB_KW)

# Row 2: Ensemble
titles2 = ['Ground Truth', 'Ensemble Mean', 'Ens. Uncertainty', 'Abs. Error']
imgs2   = [gt_s, ens_rec_s, ens_unc_s, ens_err_s]
cmaps2  = ['gray', 'gray', 'magma', 'hot']
for j, (img, title, cmap) in enumerate(zip(imgs2, titles2, cmaps2)):
    ax = fig.add_subplot(gs[1, j])
    im = ax.imshow(img, cmap=cmap)
    ax.set_title(title); ax.axis('off')
    if j >= 2:
        plt.colorbar(im, ax=ax, **CB_KW)

# Row 3: Sparsification + Scatter
ax_sp = fig.add_subplot(gs[2, :2])
ax_sp.plot(mc_fracs, mc_spars, 'b-', lw=1.5, label='MC Dropout')
ax_sp.plot(ens_fracs, ens_spars, 'r-', lw=1.5, label='Deep Ensemble')
ax_sp.plot(mc_fracs, mc_oracle, 'k--', lw=1, label='Oracle', alpha=0.7)
ax_sp.plot(mc_fracs, random_spars, 'gray', lw=1, ls=':', label='Random', alpha=0.7)
ax_sp.set_xlabel('Fraction removed'); ax_sp.set_ylabel('Residual MSE')
ax_sp.set_title('Sparsification'); ax_sp.legend(fontsize=7, loc='upper right')
ax_sp.grid(True, alpha=0.3)

ax_sc = fig.add_subplot(gs[2, 2:])
n_sub = 5000
rng = np.random.RandomState(42)
idx_sub = rng.choice(mc_stds_all.flatten().shape[0], n_sub, replace=False)
ax_sc.scatter(mc_stds_all.flatten()[idx_sub], errors_mc_all.flatten()[idx_sub],
              alpha=0.15, s=3, c='blue', label=f'MC (r={mc_pearson:.3f})')
ax_sc.scatter(ens_stds_all.flatten()[idx_sub], errors_ens_all.flatten()[idx_sub],
              alpha=0.15, s=3, c='red', label=f'Ens (r={ens_pearson:.3f})')
ax_sc.set_xlabel('Predicted Uncertainty'); ax_sc.set_ylabel('Abs. Error')
ax_sc.set_title('Uncertainty vs Error'); ax_sc.legend(fontsize=7, markerscale=5)
ax_sc.grid(True, alpha=0.3)

save_fig(fig, 'fig13_ensemble_vs_mcdropout')


# ============================================================
# FIG 14: Reliability Diagrams (main paper)
# ============================================================
print("Fig 14: reliability diagrams")
fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
plt.subplots_adjust(wspace=0.25)

for ax, unc_data, err_data, method, color, ece_val in [
    (axes[0], mc_stds_all, errors_mc_all, 'MC Dropout', 'blue', mc_ece),
    (axes[1], ens_stds_all, errors_ens_all, 'Deep Ensemble', 'red', ens_ece),
]:
    n_bins = 15
    unc_flat = unc_data.flatten()
    err_flat = err_data.flatten()
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(unc_flat, percentiles)

    bmu, bme = [], []
    for b_i in range(n_bins):
        mask = (unc_flat >= bin_edges[b_i]) & (unc_flat < bin_edges[b_i+1] + 1e-10)
        if mask.sum() > 0:
            bmu.append(unc_flat[mask].mean())
            bme.append(err_flat[mask].mean())
    bmu = np.array(bmu); bme = np.array(bme)
    mx = max(bmu.max(), bme.max())
    bmu_n = bmu / (mx + 1e-10)
    bme_n = bme / (mx + 1e-10)

    n = len(bmu_n)
    ax.bar(range(n), bme_n, alpha=0.5, color=color, label='Observed error', width=0.8)
    ax.plot(range(n), bmu_n, 'o-', color=color, ms=4, lw=1.5, label='Predicted unc.')
    diag = np.linspace(0, 1, n)
    ax.plot(range(n), diag, 'k--', alpha=0.3, lw=1, label='Perfect')
    ax.set_xlabel('Uncertainty bin', fontsize=10)
    ax.set_ylabel('Normalised value', fontsize=10)
    ax.set_title(f'{method} (ECE={ece_val:.4f})', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

save_fig(fig, 'fig14_reliability_diagrams')


# ============================================================
# FIG 15: Uncertainty-Segmentation Correlation (main paper)
# ============================================================
print("Fig 15: uncertainty-seg correlation")
fig, axes = plt.subplots(1, 5, figsize=(11, 2.6))
plt.subplots_adjust(wspace=0.12)

s = sample_idx
gt_s = targets_all[s]
mc_rec_s = mc_means_all[s]
mc_unc_s = mc_stds_all[s]
seg_err_s = seg_error_maps_list[s]

axes[0].imshow(gt_s, cmap='gray'); axes[0].set_title('Ground Truth'); axes[0].axis('off')
axes[1].imshow(mc_rec_s, cmap='gray'); axes[1].set_title('Reconstruction'); axes[1].axis('off')
im2 = axes[2].imshow(mc_unc_s, cmap='magma')
axes[2].set_title('Uncertainty'); axes[2].axis('off')
plt.colorbar(im2, ax=axes[2], **CB_KW)
im3 = axes[3].imshow(seg_err_s, cmap='Reds')
axes[3].set_title('Seg. Error'); axes[3].axis('off')
plt.colorbar(im3, ax=axes[3], **CB_KW)
axes[4].imshow(gt_s, cmap='gray', alpha=0.5)
axes[4].imshow(mc_unc_s, cmap='magma', alpha=0.3)
axes[4].contour(seg_err_s, levels=[0.5], colors='red', linewidths=0.8)
axes[4].set_title(f'Overlay (r={mean_seg_corr_mc:.3f})'); axes[4].axis('off')

save_fig(fig, 'fig15_uncertainty_seg_correlation')


# ============================================================
# FIG A1: Dataset Overview (appendix, renamed from fig1)
# ============================================================
print("Fig A1: dataset overview")
test_ds_full = MRIReconDataset('data/processed_data/mr_256/test/npz',
                                acceleration=4, fixed_masks=True)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
plt.subplots_adjust(hspace=0.08, wspace=0.06)

for i in range(5):
    s = test_ds_full[i * 8]
    img = s['target'].squeeze().numpy()
    lbl = s['label'].numpy()
    axes[0, i].imshow(img, cmap='gray'); axes[0, i].set_title(f'MR Slice {i+1}'); axes[0, i].axis('off')
    axes[1, i].imshow(lbl, cmap='tab10', vmin=0, vmax=7); axes[1, i].set_title('Segmentation'); axes[1, i].axis('off')

axes[0, 0].set_ylabel('MR Image', fontsize=10, fontweight='bold')
axes[1, 0].set_ylabel('Label Map', fontsize=10, fontweight='bold')

save_fig(fig, 'figA1_dataset_overview')


# ============================================================
# FIG A2: Quality vs Acceleration (appendix, renamed from fig5)
# ============================================================
print("Fig A2: quality vs acceleration")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
plt.subplots_adjust(wspace=0.28)

Rs = sorted([int(k) for k in multi_accel.keys()])
psnr_means = [multi_accel[str(r)]['psnr_mean'] for r in Rs]
psnr_stds  = [multi_accel[str(r)]['psnr_std'] for r in Rs]
ssim_means = [multi_accel[str(r)]['ssim_mean'] for r in Rs]
ssim_stds  = [multi_accel[str(r)]['ssim_std'] for r in Rs]
zf_psnr    = [multi_accel[str(r)]['zf_psnr_mean'] for r in Rs]
zf_ssim    = [multi_accel[str(r)]['zf_ssim_mean'] for r in Rs]

ax1.errorbar(Rs, psnr_means, yerr=psnr_stds, fmt='o-', color='#2196F3', lw=2, capsize=4, ms=5, label='Ours')
ax1.plot(Rs, zf_psnr, 's--', color='#9E9E9E', lw=1.5, ms=5, label='Zero-filled')
ax1.set_xlabel('Acceleration R'); ax1.set_ylabel('PSNR (dB)')
ax1.set_title('PSNR vs Acceleration'); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
ax1.set_xticks(Rs)

ax2.errorbar(Rs, ssim_means, yerr=ssim_stds, fmt='o-', color='#2196F3', lw=2, capsize=4, ms=5, label='Ours')
ax2.plot(Rs, zf_ssim, 's--', color='#9E9E9E', lw=1.5, ms=5, label='Zero-filled')
ax2.set_xlabel('Acceleration R'); ax2.set_ylabel('SSIM')
ax2.set_title('SSIM vs Acceleration'); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
ax2.set_xticks(Rs)

save_fig(fig, 'figA2_quality_vs_acceleration')


# ============================================================
# FIG A3: XAI Explainability (appendix, renamed from fig6)
# ============================================================
print("Fig A3: XAI explainability (computing saliency, Grad-CAM, IG...)")

# Pick best-variance sample
test_ds_xai = MRIReconDataset('data/processed_data/mr_256/test/npz',
                               acceleration=4,
                               center_fraction=cfg4.get('center_fraction', 0.08),
                               fixed_masks=True)
best_idx, best_var = 0, 0.0
for i in range(min(len(test_ds_xai), 30)):
    v = test_ds_xai[i]['target'].var().item()
    if v > best_var:
        best_var = v; best_idx = i
xai_sample = test_ds_xai[best_idx]

model_R4.eval()
with torch.no_grad():
    us_t = xai_sample['undersampled'].unsqueeze(0).to(DEVICE)
    ks_t = xai_sample['kspace'].unsqueeze(0).to(DEVICE)
    msk_t = xai_sample['mask'].unsqueeze(0).to(DEVICE)
    recon_t = model_R4(us_t, ks_t, msk_t)

input_img  = xai_sample['undersampled'].squeeze().cpu().numpy()
target_img = xai_sample['target'].squeeze().cpu().numpy()
recon_img  = recon_t.squeeze().cpu().numpy()
error_map  = np.abs(recon_img - target_img)

# Saliency (loss)
model_R4.eval()
us_g = xai_sample['undersampled'].unsqueeze(0).to(DEVICE).requires_grad_(True)
tgt_g = xai_sample['target'].unsqueeze(0).to(DEVICE)
ks_g = xai_sample['kspace'].unsqueeze(0).to(DEVICE)
msk_g = xai_sample['mask'].unsqueeze(0).to(DEVICE)
out_g = model_R4(us_g, ks_g, msk_g)
F.l1_loss(out_g, tgt_g).backward()
saliency_loss = us_g.grad.data.abs().squeeze().cpu().numpy()

# Saliency (output)
model_R4.eval()
us_g2 = xai_sample['undersampled'].unsqueeze(0).to(DEVICE).requires_grad_(True)
out_g2 = model_R4(us_g2, ks_g, msk_g)
out_g2.sum().backward()
saliency_output = us_g2.grad.data.abs().squeeze().cpu().numpy()

# Grad-CAM
def compute_gradcam(model, sample, device, layer_name):
    model.eval()
    acts, grads = {}, {}
    target_mod = dict(model.named_modules())[layer_name]
    fh = target_mod.register_forward_hook(lambda m, i, o: acts.update(value=o.detach()))
    bh = target_mod.register_full_backward_hook(lambda m, gi, go: grads.update(value=go[0].detach()))
    us_l = sample['undersampled'].unsqueeze(0).to(device).requires_grad_(True)
    ks_l = sample['kspace'].unsqueeze(0).to(device)
    msk_l = sample['mask'].unsqueeze(0).to(device)
    r = model(us_l, ks_l, msk_l)
    r.sum().backward()
    fh.remove(); bh.remove()
    w = grads['value'].mean(dim=(2,3), keepdim=True)
    cam = F.relu((w * acts['value']).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(256,256), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    if cam.max() > 0: cam = cam / cam.max()
    return cam

gc_layers = {'Enc L1':'enc1', 'Enc L2':'enc2', 'Bottleneck':'bottleneck', 'Dec L2':'dec2', 'Dec L1':'dec1'}
gc_maps = {k: compute_gradcam(model_R4, xai_sample, DEVICE, v) for k, v in gc_layers.items()}

# Integrated Gradients
model_R4.eval()
us_ig = xai_sample['undersampled'].unsqueeze(0).to(DEVICE)
baseline_ig = torch.zeros_like(us_ig)
ig_acc = torch.zeros_like(us_ig)
n_steps = 50
for step in range(n_steps + 1):
    alpha = step / n_steps
    interp = (baseline_ig + alpha * (us_ig - baseline_ig)).clone().detach().requires_grad_(True)
    r = model_R4(interp, ks_g, msk_g)
    r.sum().backward()
    if interp.grad is not None:
        ig_acc += interp.grad.detach()
ig_map = ((us_ig - baseline_ig) * ig_acc / (n_steps + 1)).abs().squeeze().cpu().numpy()

# Smooth
saliency_loss_s  = gaussian_filter(saliency_loss, sigma=2)
saliency_output_s = gaussian_filter(saliency_output, sigma=2)
ig_smooth = gaussian_filter(ig_map, sigma=2)

# Correlations
error_flat_xai = error_map.flatten()
xai_methods = {
    'Saliency\n(loss)': saliency_loss_s.flatten(),
    'Saliency\n(output)': saliency_output_s.flatten(),
    'GradCAM\nEnc L1': gc_maps['Enc L1'].flatten(),
    'GradCAM\nBottleneck': gc_maps['Bottleneck'].flatten(),
    'GradCAM\nDec L1': gc_maps['Dec L1'].flatten(),
    'Integ.\nGradients': ig_smooth.flatten(),
}
corrs = {k: pearsonr(v, error_flat_xai)[0] for k, v in xai_methods.items()}

# Build figure
fig, axes = plt.subplots(3, 5, figsize=(13, 8))
plt.subplots_adjust(hspace=0.15, wspace=0.10)

# Row 1
r1_data = [
    (input_img, 'Input (ZF)', 'gray'),
    (recon_img, 'Reconstruction', 'gray'),
    (error_map, 'Error Map', 'hot'),
    (saliency_loss_s, 'Saliency (Loss)', 'inferno'),
    (saliency_output_s, 'Saliency (Output)', 'inferno'),
]
for j, (img, title, cmap) in enumerate(r1_data):
    ax = axes[0, j]
    vmax_r = np.percentile(img, 99) if cmap != 'gray' else None
    im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax_r)
    ax.set_title(title); ax.axis('off')
    if cmap != 'gray':
        fig.colorbar(im, ax=ax, **CB_KW)

# Row 2: Grad-CAM
for j, (label, cam) in enumerate(gc_maps.items()):
    ax = axes[1, j]
    ax.imshow(recon_img, cmap='gray', vmin=0, vmax=1)
    im = ax.imshow(cam, cmap='jet', alpha=0.45, vmin=0, vmax=1)
    ax.set_title(f'GradCAM: {label}'); ax.axis('off')
    fig.colorbar(im, ax=ax, **CB_KW)

# Row 3: IG heatmap, IG overlay, bar chart (span 3 cols)
ax = axes[2, 0]
vmax_ig_r = np.percentile(ig_smooth, 99)
im = ax.imshow(ig_smooth, cmap='inferno', vmin=0, vmax=vmax_ig_r)
ax.set_title('Integ. Gradients'); ax.axis('off')
fig.colorbar(im, ax=ax, **CB_KW)

ax = axes[2, 1]
ax.imshow(recon_img, cmap='gray', vmin=0, vmax=1)
ig_norm = ig_smooth / (ig_smooth.max() + 1e-8)
im = ax.imshow(ig_norm, cmap='magma', alpha=0.5, vmin=0, vmax=1)
ax.set_title('IG Overlay'); ax.axis('off')
fig.colorbar(im, ax=ax, **CB_KW)

# Remove axes 2,2-4 and create merged bar chart
for k in range(2, 5):
    axes[2, k].remove()
gs_xai = axes[2, 2].get_gridspec()
ax_bar = fig.add_subplot(gs_xai[2, 2:5])

names = list(corrs.keys())
vals = [corrs[n] for n in names]
vmin_c, vmax_c = min(vals), max(vals)
norm_c = [(v - vmin_c)/(vmax_c - vmin_c + 1e-8) for v in vals]
colors_c = plt.cm.RdYlGn_r(norm_c)
bars = ax_bar.bar(names, vals, color=colors_c, edgecolor='black', linewidth=0.7)
ax_bar.set_ylabel('Pearson r with Error')
ax_bar.set_title('XAI-Error Correlation')
ax_bar.set_ylim(min(0, vmin_c*1.3), max(0, vmax_c*1.25))
ax_bar.axhline(0, color='black', lw=0.5)
ax_bar.tick_params(axis='x', labelsize=7)
for bar, v in zip(bars, vals):
    off = 0.008 if v >= 0 else -0.008
    ax_bar.text(bar.get_x()+bar.get_width()/2, v+off, f'{v:.3f}',
                ha='center', va='bottom' if v>=0 else 'top', fontsize=7)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)

save_fig(fig, 'figA3_explainability_analysis')


# ============================================================
# FIG A4: Adversarial Robustness (appendix, renamed from fig9)
# ============================================================
print("Fig A4: adversarial robustness")
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
plt.subplots_adjust(wspace=0.30)

eps = adv_results['epsilon']

axes[0].plot(eps, adv_results['clean_psnr'], 'k--', lw=2, label='Clean', marker='*', ms=7)
axes[0].plot(eps, adv_results['fgsm_psnr'], 'r-o', lw=2, label='FGSM', ms=5)
axes[0].plot(eps, adv_results['pgd_psnr'], 'b-s', lw=2, label='PGD', ms=5)
axes[0].set_xlabel(r'$\epsilon$'); axes[0].set_ylabel('PSNR (dB)')
axes[0].set_title('(a) PSNR vs Attack'); axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

axes[1].plot(eps, adv_results['clean_ssim'], 'k--', lw=2, label='Clean', marker='*', ms=7)
axes[1].plot(eps, adv_results['fgsm_ssim'], 'r-o', lw=2, label='FGSM', ms=5)
axes[1].plot(eps, adv_results['pgd_ssim'], 'b-s', lw=2, label='PGD', ms=5)
axes[1].set_xlabel(r'$\epsilon$'); axes[1].set_ylabel('SSIM')
axes[1].set_title('(b) SSIM vs Attack'); axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

axes[2].plot(eps[1:], adv_results['fgsm_unc'][1:], 'r-o', lw=2, label='FGSM', ms=5)
axes[2].plot(eps[1:], adv_results['pgd_unc'][1:], 'b-s', lw=2, label='PGD', ms=5)
axes[2].set_xlabel(r'$\epsilon$'); axes[2].set_ylabel('Mean Uncertainty')
axes[2].set_title('(c) Uncertainty Response'); axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

save_fig(fig, 'figA4_adversarial_robustness')


# ============================================================
# FIG A5: MC Dropout Detail (appendix, renamed from fig7)
# ============================================================
print("Fig A5: MC dropout detail")
s = sample_idx
gt_mc = targets_all[s]
mc_mean_np = mc_means_all[s]
mc_std_np  = mc_stds_all[s]
error_mc_np = errors_mc_all[s]

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
plt.subplots_adjust(hspace=0.18, wspace=0.12)

axes[0,0].imshow(gt_mc, cmap='gray'); axes[0,0].set_title('(a) Ground Truth'); axes[0,0].axis('off')

psnr_mc_s = 10*np.log10(1/(np.mean((mc_mean_np - gt_mc)**2)+1e-10))
axes[0,1].imshow(mc_mean_np, cmap='gray')
axes[0,1].set_title(f'(b) MC Mean ({psnr_mc_s:.1f} dB)'); axes[0,1].axis('off')

im = axes[0,2].imshow(mc_std_np, cmap='magma')
plt.colorbar(im, ax=axes[0,2], **CB_KW)
axes[0,2].set_title('(c) Uncertainty'); axes[0,2].axis('off')

im = axes[0,3].imshow(error_mc_np, cmap='hot')
plt.colorbar(im, ax=axes[0,3], **CB_KW)
axes[0,3].set_title('(d) Abs. Error'); axes[0,3].axis('off')

# Row 2
axes[1,0].imshow(gt_mc, cmap='gray', alpha=0.5)
axes[1,0].imshow(mc_std_np, cmap='magma', alpha=0.5)
axes[1,0].set_title('(e) Unc. Overlay'); axes[1,0].axis('off')

# Scatter
idx_mc = np.random.RandomState(42).choice(mc_std_np.size, 5000, replace=False)
axes[1,1].scatter(mc_std_np.flatten()[idx_mc], error_mc_np.flatten()[idx_mc], alpha=0.2, s=2, c='#2196F3')
unc_err_r = np.corrcoef(mc_std_np.flatten(), error_mc_np.flatten())[0,1]
z = np.polyfit(mc_std_np.flatten(), error_mc_np.flatten(), 1)
p_line = np.poly1d(z)
x_ln = np.linspace(mc_std_np.min(), mc_std_np.max(), 100)
axes[1,1].plot(x_ln, p_line(x_ln), 'r-', lw=2, label='Linear fit')
axes[1,1].set_xlabel('Uncertainty'); axes[1,1].set_ylabel('Abs. Error')
axes[1,1].set_title(f'(f) Corr: r={unc_err_r:.3f}'); axes[1,1].legend(fontsize=7)

# Sparsification
flat_err_mc = error_mc_np.flatten()
flat_unc_mc = mc_std_np.flatten()
fracs_mc = np.linspace(0, 0.95, 50)
oracle_mse_mc, unc_mse_mc, rand_mse_mc = [], [], []
rng_mc = np.random.RandomState(42)
for f in fracs_mc:
    nr = int(f * len(flat_err_mc))
    if nr == 0:
        oracle_mse_mc.append(np.mean(flat_err_mc**2))
        unc_mse_mc.append(np.mean(flat_err_mc**2))
        rand_mse_mc.append(np.mean(flat_err_mc**2))
    else:
        oi = np.argsort(flat_err_mc)[:len(flat_err_mc)-nr]
        oracle_mse_mc.append(np.mean(flat_err_mc[oi]**2))
        ui = np.argsort(flat_unc_mc)[:len(flat_unc_mc)-nr]
        unc_mse_mc.append(np.mean(flat_err_mc[ui]**2))
        ri = rng_mc.choice(len(flat_err_mc), len(flat_err_mc)-nr, replace=False)
        rand_mse_mc.append(np.mean(flat_err_mc[ri]**2))

axes[1,2].plot(fracs_mc*100, oracle_mse_mc, 'g-', lw=2, label='Oracle')
axes[1,2].plot(fracs_mc*100, unc_mse_mc, 'b-', lw=2, label='MC Dropout')
axes[1,2].plot(fracs_mc*100, rand_mse_mc, 'r--', lw=2, label='Random')
axes[1,2].set_xlabel('% Pixels Removed'); axes[1,2].set_ylabel('MSE')
axes[1,2].set_title('(g) Sparsification'); axes[1,2].legend(fontsize=7); axes[1,2].grid(True, alpha=0.3)

# Calibration bins
n_bins_mc = 15
bin_edges_mc = np.linspace(flat_unc_mc.min(), flat_unc_mc.max(), n_bins_mc+1)
bmu_mc, bme_mc = [], []
for bi in range(n_bins_mc):
    mk = (flat_unc_mc >= bin_edges_mc[bi]) & (flat_unc_mc < bin_edges_mc[bi+1]+1e-10)
    if mk.sum() > 0:
        bmu_mc.append(flat_unc_mc[mk].mean())
        bme_mc.append(flat_err_mc[mk].mean())
bmu_mc = np.array(bmu_mc); bme_mc = np.array(bme_mc)
mx_mc = max(bmu_mc.max(), bme_mc.max())
axes[1,3].bar(range(len(bmu_mc)), bme_mc/(mx_mc+1e-10), alpha=0.4, color='blue', label='Error')
axes[1,3].plot(range(len(bmu_mc)), bmu_mc/(mx_mc+1e-10), 'o-', color='blue', ms=3, lw=1.5, label='Uncertainty')
axes[1,3].plot(range(len(bmu_mc)), np.linspace(0,1,len(bmu_mc)), 'k--', alpha=0.3, label='Perfect')
axes[1,3].set_xlabel('Bin'); axes[1,3].set_ylabel('Norm. value')
axes[1,3].set_title('(h) Calibration'); axes[1,3].legend(fontsize=6); axes[1,3].grid(True, alpha=0.3)

save_fig(fig, 'figA5_mc_dropout_detail')


# ============================================================
# FIG A6: Cross-Domain (appendix, renamed from fig11)
# ============================================================
print("Fig A6: cross-domain")
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
plt.subplots_adjust(wspace=0.30)

mr_psnrs = cross_domain['mr_psnrs']
ct_psnrs = cross_domain['ct_psnrs']
mr_ssims = cross_domain.get('mr_ssims', [cross_domain['mr_ssim']] * len(mr_psnrs))
ct_ssims = cross_domain.get('ct_ssims', [cross_domain['ct_ssim']] * len(ct_psnrs))
mr_uncs  = cross_domain.get('mr_uncs', [cross_domain['mr_unc']] * len(mr_psnrs))
ct_uncs  = cross_domain.get('ct_uncs', [cross_domain['ct_unc']] * len(ct_psnrs))

bp1 = axes[0].boxplot([mr_psnrs, ct_psnrs], tick_labels=['MR', 'CT'], patch_artist=True, widths=0.6)
bp1['boxes'][0].set_facecolor('#2196F3'); bp1['boxes'][1].set_facecolor('#F44336')
axes[0].set_ylabel('PSNR (dB)'); axes[0].set_title('(a) Recon Quality'); axes[0].grid(True, alpha=0.3)

bp2 = axes[1].boxplot([mr_ssims, ct_ssims], tick_labels=['MR', 'CT'], patch_artist=True, widths=0.6)
bp2['boxes'][0].set_facecolor('#2196F3'); bp2['boxes'][1].set_facecolor('#F44336')
axes[1].set_ylabel('SSIM'); axes[1].set_title('(b) Structural Similarity'); axes[1].grid(True, alpha=0.3)

bp3 = axes[2].boxplot([mr_uncs, ct_uncs], tick_labels=['MR', 'CT'], patch_artist=True, widths=0.6)
bp3['boxes'][0].set_facecolor('#2196F3'); bp3['boxes'][1].set_facecolor('#F44336')
axes[2].set_ylabel('Uncertainty'); axes[2].set_title('(c) Model Uncertainty'); axes[2].grid(True, alpha=0.3)

save_fig(fig, 'figA6_cross_domain')


# ============================================================
# FIG A7: Downstream Segmentation (appendix, renamed from fig12)
# ============================================================
print("Fig A7: downstream segmentation")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
plt.subplots_adjust(wspace=0.28)

Rs_d = sorted([int(k) for k in dice_results.keys()])
gt_dices    = [dice_results[str(r)]['gt_dice'] for r in Rs_d]
recon_dices = [dice_results[str(r)]['recon_dice'] for r in Rs_d]
zf_dices    = [dice_results[str(r)]['zf_dice'] for r in Rs_d]
recon_stds  = [dice_results[str(r)]['recon_std'] for r in Rs_d]
zf_stds     = [dice_results[str(r)]['zf_std'] for r in Rs_d]

ax1.axhline(y=gt_dices[0], color='green', ls=':', lw=2, label=f'GT Dice={gt_dices[0]:.3f}')
ax1.errorbar(Rs_d, recon_dices, yerr=recon_stds, fmt='o-', color='#2196F3', lw=2, capsize=4, ms=6, label='Ours')
ax1.errorbar(Rs_d, zf_dices, yerr=zf_stds, fmt='s--', color='#9E9E9E', lw=1.5, capsize=4, ms=6, label='Zero-filled')
ax1.set_xlabel('Acceleration R'); ax1.set_ylabel('Mean Dice')
ax1.set_title('(a) Seg. Quality'); ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)
ax1.set_xticks(Rs_d)

pres_r = [r/g if g > 0 else 0 for r,g in zip(recon_dices, gt_dices)]
pres_z = [z/g if g > 0 else 0 for z,g in zip(zf_dices, gt_dices)]
x_d = np.arange(len(Rs_d))
w = 0.35
ax2.bar(x_d - w/2, [p*100 for p in pres_r], w, label='Ours', color='#2196F3', edgecolor='black', lw=0.5)
ax2.bar(x_d + w/2, [p*100 for p in pres_z], w, label='Zero-filled', color='#9E9E9E', edgecolor='black', lw=0.5)
ax2.axhline(y=100, color='green', ls=':', lw=2, label='GT Ref')
ax2.set_xlabel('Acceleration R'); ax2.set_ylabel('Dice Preservation (%)')
ax2.set_title('(b) Quality Preservation'); ax2.set_xticks(x_d); ax2.set_xticklabels([f'R={r}' for r in Rs_d])
ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3, axis='y')

save_fig(fig, 'figA7_downstream_segmentation')


# ============================================================
# Summary
# ============================================================
print("\n=== ALL FIGURES REGENERATED ===")
expected = [
    'fig4_reconstruction_comparison',
    'fig13_ensemble_vs_mcdropout',
    'fig14_reliability_diagrams',
    'fig15_uncertainty_seg_correlation',
    'figA1_dataset_overview',
    'figA2_quality_vs_acceleration',
    'figA3_explainability_analysis',
    'figA4_adversarial_robustness',
    'figA5_mc_dropout_detail',
    'figA6_cross_domain',
    'figA7_downstream_segmentation',
]
for name in expected:
    for ext in ['pdf', 'png']:
        path = f'{FIG_DIR}/{name}.{ext}'
        if os.path.exists(path):
            sz = os.path.getsize(path) / 1024
            print(f'  OK  {name}.{ext} ({sz:.0f} KB)')
        else:
            print(f'  MISSING  {name}.{ext}')
