#!/usr/bin/env python3
"""
Experiment runner for Trustworthy AI in MRI Reconstruction.
Runs all experiments end-to-end and saves results + figures.

Experiments:
1. Train final reconstruction models (R=4, R=8) with best Optuna params
2. Train 3-model ensemble for Deep Ensemble uncertainty
3. Full test set evaluation at multiple acceleration factors
4. MC Dropout uncertainty quantification
5. Perturbation studies (k-space noise)
6. Adversarial attacks (FGSM, PGD)
7. Cross-domain evaluation (MR→CT)
8. Train segmentation model + downstream Dice analysis
9. Generate all publication figures
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import get_dataloaders, MRIReconDataset, create_cartesian_mask, image_to_kspace, kspace_to_image
from models import ReconUNet, SegmentationUNet
from losses import CombinedLoss, compute_psnr, compute_ssim, compute_nmse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data/processed_data')
CKPT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
FIG_DIR = os.path.join(PROJECT_ROOT, 'latex/figures')

# Publication-quality plot settings
plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif', 'axes.labelsize': 11,
    'axes.titlesize': 12, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05
})

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


def get_best_config():
    """Load best config from Optuna or use defaults."""
    optuna_file = os.path.join(CKPT_DIR, 'optuna_best_params_R4.json')
    if os.path.exists(optuna_file):
        with open(optuna_file) as f:
            data = json.load(f)
        bp = data['best_params']
        return {
            'lr': bp.get('lr', 1e-3),
            'batch_size': bp.get('batch_size', 8),
            'base_features': bp.get('base_features', 64),
            'dropout_rate': bp.get('dropout_rate', 0.1),
            'ssim_weight': bp.get('ssim_weight', 0.84),
            'weight_decay': bp.get('weight_decay', 1e-4),
            'use_dc': True,
            'num_dc_cascades': bp.get('num_dc_cascades', 1),
            'center_fraction': 0.08,
        }
    return {
        'lr': 1e-3, 'batch_size': 8, 'base_features': 64,
        'dropout_rate': 0.1, 'ssim_weight': 0.84, 'weight_decay': 1e-4,
        'use_dc': True, 'num_dc_cascades': 1, 'center_fraction': 0.08,
    }


def train_model_full(config, acceleration, max_epochs=50, tag='final'):
    """Train a model to completion."""
    save_dir = os.path.join(CKPT_DIR, f'{tag}_R{acceleration}')
    best_path = os.path.join(save_dir, f'best_model_R{acceleration}.pth')

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
        print(f'[SKIP] R={acceleration} model exists (PSNR={ckpt["best_psnr"]:.2f}dB)')
        return ckpt

    os.makedirs(save_dir, exist_ok=True)
    print(f'\n[TRAIN] R={acceleration}x model ({max_epochs} epochs)...')

    train_loader, val_loader, _ = get_dataloaders(
        DATA_ROOT, 'mr', acceleration=acceleration,
        batch_size=config['batch_size'], num_workers=4)

    model = ReconUNet(
        base_features=config['base_features'],
        dropout_rate=config['dropout_rate'],
        use_dc=config['use_dc'],
        num_dc_cascades=config.get('num_dc_cascades', 1)
    ).to(DEVICE)

    criterion = CombinedLoss(alpha=config['ssim_weight']).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'],
                            weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    best_psnr = 0
    history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': [], 'val_nmse': []}

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        total_loss, total_psnr, n = 0, 0, 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            us = batch['undersampled'].to(DEVICE)
            tgt = batch['target'].to(DEVICE)
            ks = batch['kspace'].to(DEVICE)
            msk = batch['mask'].to(DEVICE)

            optimizer.zero_grad()
            pred = model(us, ks, msk)
            loss = criterion(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                total_loss += loss.item() * us.size(0)
                total_psnr += compute_psnr(pred, tgt).item() * us.size(0)
                n += us.size(0)

        # Validate
        model.eval()
        v_loss, v_psnr, v_ssim, v_nmse, v_n = 0, 0, 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                us = batch['undersampled'].to(DEVICE)
                tgt = batch['target'].to(DEVICE)
                ks = batch['kspace'].to(DEVICE)
                msk = batch['mask'].to(DEVICE)
                pred = model(us, ks, msk)
                bs = us.size(0)
                v_loss += criterion(pred, tgt).item() * bs
                v_psnr += compute_psnr(pred, tgt).item() * bs
                v_ssim += compute_ssim(pred, tgt).item() * bs
                v_nmse += compute_nmse(pred, tgt).item() * bs
                v_n += bs

        scheduler.step()
        val_metrics = {'loss': v_loss/v_n, 'psnr': v_psnr/v_n, 'ssim': v_ssim/v_n, 'nmse': v_nmse/v_n}

        history['train_loss'].append(total_loss / n)
        history['val_loss'].append(val_metrics['loss'])
        history['val_psnr'].append(val_metrics['psnr'])
        history['val_ssim'].append(val_metrics['ssim'])
        history['val_nmse'].append(val_metrics['nmse'])

        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'config': config, 'best_psnr': best_psnr,
                'val_metrics': val_metrics, 'acceleration': acceleration,
            }, best_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f'  Ep {epoch}/{max_epochs}: Loss={total_loss/n:.4f} '
                  f'Val PSNR={val_metrics["psnr"]:.2f} SSIM={val_metrics["ssim"]:.4f}')

    with open(os.path.join(save_dir, f'history_R{acceleration}.json'), 'w') as f:
        json.dump(history, f)

    print(f'  Best PSNR: {best_psnr:.2f}dB')
    return torch.load(best_path, map_location=DEVICE, weights_only=False)


def load_model(acceleration, tag='final'):
    """Load a trained model."""
    path = os.path.join(CKPT_DIR, f'{tag}_R{acceleration}', f'best_model_R{acceleration}.pth')
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    config = ckpt['config']
    model = ReconUNet(
        base_features=config['base_features'],
        dropout_rate=config['dropout_rate'],
        use_dc=True,
        num_dc_cascades=config.get('num_dc_cascades', 1)
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    return model


# ================================================================
# EXPERIMENT 1: Train final models
# ================================================================
def experiment_1_train():
    print('\n' + '='*70)
    print('EXPERIMENT 1: Training Final Models')
    print('='*70)
    config = get_best_config()
    for R in [4, 8]:
        train_model_full(config, R, max_epochs=50)


# ================================================================
# EXPERIMENT 2: Train ensemble
# ================================================================
def experiment_2_ensemble():
    print('\n' + '='*70)
    print('EXPERIMENT 2: Training Deep Ensemble (3 members)')
    print('='*70)
    config = get_best_config()
    for i in range(3):
        print(f'\nEnsemble member {i+1}/3')
        torch.manual_seed(42 + i * 1000)
        np.random.seed(42 + i * 1000)
        train_model_full(config, acceleration=4, max_epochs=40, tag=f'ensemble_{i}')


# ================================================================
# EXPERIMENT 3: Full test evaluation
# ================================================================
def experiment_3_evaluate():
    print('\n' + '='*70)
    print('EXPERIMENT 3: Full Test Set Evaluation')
    print('='*70)

    results = {}
    for R in [2, 4, 6, 8, 10]:
        _, _, test_loader = get_dataloaders(DATA_ROOT, 'mr', acceleration=R, batch_size=4, num_workers=4)
        model = load_model(4 if R <= 6 else 8)
        model.eval()

        psnrs, ssims, nmses, zf_psnrs, zf_ssims = [], [], [], [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'R={R}x', leave=False):
                us = batch['undersampled'].to(DEVICE)
                tgt = batch['target'].to(DEVICE)
                ks = batch['kspace'].to(DEVICE)
                msk = batch['mask'].to(DEVICE)
                pred = model(us, ks, msk)

                for b in range(us.size(0)):
                    p, t, u = pred[b:b+1], tgt[b:b+1], us[b:b+1]
                    psnrs.append(compute_psnr(p, t).item())
                    ssims.append(compute_ssim(p, t).item())
                    nmses.append(compute_nmse(p, t).item())
                    zf_psnrs.append(compute_psnr(u, t).item())
                    zf_ssims.append(compute_ssim(u, t).item())

        results[R] = {
            'psnr': float(np.mean(psnrs)), 'psnr_std': float(np.std(psnrs)),
            'ssim': float(np.mean(ssims)), 'ssim_std': float(np.std(ssims)),
            'nmse': float(np.mean(nmses)), 'nmse_std': float(np.std(nmses)),
            'zf_psnr': float(np.mean(zf_psnrs)), 'zf_ssim': float(np.mean(zf_ssims)),
        }
        print(f'  R={R}x: PSNR={np.mean(psnrs):.2f}±{np.std(psnrs):.2f}, '
              f'SSIM={np.mean(ssims):.4f}±{np.std(ssims):.4f}')

    with open(os.path.join(CKPT_DIR, 'test_results.json'), 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    # Figure: PSNR/SSIM vs acceleration
    Rs = sorted(results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.errorbar(Rs, [results[R]['psnr'] for R in Rs],
                 yerr=[results[R]['psnr_std'] for R in Rs],
                 marker='o', color='#2196F3', linewidth=2, capsize=5, markersize=8, label='Ours (U-Net+DC)')
    ax1.plot(Rs, [results[R]['zf_psnr'] for R in Rs],
             marker='s', color='#9E9E9E', linewidth=2, linestyle='--', markersize=8, label='Zero-filled')
    ax1.set_xlabel('Acceleration Factor (R)'); ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Reconstruction Quality vs. Acceleration'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.errorbar(Rs, [results[R]['ssim'] for R in Rs],
                 yerr=[results[R]['ssim_std'] for R in Rs],
                 marker='o', color='#4CAF50', linewidth=2, capsize=5, markersize=8, label='Ours (U-Net+DC)')
    ax2.plot(Rs, [results[R]['zf_ssim'] for R in Rs],
             marker='s', color='#9E9E9E', linewidth=2, linestyle='--', markersize=8, label='Zero-filled')
    ax2.set_xlabel('Acceleration Factor (R)'); ax2.set_ylabel('SSIM')
    ax2.set_title('Structural Similarity vs. Acceleration'); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig5_psnr_ssim_vs_acceleration.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig5_psnr_ssim_vs_acceleration.png'))
    plt.close()
    print('  Saved: fig5_psnr_ssim_vs_acceleration.pdf')
    return results


# ================================================================
# EXPERIMENT 4: MC Dropout + Perturbation + Adversarial
# ================================================================
def experiment_4_trustworthy():
    print('\n' + '='*70)
    print('EXPERIMENT 4: Trustworthiness Analysis')
    print('='*70)

    model = load_model(4)
    _, _, test_loader = get_dataloaders(DATA_ROOT, 'mr', acceleration=4, batch_size=1, num_workers=2)

    # Collect test samples
    test_samples = []
    for i, batch in enumerate(test_loader):
        if i >= 30: break
        test_samples.append(batch)

    # 4a: Perturbation Study
    print('\n[4a] K-Space Perturbation Study...')
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    perturb_results = {nl: {'psnrs': [], 'ssims': [], 'uncs': []} for nl in noise_levels}

    for nl in tqdm(noise_levels, desc='Noise levels'):
        for s in test_samples[:20]:
            us = s['undersampled'].to(DEVICE)
            tgt = s['target'].to(DEVICE)
            ks = s['kspace'].to(DEVICE)
            msk = s['mask'].to(DEVICE)

            if nl > 0:
                noise = torch.randn_like(ks) * nl
                ks_n = ks + noise
                ks_c = torch.complex(ks_n[:, 0], ks_n[:, 1])
                us_n = torch.abs(torch.fft.fftshift(
                    torch.fft.ifft2(torch.fft.ifftshift(ks_c * msk.squeeze(1)))
                )).unsqueeze(1)
            else:
                ks_n, us_n = ks, us

            model.eval()
            with torch.no_grad():
                pred = model(us_n, ks_n, msk)
            perturb_results[nl]['psnrs'].append(compute_psnr(pred, tgt).item())
            perturb_results[nl]['ssims'].append(compute_ssim(pred, tgt).item())

            # MC uncertainty
            model.train()
            mc_preds = []
            with torch.no_grad():
                for _ in range(10):
                    mc_preds.append(model(us_n, ks_n, msk))
            perturb_results[nl]['uncs'].append(torch.stack(mc_preds).std(dim=0).mean().item())

    # Save and plot
    perturb_summary = {}
    for nl in noise_levels:
        perturb_summary[str(nl)] = {
            'psnr': float(np.mean(perturb_results[nl]['psnrs'])),
            'ssim': float(np.mean(perturb_results[nl]['ssims'])),
            'unc': float(np.mean(perturb_results[nl]['uncs'])),
            'psnr_std': float(np.std(perturb_results[nl]['psnrs'])),
        }
        print(f'  Noise={nl:.2f}: PSNR={perturb_summary[str(nl)]["psnr"]:.2f}, Unc={perturb_summary[str(nl)]["unc"]:.6f}')

    with open(os.path.join(CKPT_DIR, 'perturbation_results.json'), 'w') as f:
        json.dump(perturb_summary, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    psnrs_p = [perturb_summary[str(nl)]['psnr'] for nl in noise_levels]
    ssims_p = [perturb_summary[str(nl)]['ssim'] for nl in noise_levels]
    uncs_p = [perturb_summary[str(nl)]['unc'] for nl in noise_levels]

    axes[0].plot(noise_levels, psnrs_p, 'o-', color='#2196F3', linewidth=2, markersize=8)
    axes[0].set_xlabel('K-Space Noise Std'); axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('(a) Reconstruction Quality vs. Noise'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(noise_levels, ssims_p, 's-', color='#4CAF50', linewidth=2, markersize=8)
    axes[1].set_xlabel('K-Space Noise Std'); axes[1].set_ylabel('SSIM')
    axes[1].set_title('(b) Structural Similarity vs. Noise'); axes[1].grid(True, alpha=0.3)

    ax2 = axes[2]; ax2b = ax2.twinx()
    ax2.plot(noise_levels, uncs_p, 'D-', color='#F44336', linewidth=2, markersize=8, label='Uncertainty')
    ax2b.plot(noise_levels, psnrs_p, 'o--', color='#2196F3', linewidth=2, label='PSNR')
    ax2.set_xlabel('K-Space Noise Std'); ax2.set_ylabel('Mean Uncertainty', color='#F44336')
    ax2b.set_ylabel('PSNR (dB)', color='#2196F3')
    ax2.set_title('(c) Uncertainty Tracks Degradation')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Perturbation Study: K-Space Noise Robustness', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig8_perturbation_study.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig8_perturbation_study.png'))
    plt.close()
    print('  Saved: fig8_perturbation_study.pdf')

    # 4b: Adversarial Attacks
    print('\n[4b] Adversarial Attack Study...')
    epsilons = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
    adv_results = {eps: {'fgsm_psnrs': [], 'pgd_psnrs': [], 'fgsm_ssims': [],
                         'pgd_ssims': [], 'fgsm_uncs': [], 'pgd_uncs': [],
                         'clean_psnrs': [], 'clean_ssims': []} for eps in epsilons}

    for eps in tqdm(epsilons, desc='Adversarial'):
        for s in test_samples[:15]:
            us = s['undersampled'].to(DEVICE)
            tgt = s['target'].to(DEVICE)
            ks = s['kspace'].to(DEVICE)
            msk = s['mask'].to(DEVICE)

            model.eval()
            with torch.no_grad():
                clean_pred = model(us, ks, msk)
            adv_results[eps]['clean_psnrs'].append(compute_psnr(clean_pred, tgt).item())
            adv_results[eps]['clean_ssims'].append(compute_ssim(clean_pred, tgt).item())

            if eps == 0:
                adv_results[eps]['fgsm_psnrs'].append(adv_results[eps]['clean_psnrs'][-1])
                adv_results[eps]['pgd_psnrs'].append(adv_results[eps]['clean_psnrs'][-1])
                adv_results[eps]['fgsm_ssims'].append(adv_results[eps]['clean_ssims'][-1])
                adv_results[eps]['pgd_ssims'].append(adv_results[eps]['clean_ssims'][-1])
                adv_results[eps]['fgsm_uncs'].append(0.0)
                adv_results[eps]['pgd_uncs'].append(0.0)
                continue

            # FGSM
            us_adv = us.clone().detach().requires_grad_(True)
            pred = model(us_adv, ks, msk)
            loss = F.l1_loss(pred, tgt)
            loss.backward()
            adv_fgsm = (us + eps * us_adv.grad.data.sign()).clamp(0, 1).detach()

            # PGD
            adv_pgd = us.clone().detach()
            alpha = eps / 5 * 2
            for _ in range(7):
                adv_pgd.requires_grad_(True)
                pred = model(adv_pgd, ks, msk)
                F.l1_loss(pred, tgt).backward()
                adv_pgd = (adv_pgd.detach() + alpha * adv_pgd.grad.data.sign())
                delta = torch.clamp(adv_pgd - us, -eps, eps)
                adv_pgd = (us + delta).clamp(0, 1).detach()

            model.eval()
            with torch.no_grad():
                pred_f = model(adv_fgsm, ks, msk)
                pred_p = model(adv_pgd, ks, msk)
            adv_results[eps]['fgsm_psnrs'].append(compute_psnr(pred_f, tgt).item())
            adv_results[eps]['pgd_psnrs'].append(compute_psnr(pred_p, tgt).item())
            adv_results[eps]['fgsm_ssims'].append(compute_ssim(pred_f, tgt).item())
            adv_results[eps]['pgd_ssims'].append(compute_ssim(pred_p, tgt).item())

            # Uncertainty on adversarial
            model.train()
            with torch.no_grad():
                mc_f = [model(adv_fgsm, ks, msk) for _ in range(10)]
                mc_p = [model(adv_pgd, ks, msk) for _ in range(10)]
            adv_results[eps]['fgsm_uncs'].append(torch.stack(mc_f).std(0).mean().item())
            adv_results[eps]['pgd_uncs'].append(torch.stack(mc_p).std(0).mean().item())

    adv_summary = {}
    for eps in epsilons:
        adv_summary[str(eps)] = {
            'fgsm_psnr': float(np.mean(adv_results[eps]['fgsm_psnrs'])),
            'pgd_psnr': float(np.mean(adv_results[eps]['pgd_psnrs'])),
            'fgsm_ssim': float(np.mean(adv_results[eps]['fgsm_ssims'])),
            'pgd_ssim': float(np.mean(adv_results[eps]['pgd_ssims'])),
            'fgsm_unc': float(np.mean(adv_results[eps]['fgsm_uncs'])),
            'pgd_unc': float(np.mean(adv_results[eps]['pgd_uncs'])),
            'clean_psnr': float(np.mean(adv_results[eps]['clean_psnrs'])),
        }

    with open(os.path.join(CKPT_DIR, 'adversarial_results.json'), 'w') as f:
        json.dump(adv_summary, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ep_vals = epsilons

    axes[0].plot(ep_vals, [adv_summary[str(e)]['clean_psnr'] for e in ep_vals], 'k--*', linewidth=2, label='Clean')
    axes[0].plot(ep_vals, [adv_summary[str(e)]['fgsm_psnr'] for e in ep_vals], 'r-o', linewidth=2, label='FGSM')
    axes[0].plot(ep_vals, [adv_summary[str(e)]['pgd_psnr'] for e in ep_vals], 'b-s', linewidth=2, label='PGD')
    axes[0].set_xlabel('Perturbation ε'); axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('(a) PSNR Under Attack'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep_vals, [adv_summary[str(e)]['fgsm_ssim'] for e in ep_vals], 'r-o', linewidth=2, label='FGSM')
    axes[1].plot(ep_vals, [adv_summary[str(e)]['pgd_ssim'] for e in ep_vals], 'b-s', linewidth=2, label='PGD')
    axes[1].set_xlabel('Perturbation ε'); axes[1].set_ylabel('SSIM')
    axes[1].set_title('(b) SSIM Under Attack'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(ep_vals[1:], [adv_summary[str(e)]['fgsm_unc'] for e in ep_vals[1:]], 'r-o', linewidth=2, label='FGSM Unc.')
    axes[2].plot(ep_vals[1:], [adv_summary[str(e)]['pgd_unc'] for e in ep_vals[1:]], 'b-s', linewidth=2, label='PGD Unc.')
    axes[2].set_xlabel('Perturbation ε'); axes[2].set_ylabel('Mean Uncertainty')
    axes[2].set_title('(c) Uncertainty Detects Attacks'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle('Adversarial Robustness Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig9_adversarial_robustness.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig9_adversarial_robustness.png'))
    plt.close()
    print('  Saved: fig9_adversarial_robustness.pdf')

    # 4c: Cross-Domain
    print('\n[4c] Cross-Domain Evaluation (MR→CT)...')
    _, _, ct_loader = get_dataloaders(DATA_ROOT, 'ct', acceleration=4, batch_size=1, num_workers=2)
    _, _, mr_loader = get_dataloaders(DATA_ROOT, 'mr', acceleration=4, batch_size=1, num_workers=2)
    model_mr = load_model(4)

    mr_psnrs, mr_uncs, ct_psnrs, ct_uncs = [], [], [], []
    mr_ssims, ct_ssims = [], []

    for loader, psnrs, ssims, uncs, name in [
        (mr_loader, mr_psnrs, mr_ssims, mr_uncs, 'MR'),
        (ct_loader, ct_psnrs, ct_ssims, ct_uncs, 'CT')
    ]:
        for i, batch in enumerate(loader):
            if i >= 50: break
            us = batch['undersampled'].to(DEVICE)
            tgt = batch['target'].to(DEVICE)
            ks = batch['kspace'].to(DEVICE)
            msk = batch['mask'].to(DEVICE)

            model_mr.eval()
            with torch.no_grad():
                pred = model_mr(us, ks, msk)
            psnrs.append(compute_psnr(pred, tgt).item())
            ssims.append(compute_ssim(pred, tgt).item())

            model_mr.train()
            mc = []
            with torch.no_grad():
                for _ in range(10):
                    mc.append(model_mr(us, ks, msk))
            uncs.append(torch.stack(mc).std(0).mean().item())

        print(f'  {name}: PSNR={np.mean(psnrs):.2f}±{np.std(psnrs):.2f}, Unc={np.mean(uncs):.6f}')

    cross_domain = {
        'mr_psnr': float(np.mean(mr_psnrs)), 'mr_ssim': float(np.mean(mr_ssims)), 'mr_unc': float(np.mean(mr_uncs)),
        'ct_psnr': float(np.mean(ct_psnrs)), 'ct_ssim': float(np.mean(ct_ssims)), 'ct_unc': float(np.mean(ct_uncs)),
        'mr_psnrs': [float(x) for x in mr_psnrs], 'ct_psnrs': [float(x) for x in ct_psnrs],
        'mr_uncs': [float(x) for x in mr_uncs], 'ct_uncs': [float(x) for x in ct_uncs],
        'mr_ssims': [float(x) for x in mr_ssims], 'ct_ssims': [float(x) for x in ct_ssims],
    }
    with open(os.path.join(CKPT_DIR, 'cross_domain_results.json'), 'w') as f:
        json.dump(cross_domain, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (data_mr, data_ct, ylabel, title) in enumerate([
        (mr_psnrs, ct_psnrs, 'PSNR (dB)', '(a) Reconstruction Quality'),
        (mr_ssims, ct_ssims, 'SSIM', '(b) Structural Similarity'),
        (mr_uncs, ct_uncs, 'Mean Uncertainty', '(c) Model Uncertainty'),
    ]):
        bp = axes[i].boxplot([data_mr, data_ct], labels=['MR (In-domain)', 'CT (Out-of-domain)'],
                             patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('#2196F3')
        bp['boxes'][1].set_facecolor('#F44336')
        axes[i].set_ylabel(ylabel); axes[i].set_title(title); axes[i].grid(True, alpha=0.3)

    plt.suptitle('Cross-Domain Robustness: MR-Trained Model on CT Data', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig11_cross_domain.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig11_cross_domain.png'))
    plt.close()
    print('  Saved: fig11_cross_domain.pdf')


# ================================================================
# EXPERIMENT 5: Downstream Segmentation
# ================================================================
def experiment_5_segmentation():
    print('\n' + '='*70)
    print('EXPERIMENT 5: Downstream Segmentation Analysis')
    print('='*70)

    seg_path = os.path.join(CKPT_DIR, 'seg_model.pth')
    seg_model = SegmentationUNet(in_channels=1, num_classes=8, base_features=32).to(DEVICE)

    if os.path.exists(seg_path):
        seg_model.load_state_dict(torch.load(seg_path, map_location=DEVICE, weights_only=True))
        print('[SKIP] Loaded existing segmentation model')
    else:
        print('Training segmentation model...')
        seg_optimizer = optim.Adam(seg_model.parameters(), lr=1e-3)
        seg_criterion = nn.CrossEntropyLoss()

        train_loader, _, _ = get_dataloaders(DATA_ROOT, 'mr', acceleration=2, batch_size=16, num_workers=4)
        for epoch in range(1, 31):
            seg_model.train()
            for batch in tqdm(train_loader, leave=False, desc=f'Seg Ep {epoch}'):
                images = batch['target'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                seg_optimizer.zero_grad()
                loss = seg_criterion(seg_model(images), labels)
                loss.backward()
                seg_optimizer.step()
            if epoch % 10 == 0:
                print(f'  Epoch {epoch} done')

        torch.save(seg_model.state_dict(), seg_path)
        print('  Segmentation model saved')

    # Dice evaluation
    def dice_score(pred_mask, gt_mask, num_classes=8):
        dices = []
        for c in range(1, num_classes):
            p = (pred_mask == c).float()
            g = (gt_mask == c).float()
            if g.sum() == 0 and p.sum() == 0: continue
            dice = (2 * (p * g).sum()) / (p.sum() + g.sum() + 1e-8)
            dices.append(dice.item())
        return np.mean(dices) if dices else 0.0

    seg_model.eval()
    recon_model = load_model(4)

    dice_results = {}
    for R in [2, 4, 6, 8, 10]:
        _, _, test_loader = get_dataloaders(DATA_ROOT, 'mr', acceleration=R, batch_size=1, num_workers=2)
        gt_dices, recon_dices, zf_dices = [], [], []

        for i, batch in enumerate(test_loader):
            if i >= 50: break
            tgt = batch['target'].to(DEVICE)
            us = batch['undersampled'].to(DEVICE)
            ks = batch['kspace'].to(DEVICE)
            msk = batch['mask'].to(DEVICE)
            label = batch['label'].to(DEVICE)

            recon_model.eval()
            with torch.no_grad():
                recon = recon_model(us, ks, msk)
                seg_gt = seg_model(tgt).argmax(1)
                seg_recon = seg_model(recon).argmax(1)
                seg_zf = seg_model(us).argmax(1)

            gt_dices.append(dice_score(seg_gt, label))
            recon_dices.append(dice_score(seg_recon, label))
            zf_dices.append(dice_score(seg_zf, label))

        dice_results[R] = {
            'gt_dice': float(np.mean(gt_dices)), 'recon_dice': float(np.mean(recon_dices)),
            'zf_dice': float(np.mean(zf_dices)),
            'gt_std': float(np.std(gt_dices)), 'recon_std': float(np.std(recon_dices)),
            'zf_std': float(np.std(zf_dices)),
        }
        print(f'  R={R}x: GT={np.mean(gt_dices):.4f}, Recon={np.mean(recon_dices):.4f}, ZF={np.mean(zf_dices):.4f}')

    with open(os.path.join(CKPT_DIR, 'dice_results.json'), 'w') as f:
        json.dump({str(k): v for k, v in dice_results.items()}, f, indent=2)

    # Plot
    Rs = sorted(dice_results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.axhline(y=dice_results[Rs[0]]['gt_dice'], color='green', linestyle=':', linewidth=2,
                label=f'GT Dice = {dice_results[Rs[0]]["gt_dice"]:.3f}')
    ax1.errorbar(Rs, [dice_results[R]['recon_dice'] for R in Rs],
                 yerr=[dice_results[R]['recon_std'] for R in Rs],
                 marker='o', color='#2196F3', linewidth=2, capsize=5, label='Our Reconstruction')
    ax1.errorbar(Rs, [dice_results[R]['zf_dice'] for R in Rs],
                 yerr=[dice_results[R]['zf_std'] for R in Rs],
                 marker='s', color='#9E9E9E', linewidth=2, capsize=5, linestyle='--', label='Zero-filled')
    ax1.set_xlabel('Acceleration Factor (R)'); ax1.set_ylabel('Mean Dice Score')
    ax1.set_title('(a) Downstream Segmentation Quality'); ax1.legend(); ax1.grid(True, alpha=0.3)

    # Preservation
    gt_d = dice_results[Rs[0]]['gt_dice']
    x = np.arange(len(Rs)); w = 0.35
    ax2.bar(x - w/2, [dice_results[R]['recon_dice']/gt_d*100 for R in Rs], w,
            label='Reconstruction', color='#2196F3', edgecolor='black', linewidth=0.5)
    ax2.bar(x + w/2, [dice_results[R]['zf_dice']/gt_d*100 for R in Rs], w,
            label='Zero-filled', color='#9E9E9E', edgecolor='black', linewidth=0.5)
    ax2.axhline(y=100, color='green', linestyle=':', linewidth=2)
    ax2.set_xticks(x); ax2.set_xticklabels([f'R={r}' for r in Rs])
    ax2.set_ylabel('Dice Preservation (%)'); ax2.set_title('(b) Diagnostic Quality Preservation')
    ax2.legend(); ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Downstream: Cardiac Segmentation on Reconstructed Images', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig12_downstream_segmentation.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig12_downstream_segmentation.png'))
    plt.close()
    print('  Saved: fig12_downstream_segmentation.pdf')


# ================================================================
# EXPERIMENT 6: Visual figures (reconstruction comparison, XAI)
# ================================================================
def experiment_6_visual_figures():
    print('\n' + '='*70)
    print('EXPERIMENT 6: Generating Visual Figures')
    print('='*70)

    model_R4 = load_model(4)
    model_R8 = load_model(8)
    _, _, test4 = get_dataloaders(DATA_ROOT, 'mr', acceleration=4, batch_size=1, num_workers=2)
    _, _, test8 = get_dataloaders(DATA_ROOT, 'mr', acceleration=8, batch_size=1, num_workers=2)

    # Get a good sample
    iter4, iter8 = iter(test4), iter(test8)
    for _ in range(8):
        b4 = next(iter4); b8 = next(iter8)

    # Fig 4: Reconstruction comparison
    model_R4.eval(); model_R8.eval()
    with torch.no_grad():
        pred4 = model_R4(b4['undersampled'].to(DEVICE), b4['kspace'].to(DEVICE), b4['mask'].to(DEVICE))
        pred8 = model_R8(b8['undersampled'].to(DEVICE), b8['kspace'].to(DEVICE), b8['mask'].to(DEVICE))

    gt = b4['target'][0,0].numpy()
    zf4 = b4['undersampled'][0,0].numpy()
    zf8 = b8['undersampled'][0,0].numpy()
    r4 = pred4[0,0].cpu().numpy()
    r8 = pred8[0,0].cpu().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    vmax_err = 0.15

    images_row1 = [gt, zf4, r4, zf8, r8]
    titles_row1 = ['Ground Truth',
                   f'Zero-filled R=4x\nPSNR={10*np.log10(1/(np.mean((zf4-gt)**2)+1e-10)):.1f}dB',
                   f'Ours R=4x\nPSNR={10*np.log10(1/(np.mean((r4-gt)**2)+1e-10)):.1f}dB',
                   f'Zero-filled R=8x\nPSNR={10*np.log10(1/(np.mean((zf8-gt)**2)+1e-10)):.1f}dB',
                   f'Ours R=8x\nPSNR={10*np.log10(1/(np.mean((r8-gt)**2)+1e-10)):.1f}dB']

    for i in range(5):
        axes[0, i].imshow(images_row1[i], cmap='gray'); axes[0, i].set_title(titles_row1[i], fontsize=9)
        axes[0, i].axis('off')

    axes[1, 0].axis('off')
    for i, (img, title) in enumerate([(zf4, 'Error ZF R=4'), (r4, 'Error Ours R=4'),
                                       (zf8, 'Error ZF R=8'), (r8, 'Error Ours R=8')]):
        axes[1, i+1].imshow(np.abs(img - gt), cmap='hot', vmin=0, vmax=vmax_err)
        axes[1, i+1].set_title(title, fontsize=9); axes[1, i+1].axis('off')

    plt.suptitle('MRI Reconstruction: U-Net + Data Consistency', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_reconstruction_comparison.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig4_reconstruction_comparison.png'))
    plt.close()
    print('  Saved: fig4_reconstruction_comparison.pdf')

    # Fig 1: Dataset overview
    mr_files = sorted(os.listdir(os.path.join(DATA_ROOT, 'mr_256/train/npz')))[:5]
    samples = [np.load(os.path.join(DATA_ROOT, 'mr_256/train/npz', f)) for f in mr_files]

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for i, s in enumerate(samples):
        img = s['image']
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[0, i].imshow(img, cmap='gray'); axes[0, i].set_title(f'MR Slice {i+1}'); axes[0, i].axis('off')
        axes[1, i].imshow(s['label'], cmap='tab10', vmin=0, vmax=7); axes[1, i].set_title('Segmentation'); axes[1, i].axis('off')

    plt.suptitle('MM-WHS Cardiac MR Dataset', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_dataset_overview.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig1_dataset_overview.png'))
    plt.close()
    print('  Saved: fig1_dataset_overview.pdf')

    # Fig 2: K-space undersampling
    sample_img = samples[2]['image'].astype(np.float32)
    sample_img = (sample_img - sample_img.min()) / (sample_img.max() - sample_img.min() + 1e-8)
    kspace = image_to_kspace(sample_img)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0,0].imshow(sample_img, cmap='gray'); axes[0,0].set_title('Ground Truth'); axes[0,0].axis('off')
    axes[1,0].imshow(np.log1p(np.abs(kspace)), cmap='hot'); axes[1,0].set_title('Full k-space'); axes[1,0].axis('off')

    for i, R in enumerate([2, 4, 8]):
        mask = create_cartesian_mask(sample_img.shape, R, seed=42)
        uk = kspace * mask
        zf = np.abs(kspace_to_image(uk))
        psnr_v = 10 * np.log10(1 / (np.mean((zf - sample_img)**2) + 1e-10))
        axes[0,i+1].imshow(np.log1p(np.abs(uk)), cmap='hot')
        axes[0,i+1].set_title(f'k-space R={R}x ({mask.mean()*100:.0f}%)'); axes[0,i+1].axis('off')
        axes[1,i+1].imshow(zf, cmap='gray')
        axes[1,i+1].set_title(f'Zero-filled R={R}x (PSNR={psnr_v:.1f}dB)'); axes[1,i+1].axis('off')

    plt.suptitle('K-Space Undersampling Simulation', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig2_kspace_undersampling.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig2_kspace_undersampling.png'))
    plt.close()
    print('  Saved: fig2_kspace_undersampling.pdf')

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for R, color, label in [(4, '#2196F3', 'R=4x'), (8, '#F44336', 'R=8x')]:
        hp = os.path.join(CKPT_DIR, f'final_R{R}', f'history_R{R}.json')
        if not os.path.exists(hp): continue
        with open(hp) as f: h = json.load(f)
        eps = range(1, len(h['train_loss'])+1)
        axes[0].plot(eps, h['train_loss'], color=color, alpha=0.5, label=f'{label} Train')
        axes[0].plot(eps, h['val_loss'], color=color, linestyle='--', label=f'{label} Val')
        axes[1].plot(eps, h['val_psnr'], color=color, label=label)
        axes[2].plot(eps, h['val_ssim'], color=color, label=label)

    for ax, title, ylabel in zip(axes, ['Loss', 'Val PSNR', 'Val SSIM'], ['Loss', 'PSNR (dB)', 'SSIM']):
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle('Training Convergence', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig3_training_curves.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig3_training_curves.png'))
    plt.close()
    print('  Saved: fig3_training_curves.pdf')

    # MC Dropout uncertainty visualization
    model_R4.train()  # Enable dropout
    mc_preds = []
    with torch.no_grad():
        for _ in range(30):
            mc_preds.append(model_R4(b4['undersampled'].to(DEVICE), b4['kspace'].to(DEVICE), b4['mask'].to(DEVICE)).cpu())

    mc_stack = torch.stack(mc_preds)
    mc_mean = mc_stack.mean(0)[0,0].numpy()
    mc_std = mc_stack.std(0)[0,0].numpy()
    error = np.abs(mc_mean - gt)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0,0].imshow(gt, cmap='gray'); axes[0,0].set_title('(a) Ground Truth')
    axes[0,1].imshow(mc_mean, cmap='gray'); axes[0,1].set_title(f'(b) MC Mean (PSNR={10*np.log10(1/(np.mean((mc_mean-gt)**2)+1e-10)):.1f}dB)')
    im = axes[0,2].imshow(mc_std, cmap='magma'); plt.colorbar(im, ax=axes[0,2], fraction=0.046)
    axes[0,2].set_title('(c) Uncertainty (Std)')
    im = axes[0,3].imshow(error, cmap='hot'); plt.colorbar(im, ax=axes[0,3], fraction=0.046)
    axes[0,3].set_title('(d) Absolute Error')

    axes[1,0].imshow(gt, cmap='gray', alpha=0.5); axes[1,0].imshow(mc_std, cmap='magma', alpha=0.5)
    axes[1,0].set_title('(e) Uncertainty Overlay')

    idx = np.random.choice(mc_std.size, 3000, replace=False)
    axes[1,1].scatter(mc_std.flatten()[idx], error.flatten()[idx], alpha=0.2, s=2, c='#2196F3')
    r_corr = stats.pearsonr(mc_std.flatten(), error.flatten())[0]
    axes[1,1].set_xlabel('Uncertainty'); axes[1,1].set_ylabel('Error')
    axes[1,1].set_title(f'(f) Corr: r={r_corr:.3f}')

    # Sparsification
    flat_err, flat_unc = error.flatten(), mc_std.flatten()
    fracs = np.linspace(0, 0.95, 40)
    unc_mses, oracle_mses, rand_mses = [], [], []
    for frac in fracs:
        n_rm = int(frac * len(flat_err))
        if n_rm == 0:
            unc_mses.append(np.mean(flat_err**2)); oracle_mses.append(np.mean(flat_err**2))
            rand_mses.append(np.mean(flat_err**2)); continue
        unc_idx = np.argsort(flat_unc)[:len(flat_unc)-n_rm]
        oracle_idx = np.argsort(flat_err)[:len(flat_err)-n_rm]
        rng = np.random.RandomState(42)
        rand_idx = rng.choice(len(flat_err), len(flat_err)-n_rm, replace=False)
        unc_mses.append(np.mean(flat_err[unc_idx]**2))
        oracle_mses.append(np.mean(flat_err[oracle_idx]**2))
        rand_mses.append(np.mean(flat_err[rand_idx]**2))

    axes[1,2].plot(fracs*100, oracle_mses, 'g-', lw=2, label='Oracle')
    axes[1,2].plot(fracs*100, unc_mses, 'b-', lw=2, label='MC Dropout')
    axes[1,2].plot(fracs*100, rand_mses, 'r--', lw=2, label='Random')
    axes[1,2].set_xlabel('% Removed'); axes[1,2].set_ylabel('MSE Remaining')
    axes[1,2].set_title('(g) Sparsification'); axes[1,2].legend(fontsize=8); axes[1,2].grid(True, alpha=0.3)

    # Calibration
    n_bins = 12
    bin_edges = np.linspace(flat_unc.min(), flat_unc.max(), n_bins+1)
    bmu, bme = [], []
    for j in range(n_bins):
        m = (flat_unc >= bin_edges[j]) & (flat_unc < bin_edges[j+1])
        if m.sum() > 0: bmu.append(flat_unc[m].mean()); bme.append(flat_err[m].mean())
    axes[1,3].plot(bmu, bme, 'o-', color='#2196F3', markersize=5)
    axes[1,3].plot([0, max(bmu)], [0, max(bmu)], 'k--', alpha=0.5, label='Perfect')
    axes[1,3].set_xlabel('Predicted Uncertainty'); axes[1,3].set_ylabel('Actual Error')
    axes[1,3].set_title('(h) Calibration'); axes[1,3].legend(fontsize=8); axes[1,3].grid(True, alpha=0.3)

    for ax in axes.flat:
        if ax.images: ax.axis('off')

    plt.suptitle('MC Dropout Uncertainty Quantification (T=30)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig7_mc_dropout_uncertainty.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig7_mc_dropout_uncertainty.png'))
    plt.close()
    print('  Saved: fig7_mc_dropout_uncertainty.pdf')


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    start = time.time()
    print('='*70)
    print('TRUSTWORTHY AI IN MRI RECONSTRUCTION - Full Experiment Suite')
    print(f'Device: {DEVICE}')
    print('='*70)

    experiment_1_train()
    experiment_2_ensemble()
    experiment_3_evaluate()
    experiment_4_trustworthy()
    experiment_5_segmentation()
    experiment_6_visual_figures()

    elapsed = time.time() - start
    print(f'\n{"="*70}')
    print(f'ALL EXPERIMENTS COMPLETE in {elapsed/60:.1f} minutes')
    print(f'Figures saved to: {FIG_DIR}/')
    print(f'Checkpoints saved to: {CKPT_DIR}/')
    print(f'{"="*70}')
