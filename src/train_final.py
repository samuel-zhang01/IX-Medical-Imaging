#!/usr/bin/env python3
"""
Final training script with hardcoded best hyperparameters.
Trains for 100 epochs then reruns full evaluation + figure generation.
"""
import os, sys, json, time, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import get_dataloaders
from models import ReconUNet, SegmentationUNet
from losses import CombinedLoss, compute_psnr, compute_ssim, compute_nmse

DEVICE = torch.device('cuda')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data/processed_data')
CKPT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

# ============================================================
# HARDCODED BEST HYPERPARAMETERS (from Optuna search)
# ============================================================
BEST_CONFIG = {
    'lr': 1.62e-3,
    'batch_size': 16,
    'base_features': 32,
    'dropout_rate': 0.112,
    'ssim_weight': 0.511,
    'weight_decay': 3.37e-6,
    'use_dc': True,
    'num_dc_cascades': 3,
    'center_fraction': 0.08,
}

MAX_EPOCHS = 100  # Longer training for better convergence


def train_model(config, acceleration, max_epochs, tag='final'):
    save_dir = os.path.join(CKPT_DIR, f'{tag}_R{acceleration}')
    best_path = os.path.join(save_dir, f'best_model_R{acceleration}.pth')

    # Remove old checkpoint to force retrain
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n[TRAIN] R={acceleration}x | {max_epochs} epochs | {tag}')
    print(f'  Config: features={config["base_features"]}, dropout={config["dropout_rate"]:.3f}, '
          f'DC={config["num_dc_cascades"]}, lr={config["lr"]:.2e}')

    train_loader, val_loader, _ = get_dataloaders(
        DATA_ROOT, 'mr', acceleration=acceleration,
        batch_size=config['batch_size'], num_workers=4,
        center_fraction=config['center_fraction'])

    model = ReconUNet(
        base_features=config['base_features'],
        dropout_rate=config['dropout_rate'],
        use_dc=config['use_dc'],
        num_dc_cascades=config['num_dc_cascades']
    ).to(DEVICE)

    criterion = CombinedLoss(alpha=config['ssim_weight']).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'],
                            weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    best_psnr = 0
    history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': [], 'val_nmse': []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        t_loss, t_n = 0, 0
        for batch in tqdm(train_loader, desc=f'Ep {epoch}', leave=False):
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
            t_loss += loss.item() * us.size(0)
            t_n += us.size(0)

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
        vm = {'loss': v_loss/v_n, 'psnr': v_psnr/v_n, 'ssim': v_ssim/v_n, 'nmse': v_nmse/v_n}

        history['train_loss'].append(t_loss / t_n)
        history['val_loss'].append(vm['loss'])
        history['val_psnr'].append(vm['psnr'])
        history['val_ssim'].append(vm['ssim'])
        history['val_nmse'].append(vm['nmse'])

        if vm['psnr'] > best_psnr:
            best_psnr = vm['psnr']
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'config': config, 'best_psnr': best_psnr,
                'val_metrics': vm, 'acceleration': acceleration,
            }, best_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f'  Ep {epoch:3d}/{max_epochs} | Loss {t_loss/t_n:.4f} | '
                  f'Val PSNR {vm["psnr"]:.2f} | SSIM {vm["ssim"]:.4f} | Best {best_psnr:.2f}')

    with open(os.path.join(save_dir, f'history_R{acceleration}.json'), 'w') as f:
        json.dump(history, f)
    print(f'  Final best PSNR: {best_psnr:.2f} dB')
    return best_psnr


if __name__ == '__main__':
    t0 = time.time()
    print('='*60)
    print('FINAL TRAINING: Hardcoded best params, 100 epochs')
    print('='*60)

    # Train R=4 and R=8
    for R in [4, 8]:
        train_model(BEST_CONFIG, R, MAX_EPOCHS, tag='final')

    # Train 3-member ensemble
    for i in range(3):
        print(f'\nEnsemble member {i+1}/3')
        torch.manual_seed(42 + i * 1000)
        np.random.seed(42 + i * 1000)
        train_model(BEST_CONFIG, 4, max_epochs=80, tag=f'ensemble_{i}')

    elapsed = (time.time() - t0) / 60
    print(f'\n{"="*60}')
    print(f'Training complete in {elapsed:.1f} minutes')
    print(f'{"="*60}')

    # Now rerun full evaluation
    print('\nRunning full evaluation pipeline...')
    os.chdir(PROJECT_ROOT)
    os.system(f'python3 {os.path.join(PROJECT_ROOT, "src/run_all_experiments.py")}')
