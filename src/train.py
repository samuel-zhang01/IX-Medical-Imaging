"""
Training script for MRI reconstruction with Optuna hyperparameter optimization.
Trains U-Net with MC Dropout and Data Consistency for multiple acceleration factors.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import optuna
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import get_dataloaders
from models import ReconUNet
from losses import CombinedLoss, compute_psnr, compute_ssim, compute_nmse


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    n = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch}', leave=False)
    for batch in pbar:
        undersampled = batch['undersampled'].to(device)
        target = batch['target'].to(device)
        kspace = batch['kspace'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()
        pred = model(undersampled, kspace, mask)
        loss = criterion(pred, target)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            psnr = compute_psnr(pred, target)
            running_loss += loss.item() * undersampled.size(0)
            running_psnr += psnr.item() * undersampled.size(0)
            n += undersampled.size(0)

        pbar.set_postfix(loss=f'{loss.item():.4f}', psnr=f'{psnr.item():.2f}')

    return running_loss / n, running_psnr / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    running_nmse = 0.0
    n = 0

    for batch in loader:
        undersampled = batch['undersampled'].to(device)
        target = batch['target'].to(device)
        kspace = batch['kspace'].to(device)
        mask = batch['mask'].to(device)

        pred = model(undersampled, kspace, mask)
        loss = criterion(pred, target)

        psnr = compute_psnr(pred, target)
        ssim = compute_ssim(pred, target)
        nmse = compute_nmse(pred, target)

        bs = undersampled.size(0)
        running_loss += loss.item() * bs
        running_psnr += psnr.item() * bs
        running_ssim += ssim.item() * bs
        running_nmse += nmse.item() * bs
        n += bs

    return {
        'loss': running_loss / n,
        'psnr': running_psnr / n,
        'ssim': running_ssim / n,
        'nmse': running_nmse / n
    }


def train_model(config: dict, data_root: str, save_dir: str,
                acceleration: int = 4, max_epochs: int = 50,
                trial: optuna.Trial = None):
    """Train a single model with given config.

    Args:
        config: Hyperparameter dictionary
        data_root: Path to processed_data/
        save_dir: Directory to save checkpoints
        acceleration: K-space acceleration factor
        max_epochs: Maximum training epochs
        trial: Optuna trial for pruning

    Returns:
        Best validation PSNR
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        data_root, modality='mr',
        acceleration=acceleration,
        batch_size=config['batch_size'],
        num_workers=4,
        center_fraction=config.get('center_fraction', 0.08)
    )

    # Model
    model = ReconUNet(
        base_features=config['base_features'],
        dropout_rate=config['dropout_rate'],
        use_dc=config.get('use_dc', True),
        num_dc_cascades=config.get('num_dc_cascades', 1)
    ).to(device)

    # Loss, optimizer, scheduler
    criterion = CombinedLoss(alpha=config.get('ssim_weight', 0.84))
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'],
                            weight_decay=config.get('weight_decay', 1e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    # Training loop
    best_psnr = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': [], 'val_nmse': []}

    for epoch in range(1, max_epochs + 1):
        train_loss, train_psnr = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_psnr'].append(val_metrics['psnr'])
        history['val_ssim'].append(val_metrics['ssim'])
        history['val_nmse'].append(val_metrics['nmse'])

        print(f"Epoch {epoch}/{max_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val PSNR: {val_metrics['psnr']:.2f} | "
              f"Val SSIM: {val_metrics['ssim']:.4f} | "
              f"Val NMSE: {val_metrics['nmse']:.6f}")

        # Save best model
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'best_psnr': best_psnr,
                'val_metrics': val_metrics,
                'acceleration': acceleration,
            }, os.path.join(save_dir, f'best_model_R{acceleration}.pth'))

        # Save history
        with open(os.path.join(save_dir, f'history_R{acceleration}.json'), 'w') as f:
            json.dump(history, f)

        # Optuna pruning
        if trial is not None:
            trial.report(val_metrics['psnr'], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_psnr


def optuna_objective(trial, data_root, save_dir, acceleration, max_epochs=30):
    """Optuna objective function for hyperparameter optimization."""
    config = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16]),
        'base_features': trial.suggest_categorical('base_features', [32, 48, 64]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.05, 0.3),
        'ssim_weight': trial.suggest_float('ssim_weight', 0.5, 0.95),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'use_dc': True,
        'num_dc_cascades': trial.suggest_int('num_dc_cascades', 1, 3),
        'center_fraction': 0.08,
    }

    trial_dir = os.path.join(save_dir, f'trial_{trial.number}')
    return train_model(config, data_root, trial_dir, acceleration, max_epochs, trial)


def run_optuna_search(data_root, save_dir, acceleration=4, n_trials=15, max_epochs=25):
    """Run Optuna hyperparameter search."""
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
        study_name=f'mri_recon_R{acceleration}'
    )

    study.optimize(
        lambda trial: optuna_objective(trial, data_root, save_dir, acceleration, max_epochs),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Save study results
    best_params = study.best_params
    print(f"\nBest trial PSNR: {study.best_value:.2f}")
    print(f"Best params: {best_params}")

    with open(os.path.join(save_dir, f'optuna_best_params_R{acceleration}.json'), 'w') as f:
        json.dump({'best_params': best_params, 'best_value': study.best_value}, f, indent=2)

    return best_params, study.best_value


def train_final_model(data_root, save_dir, config, acceleration, max_epochs=50):
    """Train the final model with best hyperparameters."""
    print(f"\n{'='*60}")
    print(f"Training final model with R={acceleration}")
    print(f"Config: {config}")
    print(f"{'='*60}\n")

    final_dir = os.path.join(save_dir, f'final_R{acceleration}')
    best_psnr = train_model(config, data_root, final_dir, acceleration, max_epochs)
    print(f"\nFinal model R={acceleration}: Best PSNR = {best_psnr:.2f}")
    return best_psnr


def train_ensemble(data_root, save_dir, config, acceleration=4,
                   num_models=5, max_epochs=50):
    """Train an ensemble of models for Deep Ensemble uncertainty."""
    print(f"\n{'='*60}")
    print(f"Training ensemble of {num_models} models with R={acceleration}")
    print(f"{'='*60}\n")

    for i in range(num_models):
        print(f"\nEnsemble member {i+1}/{num_models}")
        # Different random seed for each model
        torch.manual_seed(42 + i * 1000)
        np.random.seed(42 + i * 1000)

        ensemble_dir = os.path.join(save_dir, f'ensemble_R{acceleration}', f'member_{i}')
        train_model(config, data_root, ensemble_dir, acceleration, max_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/processed_data')
    parser.add_argument('--save_dir', default='checkpoints')
    parser.add_argument('--acceleration', type=int, default=4)
    parser.add_argument('--mode', choices=['optuna', 'train', 'ensemble', 'all'], default='all')
    parser.add_argument('--n_trials', type=int, default=12)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--ensemble_size', type=int, default=5)
    args = parser.parse_args()

    if args.mode in ('optuna', 'all'):
        # Phase 1: Optuna search
        best_params, best_val = run_optuna_search(
            args.data_root, args.save_dir, args.acceleration,
            n_trials=args.n_trials, max_epochs=min(25, args.max_epochs)
        )
        best_config = {
            'lr': best_params['lr'],
            'batch_size': best_params['batch_size'],
            'base_features': best_params['base_features'],
            'dropout_rate': best_params['dropout_rate'],
            'ssim_weight': best_params['ssim_weight'],
            'weight_decay': best_params['weight_decay'],
            'use_dc': True,
            'num_dc_cascades': best_params['num_dc_cascades'],
            'center_fraction': 0.08,
        }
    else:
        # Default good config
        best_config = {
            'lr': 3e-4,
            'batch_size': 8,
            'base_features': 64,
            'dropout_rate': 0.1,
            'ssim_weight': 0.84,
            'weight_decay': 1e-4,
            'use_dc': True,
            'num_dc_cascades': 1,
            'center_fraction': 0.08,
        }

    if args.mode in ('train', 'all'):
        # Phase 2: Train final models at multiple acceleration factors
        for R in [4, 8]:
            train_final_model(
                args.data_root, args.save_dir, best_config,
                acceleration=R, max_epochs=args.max_epochs
            )

    if args.mode in ('ensemble', 'all'):
        # Phase 3: Train ensemble for uncertainty
        train_ensemble(
            args.data_root, args.save_dir, best_config,
            acceleration=4, num_models=args.ensemble_size,
            max_epochs=args.max_epochs
        )
