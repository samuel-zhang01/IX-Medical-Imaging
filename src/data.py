"""
Data loading and k-space simulation for MRI reconstruction.
Handles MM-WHS cardiac dataset with simulated single-coil k-space undersampling.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


def create_cartesian_mask(shape: Tuple[int, int], acceleration: int,
                          center_fraction: float = 0.08,
                          seed: Optional[int] = None) -> np.ndarray:
    """Create random Cartesian undersampling mask.

    Args:
        shape: (H, W) image dimensions
        acceleration: Acceleration factor R
        center_fraction: Fraction of center k-space lines to always acquire
        seed: Random seed for reproducibility

    Returns:
        Binary mask of shape (H, W)
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)

    # Always acquire center lines (low-frequency)
    num_center = int(W * center_fraction)
    center_start = (W - num_center) // 2
    mask[:, center_start:center_start + num_center] = 1.0

    # Randomly sample remaining lines
    num_total_lines = max(int(W / acceleration), num_center)
    num_random = num_total_lines - num_center

    if num_random > 0:
        rng = np.random.RandomState(seed)
        available = list(set(range(W)) - set(range(center_start, center_start + num_center)))
        chosen = rng.choice(available, size=min(num_random, len(available)), replace=False)
        mask[:, chosen] = 1.0

    return mask


def image_to_kspace(image: np.ndarray) -> np.ndarray:
    """Convert image to k-space via 2D FFT."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))


def kspace_to_image(kspace: np.ndarray) -> np.ndarray:
    """Convert k-space to image via 2D IFFT."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))


class MRIReconDataset(Dataset):
    """Dataset for MRI reconstruction from undersampled k-space.

    Each sample provides:
        - undersampled: Zero-filled reconstruction (input)
        - target: Fully-sampled ground truth image
        - mask: Undersampling mask
        - kspace: Fully-sampled k-space
        - label: Segmentation label (for downstream evaluation)
    """

    def __init__(self, data_dir: str, acceleration: int = 4,
                 center_fraction: float = 0.08, transform=None,
                 fixed_masks: bool = False):
        self.data_dir = data_dir
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.transform = transform
        self.fixed_masks = fixed_masks

        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith('.npz')
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        image = data['image'].astype(np.float32)
        label = data['label'].astype(np.int64)

        # Normalize image to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Simulate k-space
        kspace = image_to_kspace(image)

        # Create undersampling mask
        seed = idx if self.fixed_masks else None
        mask = create_cartesian_mask(image.shape, self.acceleration,
                                     self.center_fraction, seed=seed)

        # Apply mask to get undersampled k-space
        undersampled_kspace = kspace * mask

        # Zero-filled reconstruction
        undersampled = np.abs(kspace_to_image(undersampled_kspace)).astype(np.float32)

        # Convert to tensors with channel dim
        target_t = torch.from_numpy(image).unsqueeze(0)        # (1, H, W)
        undersampled_t = torch.from_numpy(undersampled).unsqueeze(0)  # (1, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0)            # (1, H, W)
        kspace_real = torch.from_numpy(np.real(kspace).astype(np.float32)).unsqueeze(0)
        kspace_imag = torch.from_numpy(np.imag(kspace).astype(np.float32)).unsqueeze(0)
        kspace_t = torch.cat([kspace_real, kspace_imag], dim=0) # (2, H, W)
        label_t = torch.from_numpy(label)                       # (H, W)

        return {
            'undersampled': undersampled_t,
            'target': target_t,
            'mask': mask_t,
            'kspace': kspace_t,
            'label': label_t,
            'filename': os.path.basename(self.files[idx])
        }


def get_dataloaders(data_root: str, modality: str = 'mr',
                    acceleration: int = 4, batch_size: int = 8,
                    num_workers: int = 4, center_fraction: float = 0.08):
    """Create train/val/test dataloaders.

    Args:
        data_root: Path to processed_data directory
        modality: 'mr' or 'ct'
        acceleration: Undersampling acceleration factor
        batch_size: Batch size
        num_workers: Number of data loading workers
        center_fraction: Fraction of center k-space to retain

    Returns:
        train_loader, val_loader, test_loader
    """
    base = os.path.join(data_root, f'{modality}_256')

    train_ds = MRIReconDataset(
        os.path.join(base, 'train', 'npz'),
        acceleration=acceleration,
        center_fraction=center_fraction,
        fixed_masks=False
    )
    val_ds = MRIReconDataset(
        os.path.join(base, 'val', 'npz'),
        acceleration=acceleration,
        center_fraction=center_fraction,
        fixed_masks=True
    )
    test_ds = MRIReconDataset(
        os.path.join(base, 'test', 'npz'),
        acceleration=acceleration,
        center_fraction=center_fraction,
        fixed_masks=True
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
