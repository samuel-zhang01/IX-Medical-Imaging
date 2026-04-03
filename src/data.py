"""
Data loading and k-space simulation for MRI reconstruction.
Handles MM-WHS cardiac dataset with simulated single-coil k-space undersampling.

K-Space Simulation Pipeline

Since the MM-WHS dataset provides magnitude images (not raw k-space), I
simulate the MRI acquisition process.  The pipeline is:
    1. Start with a fully-sampled magnitude image (ground truth)
    2. Compute the 2D DFT to get the "fully-sampled" k-space
    3. Apply a Cartesian undersampling mask to simulate accelerated acquisition
    4. Compute the inverse DFT of the undersampled k-space (zero-filled recon)
    5. Train the network to recover the ground truth from the zero-filled input

This simulation approach is standard in MRI reconstruction research when raw
multi-coil k-space data is unavailable (e.g., Schlemper et al. 2018, Hyun
et al. 2018).  The main limitation is that we assume single-coil data and
ignore noise, coil sensitivities, and other acquisition-specific artefacts.
However, it lets us use any segmented cardiac dataset (like MM-WHS) without
needing access to the scanner.

Why Cartesian Undersampling?

I chose 1D Cartesian (column-wise) undersampling because:
  - It is the most common acquisition scheme in clinical MRI, making our
    results directly relevant to practice
  - The resulting aliasing artefacts are coherent (structured) rather than
    incoherent, which makes them harder to remove but more realistic
  - It matches the undersampling pattern used in most reconstruction
    benchmarks (fastMRI, Calgary-Campinas), so our results are comparable
  - Other schemes like radial or spiral undersampling produce incoherent
    artefacts that are arguably easier for deep learning to remove

The center_fraction parameter preserves the central k-space lines (the ACS
region -- Auto-Calibration Signal).  These low-frequency components encode
the bulk of image contrast and overall structure.  Preserving them is
critical because:
  1. Without the centre, the zero-filled reconstruction would have almost no
     useful signal, making the learning problem unnecessarily hard
  2. In real accelerated MRI, the ACS lines are always acquired for parallel
     imaging calibration (GRAPPA/SENSE)
  3. The centre fraction (default 8%) at acceleration R=4 gives an effective
     acceleration of ~3.4x, which is clinically realistic
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

    The mask is column-wise (each column is either fully sampled or not),
    which simulates phase-encode undersampling in 2D Cartesian MRI.
    I first reserve the centre lines, then randomly sample from the
    remaining columns to reach the target acceleration factor.

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

    # Always acquire center lines (low-frequency / ACS region).
    # These lines carry the bulk of image contrast and are essential for
    # a reasonable zero-filled starting point.
    num_center = int(W * center_fraction)
    center_start = (W - num_center) // 2
    mask[:, center_start:center_start + num_center] = 1.0

    # Randomly sample remaining lines to reach the target total line count.
    # The total number of lines is W/R, minus the center lines we already have.
    num_total_lines = max(int(W / acceleration), num_center)
    num_random = num_total_lines - num_center

    if num_random > 0:
        rng = np.random.RandomState(seed)
        available = list(set(range(W)) - set(range(center_start, center_start + num_center)))
        chosen = rng.choice(available, size=min(num_random, len(available)), replace=False)
        mask[:, chosen] = 1.0

    return mask


def image_to_kspace(image: np.ndarray) -> np.ndarray:
    """Convert image to k-space via 2D FFT.

    I use fftshift/ifftshift to keep zero-frequency at the centre of k-space,
    which is the standard convention in MRI (makes masks easier to define
    and visualise).
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))


def kspace_to_image(kspace: np.ndarray) -> np.ndarray:
    """Convert k-space to image via 2D IFFT.

    The shift operations mirror image_to_kspace so that a round-trip
    (image -> kspace -> image) is an identity operation.
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))


class MRIReconDataset(Dataset):
    """Dataset for MRI reconstruction from undersampled k-space.

    Each sample provides:
        - undersampled: Zero-filled reconstruction (input)
        - target: Fully-sampled ground truth image
        - mask: Undersampling mask
        - kspace: Fully-sampled k-space (needed for data consistency layer)
        - label: Segmentation label (for downstream evaluation)

    The k-space simulation happens on-the-fly in __getitem__.  For training I
    use random masks (fixed_masks=False) so the network sees different
    undersampling patterns each epoch -- this acts as data augmentation and
    prevents overfitting to a specific mask pattern.  For validation and
    testing I use deterministic masks (fixed_masks=True, seed=idx) so that
    metrics are reproducible across runs.
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

        # Normalise image to [0, 1] per-slice.  I use per-slice min-max
        # normalisation because MRI intensity values are arbitrary (they
        # depend on scanner gain, coil loading, etc.) and vary widely
        # between slices and subjects.  The epsilon (1e-8) prevents
        # division by zero for blank slices.
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Simulate fully-sampled k-space from the magnitude image.
        # Note: this gives us a real-valued image in k-space, which means
        # the k-space is Hermitian-symmetric.  In real MRI the phase would
        # be non-trivial, but for magnitude-only simulation this is acceptable.
        kspace = image_to_kspace(image)

        # Create undersampling mask.  For training (fixed_masks=False), we get
        # a different random mask each time a sample is loaded -- effectively
        # augmenting the dataset by ~infinite mask variations.
        seed = idx if self.fixed_masks else None
        mask = create_cartesian_mask(image.shape, self.acceleration,
                                     self.center_fraction, seed=seed)

        # Apply mask to get undersampled k-space and perform zero-filled
        # reconstruction (IFFT of undersampled data).  This is the network input.
        undersampled_kspace = kspace * mask
        undersampled = np.abs(kspace_to_image(undersampled_kspace)).astype(np.float32)

        # Convert everything to PyTorch tensors with a channel dimension.
        # K-space is stored as two real channels [real, imag] rather than
        # complex tensors, because the data consistency layer and DataLoader
        # handle real tensors more reliably.
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

    # Training set: random masks for augmentation (each epoch sees new masks)
    train_ds = MRIReconDataset(
        os.path.join(base, 'train', 'npz'),
        acceleration=acceleration,
        center_fraction=center_fraction,
        fixed_masks=False
    )
    # Validation and test: fixed masks so that metrics are deterministic
    # and comparable across experiments
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

    # pin_memory=True for faster CPU -> GPU transfers.
    # drop_last=True for training to avoid uneven batch sizes which can
    # cause issues with BatchNorm (less relevant with InstanceNorm, but
    # good practice).  We keep all samples for val/test.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
