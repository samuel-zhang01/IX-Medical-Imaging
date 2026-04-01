"""
Loss functions for MRI reconstruction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss."""

    def __init__(self, window_size: int = 7, channel: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, channel))

    def _create_window(self, window_size, channel):
        """Create Gaussian window for SSIM computation."""
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0).expand(channel, 1, -1, -1)
        return window.contiguous()

    def _ssim(self, img1, img2):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = self.window_size // 2

        mu1 = F.conv2d(img1, self.window, padding=pad, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=pad, groups=self.channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=pad, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=pad, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=pad, groups=self.channel) - mu12

        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def forward(self, pred, target):
        self.window = self.window.to(pred.device)
        return 1 - self._ssim(pred, target)


class CombinedLoss(nn.Module):
    """Combined L1 + SSIM loss for MRI reconstruction."""

    def __init__(self, alpha: float = 0.84):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()

    def forward(self, pred, target):
        return self.alpha * self.ssim(pred, target) + (1 - self.alpha) * self.l1(pred, target)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor,
                 data_range: float = 1.0) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(data_range ** 2 / (mse + 1e-10))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute SSIM between prediction and target."""
    loss_fn = SSIMLoss()
    loss_fn.window = loss_fn.window.to(pred.device)
    return 1 - loss_fn(pred, target)


def compute_nmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Normalized Mean Squared Error."""
    return F.mse_loss(pred, target) / (target ** 2).mean()
