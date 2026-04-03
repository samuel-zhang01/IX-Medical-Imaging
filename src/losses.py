"""
Loss functions for MRI reconstruction.

Loss Design Rationale
---------------------
I use a combined L1 + SSIM loss, following Zhao et al. (2017) "Loss Functions
for Image Restoration with Neural Networks" and the approach adopted in many
MRI reconstruction papers (e.g., Zbontar et al. 2018, Souza et al. 2020).

Why not just MSE (L2)?  MSE penalises large errors heavily but is tolerant of
small, diffuse errors.  In practice this produces reconstructions that are
overly smooth -- the network learns to "hedge its bets" by predicting the mean
of possible outputs, blurring fine structures.  This is particularly bad for
medical images where sharp anatomical boundaries matter.

Why L1?  L1 loss penalises all errors equally regardless of magnitude, which
encourages sharper reconstructions and preserves edges better than L2.  It is
also more robust to outliers.  However, L1 alone treats each pixel independently
and ignores structural correlations.

Why SSIM?  SSIM (Wang et al., 2004) measures structural similarity by comparing
local patches in terms of luminance, contrast, and structure.  It captures
perceptual quality much better than pixel-wise losses and correlates better
with radiologist assessments of image quality.  Using 1-SSIM as a loss
directly optimises for structural fidelity.

Why combine them?  L1 provides pixel-level accuracy while SSIM provides
structural/perceptual quality.  The combination (with alpha=0.84 weighting
towards SSIM) gets the best of both worlds.  The 0.84 weight comes from
Zhao et al. (2017), who found this ratio optimal across multiple image
restoration tasks.  I kept this default because it worked well in my
experiments and is widely used in the MRI reconstruction literature.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss.

    I implement SSIM from scratch rather than using a library (e.g., torchmetrics)
    so that I have full control over the Gaussian window and can register it as
    a buffer for correct device placement during training.
    """

    def __init__(self, window_size: int = 7, channel: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, channel))

    def _create_window(self, window_size, channel):
        """Create Gaussian window for SSIM computation.

        I use a separable 2D Gaussian with sigma=1.5, which is the standard
        choice from the original SSIM paper (Wang et al., 2004).  The window
        size of 7 (rather than the original 11) is a practical choice: our
        images are 256x256, so a smaller window captures local structure at a
        finer granularity.  A 7x7 window with sigma=1.5 still covers ~2
        standard deviations in each direction, giving reasonable smoothing.

        The window is constructed as an outer product of two 1D Gaussians
        (separable), which is mathematically equivalent to a 2D Gaussian
        but cheaper to construct.
        """
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0).expand(channel, 1, -1, -1)
        return window.contiguous()

    def _ssim(self, img1, img2):
        # C1 and C2 are stability constants from the SSIM paper.  They prevent
        # division by zero when local means or variances are near zero.
        # C1 = (K1 * L)^2, C2 = (K2 * L)^2 where L=1 (data range), K1=0.01, K2=0.03
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = self.window_size // 2

        # Local means via Gaussian-weighted convolution
        mu1 = F.conv2d(img1, self.window, padding=pad, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=pad, groups=self.channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        # Local variances and covariance, computed using the identity:
        # Var(X) = E[X^2] - E[X]^2
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=pad, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=pad, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=pad, groups=self.channel) - mu12

        # SSIM formula: combines luminance, contrast, and structure comparisons.
        # The numerator terms (2*mu + C1) and (2*sigma12 + C2) measure
        # similarity, while the denominator terms normalise by the individual
        # statistics.
        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def forward(self, pred, target):
        self.window = self.window.to(pred.device)
        # Return 1 - SSIM so that minimising the loss maximises SSIM
        return 1 - self._ssim(pred, target)


class CombinedLoss(nn.Module):
    """Combined L1 + SSIM loss for MRI reconstruction.

    The weighting alpha=0.84 (i.e., 84% SSIM, 16% L1) follows the
    recommendation from Zhao et al. (2017).  Intuitively, SSIM dominates
    because structural fidelity matters more than pixel-level accuracy in
    medical imaging -- a small uniform intensity shift is clinically
    irrelevant, but a blurred boundary could obscure pathology.  The L1 term
    provides a pixel-level "anchor" that prevents the optimisation from
    getting stuck in SSIM local optima.
    """

    def __init__(self, alpha: float = 0.84):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()

    def forward(self, pred, target):
        return self.alpha * self.ssim(pred, target) + (1 - self.alpha) * self.l1(pred, target)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor,
                 data_range: float = 1.0) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(data_range^2 / MSE).  Higher is better.
    I include a small epsilon (1e-10) to avoid log(0) for perfect
    reconstructions during early debugging.
    """
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(data_range ** 2 / (mse + 1e-10))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute SSIM between prediction and target.

    Returns SSIM in [0, 1] (higher is better), not the loss form.
    """
    loss_fn = SSIMLoss()
    loss_fn.window = loss_fn.window.to(pred.device)
    return 1 - loss_fn(pred, target)


def compute_nmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Normalized Mean Squared Error.

    NMSE = MSE(pred, target) / mean(target^2).  This normalises the error
    by the signal power, making it comparable across images with different
    intensity ranges.  It is the standard error metric used in the fastMRI
    benchmark (Zbontar et al., 2018).
    """
    return F.mse_loss(pred, target) / (target ** 2).mean()
