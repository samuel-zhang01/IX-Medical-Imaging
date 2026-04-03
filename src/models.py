"""
U-Net model with MC Dropout for MRI reconstruction with uncertainty quantification.
Includes data consistency layer for physics-informed reconstruction.

Architecture Rationale

I chose a U-Net backbone because it remains the de-facto standard for dense
prediction tasks in medical imaging (Ronneberger et al., 2015).  The
encoder-decoder structure with skip connections is well suited to MRI
reconstruction: the encoder captures global context (important for coherent
anatomy), while the decoder restores spatial detail, and the skip connections
preserve the fine structural information that would otherwise be lost through
successive downsampling.  This is the same family of architectures used in the
original deep cascade work of Schlemper et al. (2018) and in the fastMRI
baseline of Zbontar et al. (2018).

Key design decisions:
  - Instance Normalisation instead of Batch Norm: MRI intensities vary
    significantly between subjects and scanners.  Batch Norm computes
    statistics across the mini-batch, which couples normalisation to the
    batch composition and can be unstable when batch sizes are small (common
    on medical-imaging GPUs).  Instance Norm normalises each sample
    independently, making the model invariant to per-subject intensity
    shifts -- following the recommendation from Ulyanov et al. (2017) and
    common practice in MRI reconstruction networks.
  - LeakyReLU (slope 0.2) instead of ReLU: standard ReLU can cause "dying
    neuron" problems especially in deeper networks.  LeakyReLU keeps a small
    gradient for negative activations, which I found helps training stability
    with the relatively small medical imaging datasets we work with.
  - Residual learning: the network predicts a *correction* to the zero-filled
    input rather than the full image.  This is easier to learn because the
    zero-filled reconstruction already contains most of the structural
    information -- the network only needs to remove aliasing artefacts.
    He et al. (2016) showed this helps convergence, and it is standard in
    MRI reconstruction (e.g., DAGAN, Yang et al. 2018).
  - Soft data consistency: after the U-Net produces a prediction, I enforce
    fidelity to the *acquired* k-space measurements.  This is physics-informed:
    we know exactly which k-space lines were measured, so we should trust those
    measurements.  I use a soft (learnable) weighting rather than hard
    replacement, which is more robust to noise in the measurements
    (Schlemper et al., 2018).

Uncertainty Quantification

I implement two complementary approaches:
  1. MC Dropout (Gal & Ghahramani, 2016): by keeping dropout enabled at test
     time and running T stochastic forward passes, we approximate Bayesian
     inference.  The variance across passes captures *epistemic* (model)
     uncertainty -- regions where the model is unsure tend to produce higher
     variance.  This is cheap to implement (just keep dropout on) and does not
     require any changes to the training procedure.
  2. Deep Ensembles (Lakshminarayanan et al., 2017): training M independently
     initialised models and aggregating their predictions.  Ensembles capture
     both epistemic and a degree of aleatoric uncertainty.  The ensemble
     disagreement is often better calibrated than MC Dropout alone.
     The ensemble logic lives in the training notebooks; this module provides
     the single-model building block.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class ConvBlock(nn.Module):
    """Double convolution block with optional dropout for MC Dropout.

    I use two consecutive 3x3 convolutions per block (the "double conv"
    pattern from the original U-Net).  Each convolution is followed by
    Instance Norm and LeakyReLU.  Dropout is applied *between* convolutions
    so that the second convolution learns to be robust to missing features
    -- this is where MC Dropout draws its stochastic samples at test time.

    I use Dropout2d (spatial dropout) rather than element-wise Dropout
    because dropping entire feature maps is more appropriate for
    convolutional layers -- it prevents co-adaptation of spatially
    neighbouring activations (Tompson et al., 2015).
    """

    def __init__(self, in_ch: int, out_ch: int, dropout_rate: float = 0.0):
        super().__init__()
        # bias=False because Instance Norm already learns an affine shift,
        # so a separate convolution bias would be redundant
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_ch)
        # Dropout2d zeroes entire feature channels -- more suitable for conv
        # layers than standard Dropout which zeroes individual elements
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x


class DataConsistencyLayer(nn.Module):
    """Data consistency layer enforcing fidelity to acquired k-space measurements.

    Implements soft DC: replaces acquired k-space lines with weighted combination
    of network prediction and original measurements.

    The idea is straightforward: for k-space locations that were actually
    measured, we have ground-truth data, so the network output should not
    deviate too far from those measurements.  For locations that were NOT
    measured (i.e., missing lines), we trust the network prediction entirely.

    I use a *soft* DC formulation rather than hard replacement.  In the hard
    version you would just overwrite measured locations with the original
    data.  The soft version blends them with a learnable weight lambda:
        k_dc[measured] = (1 - lambda) * k_pred + lambda * k_measured
    This is more robust because the original measurements contain noise, and
    the network may have learned to denoise.  Letting lambda be learnable
    means the model can decide how much to trust the raw measurements vs.
    its own prediction.  I pass lambda through a sigmoid so it stays in
    [0, 1].  Following Schlemper et al. (2018), I initialise lambda to a
    small value (0.1) so the network initially trusts its own prediction
    and gradually learns the right balance.
    """

    def __init__(self, learnable_lambda: bool = True):
        super().__init__()
        if learnable_lambda:
            # Initialise raw parameter so that sigmoid(0.1) ~ 0.52, giving
            # roughly equal weight to prediction and measurement at the start
            self.lambda_dc = nn.Parameter(torch.ones(1) * 0.1)
        else:
            self.register_buffer('lambda_dc', torch.ones(1) * 0.1)

    def forward(self, x_pred: torch.Tensor, kspace_full: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Apply data consistency.

        Args:
            x_pred: Network prediction (B, 1, H, W)
            kspace_full: Full k-space (B, 2, H, W) as [real, imag]
            mask: Undersampling mask (B, 1, H, W)

        Returns:
            DC-corrected image (B, 1, H, W)
        """
        # Convert the network's image-domain prediction back to k-space so we
        # can compare/blend with the original k-space measurements.
        # fftshift/ifftshift are needed because our k-space convention has
        # zero-frequency at the centre (standard MRI convention).
        pred_complex = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(x_pred.squeeze(1))),
        )

        # Reconstruct original undersampled k-space from full k-space
        kspace_complex = torch.complex(kspace_full[:, 0], kspace_full[:, 1])
        undersampled_kspace = kspace_complex * mask.squeeze(1)

        # Soft data consistency: blend prediction and measurement at acquired
        # locations; keep prediction unchanged at unacquired locations
        lam = torch.sigmoid(self.lambda_dc)
        dc_kspace = pred_complex.clone()
        dc_kspace = torch.where(
            mask.squeeze(1).bool(),
            (1 - lam) * pred_complex + lam * undersampled_kspace,
            pred_complex
        )

        # Convert corrected k-space back to image domain.  We take the
        # magnitude because MRI images are conventionally displayed as
        # magnitude images (the phase is discarded for display).
        dc_image = torch.abs(torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(dc_kspace))
        ))

        return dc_image.unsqueeze(1)


class ReconUNet(nn.Module):
    """U-Net for MRI reconstruction with MC Dropout and Data Consistency.

    Architecture: 4-level encoder-decoder with skip connections.
    Supports MC Dropout for uncertainty quantification at inference time.

    I use 4 encoder levels (64 -> 128 -> 256 -> 512 features) with a 1024-
    feature bottleneck.  This gives a receptive field large enough to capture
    global anatomy structure in 256x256 images while keeping the model
    trainable on a single GPU.  The feature count doubles at each level
    following standard practice -- this compensates for the spatial resolution
    halving, keeping the computational cost roughly constant per level.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_features: int = 64, dropout_rate: float = 0.1,
                 use_dc: bool = True, num_dc_cascades: int = 1):
        super().__init__()
        self.use_dc = use_dc
        self.num_dc_cascades = num_dc_cascades
        f = base_features

        # Encoder path: progressively downsample and increase feature channels.
        # Each level captures features at a different spatial scale.
        self.enc1 = ConvBlock(in_channels, f, dropout_rate)
        self.enc2 = ConvBlock(f, f * 2, dropout_rate)
        self.enc3 = ConvBlock(f * 2, f * 4, dropout_rate)
        self.enc4 = ConvBlock(f * 4, f * 8, dropout_rate)

        # Bottleneck: highest-level representation with the largest receptive
        # field.  This is where the model captures the most global context.
        self.bottleneck = ConvBlock(f * 8, f * 16, dropout_rate)

        # Decoder path: upsample with transposed convolutions and concatenate
        # skip connections.  I use ConvTranspose2d (learned upsampling) rather
        # than bilinear interpolation because it gave slightly better results
        # in my experiments, though the difference is small.
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8, dropout_rate)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4, dropout_rate)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2, dropout_rate)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock(f * 2, f, dropout_rate)

        # Output: 1x1 conv to map features to single-channel output.
        # This predicts the *residual* (artefact correction), not the full
        # image.  The residual is added to the zero-filled input below.
        self.out_conv = nn.Conv2d(f, out_channels, 1)

        self.pool = nn.MaxPool2d(2)

        # Data consistency layers -- one per cascade.  Even a single DC layer
        # helps significantly.  Multiple cascades (num_dc_cascades > 1) apply
        # the DC constraint iteratively, which is conceptually similar to
        # unrolled optimisation (Hammernik et al., 2018).
        if use_dc:
            self.dc_layers = nn.ModuleList([
                DataConsistencyLayer(learnable_lambda=True)
                for _ in range(num_dc_cascades)
            ])

    def forward_unet(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net backbone."""
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections: concatenate encoder features at each
        # level so that fine spatial details are preserved through the network
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)

    def forward(self, undersampled: torch.Tensor,
                kspace: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional data consistency.

        Args:
            undersampled: Zero-filled input (B, 1, H, W)
            kspace: Full k-space for DC (B, 2, H, W)
            mask: Undersampling mask (B, 1, H, W)

        Returns:
            Reconstructed image (B, 1, H, W)
        """
        # Residual learning: the network predicts a correction to the
        # zero-filled input.  This is much easier to learn than predicting
        # the full image from scratch, because the zero-filled reconstruction
        # already contains the bulk of the image content.
        residual = self.forward_unet(undersampled)
        recon = undersampled + residual

        # Apply data consistency cascades: each cascade re-enforces agreement
        # with the acquired k-space measurements.  This acts as a
        # physics-informed regulariser.
        if self.use_dc and kspace is not None and mask is not None:
            for dc_layer in self.dc_layers:
                recon = dc_layer(recon, kspace, mask)

        # Clamp to [0, 1] because MRI magnitude images are non-negative
        # and our normalisation maps them to this range
        return recon.clamp(0, 1)

    def mc_predict(self, undersampled: torch.Tensor,
                   kspace: Optional[torch.Tensor] = None,
                   mask: Optional[torch.Tensor] = None,
                   num_samples: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout prediction: returns mean and uncertainty.

        Following Gal & Ghahramani (2016), I keep dropout active at test time
        and run T stochastic forward passes.  Each pass samples a different
        dropout mask, effectively sampling from an approximate posterior over
        network weights.  The mean of these T predictions is a better point
        estimate than a single forward pass, and the standard deviation gives
        a pixel-wise measure of epistemic (model) uncertainty.

        I use T=20 by default as a reasonable trade-off between uncertainty
        estimate quality and computational cost.  Gal & Ghahramani showed
        that T >= 10 is usually sufficient for well-calibrated uncertainty.

        Args:
            undersampled: Zero-filled input (B, 1, H, W)
            kspace: Full k-space (B, 2, H, W)
            mask: Undersampling mask (B, 1, H, W)
            num_samples: Number of MC forward passes (T)

        Returns:
            mean_pred: Mean reconstruction (B, 1, H, W)
            uncertainty: Pixel-wise uncertainty as std (B, 1, H, W)
        """
        self.train()  # Enable dropout -- this is the key trick for MC Dropout
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(undersampled, kspace, mask)
                preds.append(pred)

        preds = torch.stack(preds, dim=0)  # (T, B, 1, H, W)
        mean_pred = preds.mean(dim=0)
        uncertainty = preds.std(dim=0)

        return mean_pred, uncertainty


class SegmentationUNet(nn.Module):
    """Simple U-Net for cardiac segmentation (downstream task evaluation).

    I use this as a frozen downstream task to evaluate reconstruction quality
    indirectly: if the reconstruction preserves clinically relevant structure,
    a segmentation model trained on fully-sampled images should still perform
    well on reconstructed images.  This is a more clinically meaningful metric
    than PSNR/SSIM alone.

    This network is deliberately simpler (3 levels, 32 base features, no
    dropout) because it only needs to segment ~8 cardiac structures at
    256x256 resolution.  No dropout because we do not need uncertainty
    here -- it is purely a deterministic evaluation tool.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 8,
                 base_features: int = 32):
        super().__init__()
        f = base_features

        self.enc1 = ConvBlock(in_channels, f, 0.0)
        self.enc2 = ConvBlock(f, f * 2, 0.0)
        self.enc3 = ConvBlock(f * 2, f * 4, 0.0)

        self.bottleneck = ConvBlock(f * 4, f * 8, 0.0)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4, 0.0)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2, 0.0)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock(f * 2, f, 0.0)

        self.out_conv = nn.Conv2d(f, num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)
