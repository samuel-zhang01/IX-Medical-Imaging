"""
U-Net model with MC Dropout for MRI reconstruction with uncertainty quantification.
Includes data consistency layer for physics-informed reconstruction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class ConvBlock(nn.Module):
    """Double convolution block with optional dropout for MC Dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout_rate: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_ch)
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
    """

    def __init__(self, learnable_lambda: bool = True):
        super().__init__()
        if learnable_lambda:
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
        # Convert prediction to k-space
        pred_complex = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(x_pred.squeeze(1))),
        )

        # Reconstruct original undersampled k-space from full k-space
        kspace_complex = torch.complex(kspace_full[:, 0], kspace_full[:, 1])
        undersampled_kspace = kspace_complex * mask.squeeze(1)

        # Soft data consistency
        lam = torch.sigmoid(self.lambda_dc)
        dc_kspace = pred_complex.clone()
        dc_kspace = torch.where(
            mask.squeeze(1).bool(),
            (1 - lam) * pred_complex + lam * undersampled_kspace,
            pred_complex
        )

        # Convert back to image
        dc_image = torch.abs(torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(dc_kspace))
        ))

        return dc_image.unsqueeze(1)


class ReconUNet(nn.Module):
    """U-Net for MRI reconstruction with MC Dropout and Data Consistency.

    Architecture: 4-level encoder-decoder with skip connections.
    Supports MC Dropout for uncertainty quantification at inference time.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_features: int = 64, dropout_rate: float = 0.1,
                 use_dc: bool = True, num_dc_cascades: int = 1):
        super().__init__()
        self.use_dc = use_dc
        self.num_dc_cascades = num_dc_cascades
        f = base_features

        # Encoder
        self.enc1 = ConvBlock(in_channels, f, dropout_rate)
        self.enc2 = ConvBlock(f, f * 2, dropout_rate)
        self.enc3 = ConvBlock(f * 2, f * 4, dropout_rate)
        self.enc4 = ConvBlock(f * 4, f * 8, dropout_rate)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16, dropout_rate)

        # Decoder
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8, dropout_rate)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4, dropout_rate)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2, dropout_rate)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock(f * 2, f, dropout_rate)

        # Output - residual learning (predict the difference)
        self.out_conv = nn.Conv2d(f, out_channels, 1)

        # Pooling
        self.pool = nn.MaxPool2d(2)

        # Data consistency
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

        # Decoder with skip connections
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
        # Residual learning: predict correction
        residual = self.forward_unet(undersampled)
        recon = undersampled + residual

        # Apply data consistency cascades
        if self.use_dc and kspace is not None and mask is not None:
            for dc_layer in self.dc_layers:
                recon = dc_layer(recon, kspace, mask)

        return recon.clamp(0, 1)

    def mc_predict(self, undersampled: torch.Tensor,
                   kspace: Optional[torch.Tensor] = None,
                   mask: Optional[torch.Tensor] = None,
                   num_samples: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout prediction: returns mean and uncertainty.

        Args:
            undersampled: Zero-filled input (B, 1, H, W)
            kspace: Full k-space (B, 2, H, W)
            mask: Undersampling mask (B, 1, H, W)
            num_samples: Number of MC forward passes

        Returns:
            mean_pred: Mean reconstruction (B, 1, H, W)
            uncertainty: Pixel-wise uncertainty as std (B, 1, H, W)
        """
        self.train()  # Enable dropout
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
    """Simple U-Net for cardiac segmentation (downstream task evaluation)."""

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
