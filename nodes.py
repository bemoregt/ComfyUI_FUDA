"""
FUDA: Fourier-based Unsupervised Domain Adaptation for ComfyUI
Based on: "Boosting unsupervised domain adaptation: A Fourier approach"
Wang et al., Knowledge-Based Systems, 2023
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core FUDA transforms
# ---------------------------------------------------------------------------

def fuda_amplitude_mix(src: torch.Tensor, ref: torch.Tensor,
                        beta: float = 0.1, alpha: float = 0.5) -> torch.Tensor:
    """
    Fourier Domain Adaptation: fuse low-frequency amplitude of reference
    into source, keeping source phase intact.

    Args:
        src  : [B, C, H, W] float tensor in [0, 1]
        ref  : [B, C, H, W] float tensor in [0, 1] (same spatial size as src)
        beta : fraction of the spectrum (from DC) to mix  [0, 0.5]
        alpha: blend weight for reference amplitude       [0, 1]
    Returns:
        adapted [B, C, H, W] float tensor
    """
    B, C, H, W = src.shape

    fft_src = torch.fft.fft2(src)
    fft_ref = torch.fft.fft2(ref)

    # shift DC to centre
    fft_src = torch.fft.fftshift(fft_src, dim=(-2, -1))
    fft_ref = torch.fft.fftshift(fft_ref, dim=(-2, -1))

    amp_src = torch.abs(fft_src)
    pha_src = torch.angle(fft_src)
    amp_ref = torch.abs(fft_ref)

    b = max(1, int(np.floor(min(H, W) * beta)))
    hc, wc = H // 2, W // 2
    h0, h1 = max(0, hc - b), min(H, hc + b)
    w0, w1 = max(0, wc - b), min(W, wc + b)

    amp_new = amp_src.clone()
    amp_new[:, :, h0:h1, w0:w1] = (
        (1.0 - alpha) * amp_src[:, :, h0:h1, w0:w1]
        + alpha        * amp_ref[:, :, h0:h1, w0:w1]
    )

    fft_new = torch.polar(amp_new, pha_src)
    fft_new = torch.fft.ifftshift(fft_new, dim=(-2, -1))
    adapted = torch.fft.ifft2(fft_new).real
    return adapted


# ---------------------------------------------------------------------------
# Fourier Transform Channel Attention (FTCA)
# Lightweight pixel-space version: reweights channels via their spectral energy
# ---------------------------------------------------------------------------

class FourierChannelAttention(nn.Module):
    """
    Pixel-space Fourier Channel Attention.
    Computes per-channel spectral energy weights and recalibrates the feature map.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        fft_x = torch.fft.fft2(x)
        energy = (torch.abs(fft_x) ** 2).mean(dim=(-2, -1))  # [B, C]
        w = self.fc(energy).unsqueeze(-1).unsqueeze(-1)       # [B, C, 1, 1]
        return x * w


# ---------------------------------------------------------------------------
# ComfyUI Nodes
# ---------------------------------------------------------------------------

class FUDANode:
    """
    FUDA Image Adaptation
    ─────────────────────
    Adapts *source_image* toward the visual style of *reference_image* using
    Fourier-domain amplitude mixing (the core FUDA transform from the paper).

    • beta  – controls the size of the low-frequency region that is mixed.
              Small values (0.01–0.1) transfer colour/style; larger values
              can also affect mid-frequency texture.
    • alpha – blend weight: 0 = no change, 1 = full reference amplitude.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image":    ("IMAGE",),
                "reference_image": ("IMAGE",),
                "beta": ("FLOAT", {
                    "default": 0.09,
                    "min": 0.001,
                    "max": 0.5,
                    "step": 0.001,
                    "display": "slider",
                    "tooltip": "Low-frequency band ratio (0.01–0.5). Higher → wider spectral mix.",
                }),
                "alpha": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Reference amplitude weight. 0=no change, 1=full reference style.",
                }),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("adapted_image",)
    FUNCTION      = "adapt"
    CATEGORY      = "FUDA"
    DESCRIPTION   = "Fourier Domain Adaptation – adapts source image style toward a reference domain."

    def adapt(self, source_image: torch.Tensor, reference_image: torch.Tensor,
              beta: float, alpha: float):
        # ComfyUI tensors: [B, H, W, C] float32
        src = source_image.permute(0, 3, 1, 2)   # → [B, C, H, W]
        ref = reference_image.permute(0, 3, 1, 2)

        # broadcast: match batch dims
        if src.shape[0] != ref.shape[0]:
            if ref.shape[0] == 1:
                ref = ref.expand(src.shape[0], -1, -1, -1)
            elif src.shape[0] == 1:
                src = src.expand(ref.shape[0], -1, -1, -1)

        # Resize reference spatially if needed
        if src.shape[-2:] != ref.shape[-2:]:
            ref = F.interpolate(ref, size=src.shape[-2:],
                                mode="bilinear", align_corners=False)

        adapted = fuda_amplitude_mix(src, ref, beta=beta, alpha=alpha)
        adapted = adapted.clamp(0.0, 1.0).permute(0, 2, 3, 1)  # → [B, H, W, C]
        return (adapted,)


class FUDAWithAttentionNode:
    """
    FUDA Image Adaptation + Fourier Channel Attention
    ──────────────────────────────────────────────────
    Same as FUDANode but also applies a Fourier-based channel attention pass
    to enhance discriminative features in the adapted result.

    Extra parameters:
    • attention_strength – blends the attention-refined output with the raw
                           adapted output (0 = skip attention, 1 = full FTCA).
    • reduction          – channel reduction ratio in the attention MLP.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image":      ("IMAGE",),
                "reference_image":   ("IMAGE",),
                "beta": ("FLOAT", {
                    "default": 0.09,
                    "min": 0.001, "max": 0.5, "step": 0.001,
                    "display": "slider",
                }),
                "alpha": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider",
                }),
                "attention_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider",
                    "tooltip": "0 = no attention, 1 = full FTCA recalibration.",
                }),
                "reduction": ("INT", {
                    "default": 4,
                    "min": 1, "max": 16, "step": 1,
                    "tooltip": "Channel reduction ratio for attention MLP.",
                }),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("adapted_image",)
    FUNCTION      = "adapt_with_attention"
    CATEGORY      = "FUDA"
    DESCRIPTION   = "FUDA + Fourier Channel Attention for enhanced style adaptation."

    def adapt_with_attention(self, source_image: torch.Tensor,
                              reference_image: torch.Tensor,
                              beta: float, alpha: float,
                              attention_strength: float, reduction: int):
        src = source_image.permute(0, 3, 1, 2)
        ref = reference_image.permute(0, 3, 1, 2)

        if src.shape[0] != ref.shape[0]:
            if ref.shape[0] == 1:
                ref = ref.expand(src.shape[0], -1, -1, -1)
            elif src.shape[0] == 1:
                src = src.expand(ref.shape[0], -1, -1, -1)

        if src.shape[-2:] != ref.shape[-2:]:
            ref = F.interpolate(ref, size=src.shape[-2:],
                                mode="bilinear", align_corners=False)

        # Step 1: Fourier amplitude adaptation
        adapted = fuda_amplitude_mix(src, ref, beta=beta, alpha=alpha)

        # Step 2: Fourier Channel Attention
        C = adapted.shape[1]
        ftca = FourierChannelAttention(channels=C, reduction=reduction)
        ftca.eval()
        with torch.no_grad():
            refined = ftca(adapted)

        # Blend
        out = (1.0 - attention_strength) * adapted + attention_strength * refined
        out = out.clamp(0.0, 1.0).permute(0, 2, 3, 1)
        return (out,)


class FUDAAmplitudeVisNode:
    """
    FUDA Amplitude Visualiser
    ─────────────────────────
    Outputs a visual representation of the Fourier amplitude spectrum of an
    image. Useful for understanding domain differences.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "log_scale": ("BOOLEAN", {"default": True,
                              "tooltip": "Apply log(1+amp) for better visualisation."}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("amplitude_map",)
    FUNCTION      = "visualise"
    CATEGORY      = "FUDA"
    DESCRIPTION   = "Visualise Fourier amplitude spectrum (shifted, per-channel mean)."

    def visualise(self, image: torch.Tensor, log_scale: bool):
        x = image.permute(0, 3, 1, 2)                          # [B,C,H,W]
        fft_x = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        amp = torch.abs(fft_x).mean(dim=1, keepdim=True)       # [B,1,H,W]
        if log_scale:
            amp = torch.log1p(amp)
        # Normalise per image to [0,1]
        B = amp.shape[0]
        amp_flat = amp.view(B, -1)
        mn = amp_flat.min(dim=1).values.view(B, 1, 1, 1)
        mx = amp_flat.max(dim=1).values.view(B, 1, 1, 1)
        amp = (amp - mn) / (mx - mn + 1e-8)
        amp = amp.expand(-1, 3, -1, -1).permute(0, 2, 3, 1)    # [B,H,W,3]
        return (amp.clamp(0, 1),)


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "FUDANode":               FUDANode,
    "FUDAWithAttentionNode":  FUDAWithAttentionNode,
    "FUDAAmplitudeVisNode":   FUDAAmplitudeVisNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FUDANode":               "FUDA Image Adaptation",
    "FUDAWithAttentionNode":  "FUDA + Channel Attention",
    "FUDAAmplitudeVisNode":   "FUDA Amplitude Visualiser",
}
