# ComfyUI-FUDA

**Fourier-based Unsupervised Domain Adaptation nodes for ComfyUI.**

Implements the FUDA method from:

> *Boosting unsupervised domain adaptation: A Fourier approach*
> Mengzhu Wang, Shanshan Wang, Ye Wang, Wei Wang, Tianyi Liang, Junyang Chen, Zhigang Luo
> Knowledge-Based Systems, 2023. https://doi.org/10.1016/j.knosys.2023.110325

---

## How It Works

The Fourier transform decomposes an image into two components:

- **Amplitude** — encodes low-level statistics: colour, brightness, and style.
- **Phase** — encodes high-level semantics: structure, edges, and content.

FUDA exploits this property to transfer the visual style of a **reference** image onto a **source** image while keeping its content intact:

1. Apply 2-D FFT to both images and shift the DC component to the centre.
2. Mix the **low-frequency amplitude** of the reference into the source using a weighted blend controlled by `beta` (region size) and `alpha` (blend strength).
3. Reconstruct via inverse FFT using the mixed amplitude and the original source phase.

An optional **Fourier Transform Channel Attention (FTCA)** module further recalibrates channel responses based on per-channel spectral energy, as described in the paper.

---

## Nodes

All nodes live under the **`FUDA`** category in the ComfyUI node menu.

---

### FUDA Image Adaptation

The core node. Transfers the low-level style of `reference_image` onto `source_image`.

| Port | Type | Description |
|---|---|---|
| `source_image` | IMAGE (input) | Image to be style-adapted |
| `reference_image` | IMAGE (input) | Target-domain reference image |
| `beta` | FLOAT 0.001–0.5 | Low-frequency band radius as a fraction of the shorter spatial dimension. Small values (0.01–0.1) affect global colour and tone; larger values also mix mid-frequency texture. **Default: 0.09** |
| `alpha` | FLOAT 0–1 | Blend weight for the reference amplitude. `0` = no change, `1` = full reference style. **Default: 0.5** |
| `adapted_image` | IMAGE (output) | Style-adapted result |

**Tips:**
- Start with `beta = 0.09`, `alpha = 0.5` and adjust from there.
- Increasing `beta` broadens the spectral region mixed — more texture change but potentially more artefacts.
- `alpha = 1.0` reproduces the original FDA (Fourier Domain Adaptation) method.

---

### FUDA + Channel Attention

Extends the core node with a **Fourier Transform Channel Attention (FTCA)** pass. After amplitude mixing, FTCA computes per-channel spectral energy weights through a small MLP and recalibrates the adapted image to sharpen discriminative features.

| Port | Type | Description |
|---|---|---|
| `source_image` | IMAGE (input) | Image to be style-adapted |
| `reference_image` | IMAGE (input) | Target-domain reference image |
| `beta` | FLOAT 0.001–0.5 | Low-frequency band ratio (see above) |
| `alpha` | FLOAT 0–1 | Reference amplitude blend weight |
| `attention_strength` | FLOAT 0–1 | Blend between raw adaptation (`0`) and FTCA-refined output (`1`). **Default: 0.5** |
| `reduction` | INT 1–16 | Channel reduction ratio inside the attention MLP. Higher = fewer parameters. **Default: 4** |
| `adapted_image` | IMAGE (output) | Attention-refined, style-adapted result |

---

### FUDA Amplitude Visualiser

Renders the centred Fourier amplitude spectrum of an image as a greyscale heatmap. Useful for comparing domain differences before and after adaptation.

| Port | Type | Description |
|---|---|---|
| `image` | IMAGE (input) | Any image |
| `log_scale` | BOOLEAN | Apply `log(1 + amplitude)` compression for better dynamic range visualisation. **Default: true** |
| `amplitude_map` | IMAGE (output) | Normalised amplitude heatmap (bright = high energy) |

---

## Installation

### Option A — Clone into `custom_nodes`

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/your-repo/ComfyUI_FUDA
```

### Option B — Symlink an existing clone

```bash
ln -s /Users/m1_4k/ComfyUI_FUDA /path/to/ComfyUI/custom_nodes/ComfyUI_FUDA
```

Restart ComfyUI. The three nodes will appear under the **FUDA** category.

### Requirements

No additional packages are needed beyond what ComfyUI already installs.

| Dependency | Version |
|---|---|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0 (for `torch.fft`) |
| NumPy | any recent |

---

## Example Workflow

```
[Load Image] ──► source_image ─┐
                                ├─► [FUDA Image Adaptation] ──► [Preview Image]
[Load Image] ──► reference_image ─┘        beta=0.09  alpha=0.5
```

To inspect the effect in the frequency domain:

```
[Load Image] ──► [FUDA Amplitude Visualiser] ──► [Preview Image]  (before)
[adapted_image] ──► [FUDA Amplitude Visualiser] ──► [Preview Image]  (after)
```

---

## Parameter Guide

| Goal | Suggested settings |
|---|---|
| Subtle colour grading | `beta=0.03`, `alpha=0.3` |
| Moderate style transfer | `beta=0.09`, `alpha=0.5` |
| Strong style transfer | `beta=0.2`, `alpha=0.8` |
| Maximum reference style | `beta=0.3`, `alpha=1.0` |
| Add attention refinement | Use **FUDA + Channel Attention**, `attention_strength=0.5` |

---

## Background: Why Fourier?

The Fourier transform separates an image's *what* (phase → semantics) from its *how it looks* (amplitude → style). By blending only the low-frequency amplitude:

- The **content and structure** of the source image are fully preserved (phase is untouched).
- The **colour palette, illumination, and global style** shift toward the reference domain.

This is the foundation of **FDA** (Fourier Domain Adaptation, Yang et al., 2020), which FUDA extends by:
1. Using a **weighted blend** instead of a hard replacement.
2. Adding **Fourier Transform Channel Attention** to capture richer spectral feature diversity.

---

## Citation

If you use this node in your work, please cite the original paper:

```bibtex
@article{wang2023fuda,
  title   = {Boosting unsupervised domain adaptation: A Fourier approach},
  author  = {Wang, Mengzhu and Wang, Shanshan and Wang, Ye and Wang, Wei
             and Liang, Tianyi and Chen, Junyang and Luo, Zhigang},
  journal = {Knowledge-Based Systems},
  year    = {2023},
  doi     = {10.1016/j.knosys.2023.110325}
}
```

---

## License

MIT
