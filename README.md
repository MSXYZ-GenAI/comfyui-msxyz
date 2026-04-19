# ComfyUI Video Anti-Aliasing Pack
<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI--Manager-Verified-green?style=flat-square&logo=github" alt="Manager">
  <img src="https://img.shields.io/github/v/release/MSXYZ-GenAI/comfyui-msxyz?style=flat-square&color=orange" alt="Release">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/github/stars/MSXYZ-GenAI/comfyui-msxyz?style=flat-square&color=gold" alt="Stars">
</p>

Advanced anti-aliasing nodes for videos and images. VideoTAADLAA combines TAA with a DLAA-inspired pipeline for sharp, stable results, while VideoAdaptiveAA targets specific aliasing-prone regions. The system utilizes temporal accumulation, jittered sampling, and CNN-based refinement for superior edge quality.

---

> [!IMPORTANT]
> **Disclaimer**
> This node does not use NVIDIA's closed-source SDKs or native DLAA/DLSS binaries. Instead, it is a custom adaptation of Temporal and Spatial Anti-Aliasing techniques commonly found in modern game engines, rebuilt entirely from scratch using PyTorch tensor architecture specifically for ComfyUI video post-processing.

---

### Preview

| TAA + DLAA Anti-Aliasing | Adaptive Anti-Aliasing |
| :---: | :---: |
| ![Video TAA + DLAA Node](https://raw.githubusercontent.com/MSXYZ-GenAI/comfyui-msxyz/refs/heads/main/VideoTAADLAA.png) | ![Video Adaptive AA Node](https://github.com/MSXYZ-GenAI/comfyui-msxyz/blob/main/VideoAdaptiveAA.png) |

---

## Features

- **Temporal Anti-Aliasing (TAA):** Stabilizes flickering in video sequences using frame history.
- **DLAA-inspired neural refinement:** Neural-based cleanup of residual aliasing without losing texture detail.
- **Ghosting-Free Logic:** Advanced history clamping prevents trailing artifacts in fast motion.
- **Halton Sequence Jittering:** Improves sub-pixel sampling for cleaner edges.
- **Edge-Aware Weighting:** Preserves fine details while blending.
- **Adaptive Anti-Aliasing (AdaptiveAA):** Targets only aliasing-prone regions for localized smoothing.

---

## Comparison

---

<p align="center">
  <strong>AdaptiveAA Comparison</strong>
</p>
<table align="center">
  <tr>
    <td align="center"><strong></strong></td>
    <td align="center"><strong></strong></td>
  </tr>
  <tr>
    <td><img src="Compare_1.png" width="450"></td>
    <td><img src="Compare_2.png" width="450"></td>
  </tr>
</table>

<br>

<p align="center">
  <strong>TAA + DLAA Comparison</strong>
  <br>
  <img src="Compare_3.png" width="650">
</p>

---

## 🚀 How to use
> - Test the effect by increasing `dlaa_strength` to **0.8** or **1.0**.  
> - If you notice ghosting behind fast-moving objects, try slightly increasing `motion_sensitivity` or decreasing `taa_alpha`.

**Parameters**
* **`taa_strength`**: Controls the impact of past frames. *(Default: 0.6 | Fast motion: 0.7 | Static: 0.95)*
* **`taa_alpha`**: Determines how quickly history updates. *(Default: 0.4 | Aggressive AA: 0.3 | Ghost-free: 0.05)*
* **`jitter_scale`**: Amount of sub-pixel shifting for edge detection. *(Default: 0.02 | Off: 0.0 | Maximum: 0.08)*
* **`dlaa_strength`**: Strength of the deep learning sharpening. *(Default: 0.3 | Pure AA: 0.0 | Aggressive: 1.0)*
* **`edge_threshold`**: Determines which pixels are detected as edges. *(Default: 0.2 | Hard edges: 0.35 | All: 0.05)*
* **`blur_radius`**: The radius of the blur applied to edges. *(Default: 0 | Softer: 2–3 | Very soft: 4–5)*
* **`reset_history`**: Clears temporal history memory. *(Default: False | Enable on scene change: True)*

---

## Installation

Recommended: Install via ComfyUI-Manager.

Manual installation:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/MSXYZ-GenAI/comfyui-msxyz.git
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.