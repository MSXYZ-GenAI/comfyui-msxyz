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
| ![Video TAA + DLAA Node](https://github.com/MSXYZ-GenAI/comfyui-msxyz/blob/main/video_taadlaa.png) | ![Video Adaptive AA Node](https://github.com/MSXYZ-GenAI/comfyui-msxyz/blob/main/video_adaptive_aa.png) |

---

## Features
- **Temporal Anti-Aliasing (TAA):** Blends frame history with motion data to eliminate flicker and sub-pixel jitter.
- **Neural Reconstruction (MSE 0.000023):** DLAA-inspired model for near-lossless, high-fidelity edge restoration.
- **Ghosting-Free Tech:** Motion-aware suppression logic to minimize trailing artifacts in high-speed sequences.
- **4-Offset Jitter:** Deterministic sub-pixel sampling for sharper and more consistent reconstruction stability.
- **Edge-Preserving Clarity:** Sobel-based detection preserves fine textures while eliminating aliasing artifacts.
- **Frequency-Aware Refinement:** Frequency and Perceptual loss optimization for crystal-clear, blur-free results.

---

## Comparison
<p align="center">
  <strong>AdaptiveAA Comparison</strong>
</p>
<table align="center">
  <tr>
    <td align="center"><strong></strong></td>
    <td align="center"><strong></strong></td>
  </tr>
  <tr>
    <td><img src="compare_1.png" width="450"></td>
    <td><img src="compare_2.png" width="450"></td>
  </tr>
</table>

<br>

<p align="center">
  <strong>TAA + DLAA Comparison</strong>
</p>
<table align="center">
  <tr>
    <td align="center"><strong></strong></td>
    <td align="center"><strong></strong></td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/4109726e-c5db-404b-9c0d-23d49b9641cf" width="500" controls autoplay muted loop>
  </video></td>
    <td><video src="https://github.com/user-attachments/assets/05f8cbb1-9388-44e5-9274-0ee80d6aa37b" width="500" controls autoplay muted loop>
  </video></td>
    <td><video src="https://github.com/user-attachments/assets/ac2ac0a8-45b9-44fb-9466-4290e1546ba4" width="500" controls autoplay muted loop>
  </video></td>
  </tr>
</table>

---

## 🚀 How to use
> - Test the effect by increasing `dlaa_strength` to **0.8** or **1.0**.  

**Parameters**
- **taa_strength:** Controls the influence of previous frames to reduce flickering.
- (Default: 0.5 | Fast Motion: 0.4 | Static: 0.9)
- **dlaa_strength:** Power of the Smart DLAA neural reconstruction model.
- (Default: 0.6 | Subtle: 0.4 | Aggressive: 1.0)
- **sharpen_strength:** Final crispness adjustment to enhance edge and texture definition.
- (Default: 0.2 | Off: 0.0 | Maximum: 2.0)
- **motion_sensitivity:** Threshold to disable temporal smoothing in moving areas.
- (Default: 0.1 | Flicker-free: 0.08 | High-speed: 0.3)

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
