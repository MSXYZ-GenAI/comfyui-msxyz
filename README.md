# ComfyUI Video Anti-Aliasing Pack
<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI--Manager-Verified-green?style=flat-square&logo=github" alt="Manager">
  <img src="https://img.shields.io/github/v/release/MSXYZ-GenAI/comfyui-msxyz?style=flat-square&color=orange" alt="Release">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/github/stars/MSXYZ-GenAI/comfyui-msxyz?style=flat-square&color=gold" alt="Stars">
</p>

High-quality anti-aliasing nodes for video and image processing in ComfyUI workflows. VideoTAADLAA combines TAA with a DLAA-inspired pipeline for sharper, more stable results, while VideoAdaptiveAA focuses on aliasing-prone regions. It uses temporal accumulation, jittered sampling, and CNN-based refinement to improve edge quality.

---

> [!IMPORTANT]
> **Disclaimer**
> This node does not use NVIDIA's closed-source SDKs or native DLAA/DLSS binaries. Instead, it is a custom adaptation of Temporal and Spatial Anti-Aliasing techniques commonly found in modern game engines, rebuilt entirely from scratch using PyTorch tensor architecture specifically for ComfyUI video post-processing.

---

## Preview
| TAA + DLAA Anti-Aliasing | Adaptive Anti-Aliasing |
| :---: | :---: |
| ![Video TAA + DLAA Node](https://github.com/MSXYZ-GenAI/comfyui-msxyz/blob/main/video_taadlaa.png) | ![Video Adaptive AA Node](https://github.com/MSXYZ-GenAI/comfyui-msxyz/blob/main/video_adaptive_aa.png) |

---

## Features
- **Temporal Anti-Aliasing (TAA):** Blends frame history to reduce flicker and sub-pixel jitter.
- **Neural Reconstruction:** DLAA-inspired model for high-quality edge restoration.
- **Hybrid Pipeline:** Combines temporal accumulation with neural reconstruction for stable, high-quality results.
- **Reduced Ghosting:** Motion-aware suppression to minimize trailing artifacts in fast motion.
- **4-Offset Jitter:** Deterministic sub-pixel jitter for sharper and more stable reconstruction.
- **Edge-Preserving Clarity:** Sobel-based detection preserves fine textures while reducing aliasing.
- **Frequency-Aware Refinement:** Uses frequency and perceptual losses for cleaner, sharper results.
- **Lightweight Model:** Optimized for fast inference with minimal VRAM usage.

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
	<td align="center"><strong></strong></td>
  </tr>
  <tr>
    <td width="33%"><video src="https://github.com/user-attachments/assets/4109726e-c5db-404b-9c0d-23d49b9641cf" width="100%" controls autoplay muted loop>
  </video></td>
    <td width="33%"><video src="https://github.com/user-attachments/assets/05f8cbb1-9388-44e5-9274-0ee80d6aa37b" width="100%" controls autoplay muted loop>
  </video></td>
    <td width="33%"><video src="https://github.com/user-attachments/assets/c61ebceb-5b15-4bc3-a708-94a692a09a61" width="100%" controls autoplay muted loop>
  </video></td>
  </tr>
</table>

---

## 🚀 How to use

Add the node to your workflow and choose a preset.

Recommended starting point:

- Use **Auto** for general footage.
- Use **Balanced** for stable video cleanup.
- Use **Sharp** for images, hair, fur, wires, and high-detail edges.
- Use **Cinematic** for smoother motion and softer temporal blending. 

---

### Presets
> Presets replace manual parameter tuning and are designed to cover most use cases.

Built-in presets for common use cases:

- **Auto** — balanced default behavior.
- **Balanced** — stable anti-aliasing with moderate sharpening.
- **Sharp** — stronger edge and detail reconstruction.
- **Cinematic** — smoother temporal blending for softer motion.

Start with **Auto**, then adjust based on your content.

---

## Installation

Recommended: Install via ComfyUI Manager.

Manual installation:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/MSXYZ-GenAI/comfyui-msxyz.git
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
