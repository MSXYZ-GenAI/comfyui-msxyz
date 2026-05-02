# ComfyUI Video Anti-Aliasing Pack
<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI--Manager-Verified-green?style=flat-square&logo=github" alt="Manager">
  <img src="https://img.shields.io/github/v/release/MSXYZ-GenAI/comfyui-msxyz?style=flat-square&color=orange" alt="Release">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/github/stars/MSXYZ-GenAI/comfyui-msxyz?style=flat-square&color=gold" alt="Stars">
</p>

Anti-aliasing nodes for ComfyUI image and video workflows. VideoTAADLAA combines TAA with a DLAA-inspired pipeline for sharper, more stable results, while VideoAdaptiveAA focuses on aliasing-prone regions. It uses temporal accumulation, jittered sampling, and CNN-based refinement to improve edge quality.

---

> [!IMPORTANT]
> **Disclaimer**
> This node does not use NVIDIA's closed-source SDKs or native DLAA/DLSS binaries. Instead, it is a custom adaptation of Temporal and Spatial Anti-Aliasing techniques commonly found in modern game engines, rebuilt entirely from scratch using PyTorch tensor architecture specifically for ComfyUI video post-processing.

---

## Preview
| TAA + DLAA Anti-Aliasing | Adaptive Anti-Aliasing |
| :---: | :---: |
| <img src="https://github.com/MSXYZ-GenAI/comfyui-msxyz/blob/main/assets/video_taadlaa.png" width="100%"> | <img src="https://github.com/MSXYZ-GenAI/comfyui-msxyz/blob/main/assets/video_adaptive_aa.png" width="100%"> |

---

### Models

**DLAANet.safetensors** is the main 1x DLAA refinement model for reducing aliasing artifacts while preserving edge sharpness.
**DLAATexture.safetensors** is a 1x detail refinement model for restoring fine texture and micro-detail after the DLAA pass.

**Note:** Both models keep the original image resolution and are not ESRGAN models.

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
    <td width="33%"><img src="assets/compare_1.png" width="100%"></td>
    <td width="33%"><img src="assets/compare_2.png" width="100%"></td>
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
    <td width="33%"><video src="https://github.com/user-attachments/assets/ddf7c74b-289d-48a7-ab43-47558aa9deab" width="100%" controls autoplay muted loop>
  </video></td>
  </tr>
</table>

---

## 🚀 How to use

Add the node to your workflow and choose a preset.

Recommended starting point:

- Use **Auto** for general footage.
- Use **Performance** for longer videos or faster previews.
- Use **Balanced** for stable video cleanup with a good quality/speed balance.
- Use **High Detail** for hair, fur, wires, fine lines, and high-detail edges.

---

## Presets

The node uses a simple game-style preset system:

| Preset | Description |
|---|---|
| **Auto** | Automatically adjusts settings based on scene motion. Good default for most videos. |
| **Performance** | Faster processing for longer videos. Uses lighter detail cleanup while keeping the image stable. |
| **Balanced** | General-purpose preset with a balance between quality and speed. |
| **High Detail** | Best quality for fine detail, edge cleanup, and temporal refinement. Slower. |

Start with **Auto**, then adjust based on your content.

---

## Installation

1. Install via **ComfyUI Manager** (Search for "Anti-Aliasing Pack") or clone this repo into your `custom_nodes` folder.
2. **Note:** The required models will be automatically downloaded when you first run the node. 
3. No additional Python dependencies (requirements) are required.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
