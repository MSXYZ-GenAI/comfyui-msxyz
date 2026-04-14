# ComfyUI Video Anti-Aliasing Pack (TAA + DLAA)

These ComfyUI nodes provide advanced anti-aliasing solutions for videos and images.  
The VideoTAADLAA node combines Temporal Anti-Aliasing (TAA) and Deep Learning Anti-Aliasing (DLAA) techniques to deliver sharp and smooth results, while the VideoAdaptiveAA node smooths only jagged (aliasing-prone) areas.

---

### Preview

| TAA + DLAA Anti-Aliasing | Adaptive Anti-Aliasing |
| :---: | :---: |
| ![Video TAA + DLAA Node](https://raw.githubusercontent.com/MSXYZ-GenAI/comfyui-msxyz/refs/heads/main/taa_dlaa_node.png) | ![Video Adaptive AA Node](https://github.com/MSXYZ-GenAI/comfyui-msxyz/blob/main/adaptiveaa_node.png) |

---

## Features

- **Temporal Anti-Aliasing (TAA):** Stabilizes flickering in video sequences using frame history.
- **DLAA (Deep Learning AA):** Neural-based cleanup of residual aliasing without losing texture detail.
- **Ghosting-Free Logic:** Advanced history clamping prevents trailing artifacts in fast motion.
- **Halton Sequence Jittering:** Improves sub-pixel sampling for cleaner edges.
- **Edge-Aware Weighting:** Preserves fine details while blending.
- **Adaptive Anti-Aliasing (AdaptiveAA):** Targets only aliasing-prone regions for localized smoothing.

---

## Compare

<p align="center">
  <img src="Compare1.png" width="640">
  <img src="Compare2.png" width="640">
  <img src="Compare3.png" width="640">
</p>

---

## Parameters

### TAA Parameters

**taa_strength**  
Recommended: 0.9 | Fast motion: 0.7 | Static scene: 0.95
**taa_history_alpha**  
Recommended: 0.1 | Aggressive AA: 0.3 | Ghost-free: 0.05
**jitter_scale**  
Recommended: 0.5 | Off: 0.0 | Maximum: 1.0

---

### DLAA Parameters
**dlaa_strength**  
Recommended: 0.7 | Pure AA (no sharpening): 0.0 | Aggressive sharpening: 1.0

---

### Edge AA Parameters
**edge_threshold**  
Recommended: 0.08 | Only hard edges: 0.2 | Almost everything: 0.02
**blur_radius**  
Recommended: 1 | Softer: 2–3 | Very soft: 4–5

---

### Control
**reset_history**  
Normal: False | On scene change: True (for 1 frame)

---

## Installation

Recommended: Install via ComfyUI-Manager.

Manual installation:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/MSXYZ-GenAI/comfyui-msxyz.git