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
  <strong>AdaptiveAA Comparison</strong>
</p>
<table align="center">
  <tr>
    <td align="center"><strong></strong></td>
    <td align="center"><strong></strong></td>
  </tr>
  <tr>
    <td><img src="Compare1.png" width="450"></td>
    <td><img src="Compare2.png" width="450"></td>
  </tr>
</table>

<br>

<p align="center">
  <strong>TAA + DLAA Comparison</strong>
  <br>
  <img src="Compare3.png" width="650">
</p>

---

### Parameters

**🚀 TAA & DLAA Parameters**
* **`taa_strength`**: Controls the impact of past frames. *(Rec: 0.9 | Fast motion: 0.7 | Static: 0.95)*
* **`taa_history_alpha`**: Determines how quickly history updates. *(Rec: 0.1 | Aggressive AA: 0.3 | Ghost-free: 0.05)*
* **`jitter_scale`**: Amount of sub-pixel shifting for edge detection. *(Rec: 0.5 | Off: 0.0 | Maximum: 1.0)*
* **`dlaa_strength`**: Strength of the deep learning sharpening. *(Rec: 0.7 | Pure AA: 0.0 | Aggressive: 1.0)*

**🎨 Edge AA & Control**
* **`edge_threshold`**: Determines which pixels are detected as edges. *(Rec: 0.08 | Hard edges: 0.2 | All: 0.02)*
* **`blur_radius`**: The radius of the blur applied to edges. *(Rec: 1 | Softer: 2–3 | Very soft: 4–5)*
* **`reset_history`**: Clears temporal history memory. *(Normal: False | On scene change: True)*

---

## Installation

Recommended: Install via ComfyUI-Manager.

Manual installation:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/MSXYZ-GenAI/comfyui-msxyz.git