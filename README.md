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
    <td><img src="Compare1.png?raw=true" width="450"></td>
    <td><img src="Compare2.png?raw=true" width="450"></td>
  </tr>
</table>

<br>

<p align="center">
  <strong>TAA + DLAA Comparison</strong>
  <br>
  <img src="Compare3.png?raw=true" width="650">
</p>

---

### Parameters

#### 🚀 TAA Parameters

*   **`taa_strength`**: Controls the overall impact of past frames on the current frame. Higher values yield smoother results but increase the risk of ghosting on moving objects.
    *   **Recommended: `0.9`** - Provides a good balance for most situations.
    *   **Fast motion: `0.7`** - Reduces ghosting in fast-paced scenes.
    *   **Static scene: `0.95`** - Maximizes smoothness in still/slow scenes.

*   **`taa_history_alpha`**: Determines how quickly the history frames are updated. Lower values update the history slower, providing a more stable image.
    *   **Recommended: `0.1`** - Offers good stability.
    *   **Aggressive AA: `0.3`** - More aggressive smoothing, but might introduce flickering.
    *   **Ghost-free: `0.05`** - Minimizes ghosting trails.

*   **`jitter_scale`**: Controls the amount of sub-pixel shifting (jitter) applied to each frame for superior edge detection.
    *   **Recommended: `0.5`** - Ideal for effective anti-aliasing.
    *   **Off: `0.0`** - Disables jittering.
    *   **Maximum: `1.0`** - Maximum effect, but might cause slight blurriness in fine details.

#### ✨ DLAA Parameters

*   **`dlaa_strength`**: Adjusts the strength of the deep learning-based sharpening and edge correction neural network.
    *   **Recommended: `0.7`** - Retains TAA smoothness while sharpening the image.
    *   **Pure AA (no sharpening): `0.0`** - Only uses TAA and Edge AA effects.
    *   **Aggressive sharpening: `1.0`** - Provides maximum sharpening.

#### 🎨 Edge AA Parameters

*   **`edge_threshold`**: The threshold value determining which pixels are detected as edges. Lower values will target more fine details.
    *   **Recommended: `0.08`** - Good balance between fine and prominent edges.
    *   **Only hard edges: `0.2`** - Targets only high-contrast, prominent edges.
    *   **Almost everything: `0.02`** - Applies edge smoothing to a massive portion of the image.

*   **`blur_radius`**: The radius of the blur applied to the detected edges.
    *   **Recommended: `1`** - Provides light and effective edge smoothing.
    *   **Softer: `2–3`** - For noticeably softer edges.
    *   **Very soft: `4–5`** - For heavily smoothed, almost "cartoonish" edges.

#### ⚙️ Control

*   **`reset_history`**: Clears the temporal history memory.
    *   **Normal: `False`** - Memory persists continuously between frames.
    *   **On scene change: `True` (for 1 frame)** - Enable this for exactly one frame during a scene cut to prevent severe ghosting artifacts from the previous scene.

---

## Installation

Recommended: Install via ComfyUI-Manager.

Manual installation:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/MSXYZ-GenAI/comfyui-msxyz.git