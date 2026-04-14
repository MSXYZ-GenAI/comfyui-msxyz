# ComfyUI Video Anti-Aliasing Pack (TAA + DLAA)
These ComfyUI nodes provide advanced anti-aliasing solutions for videos and images. The VideoTAADLAA node combines both Temporal Anti-Aliasing (TAA) and Deep Learning Anti-Aliasing (DLAA) techniques to deliver sharp and smooth results, while the VideoAdaptiveAA node smooths only the jagged (aliasing-prone) areas.

<p align="center" style="display: flex; justify-content: center; gap: 10px;">
  <img src="taa_dlaa.png" width="480">
  <img src="video_aa_node.png" width="480">
</p>

# Features
Temporal Anti-Aliasing (TAA): Stabilizes flickering in video sequences using frame-history.
DLAA (Deep Learning AA): Uses a targeted neural block to clean up remaining aliasing without losing texture.
Ghosting-Free Logic: Advanced history clamping ensures fast-moving objects don't leave "ghost trails."
Halton Sequence Jittering: Superior sub-pixel sampling distribution for cleaner edges.
Edge-Aware Weighting: Intelligent blending to prevent blurring of fine details.

# Compare
<p align="center" style="display: flex; justify-content: center; gap: 10px;">
  <img src="Compare2.png" width="640">
  <img src="Compare3.png" width="640">
  <img src="Compare4.png" width="640">
</p>

# Install Node
Recommended Install via ComfyUI-Manager.</br>
Manual Clone Repository:
code
Bash
cd ComfyUI/custom_nodes/
git clone https://github.com/MSXYZ-GenAI/comfyui-msxyz.git

# Parameters
<p>TAA Parameters</p>
taa_strength
Recommended: 0.9 | Fast motion: 0.7 | Static scene: 0.95

taa_history_alpha
Recommended: 0.1 | Aggressive AA: 0.3 | Ghost-free: 0.05

jitter_scale
Recommended: 0.5 | Off: 0.0 | Maximum: 1.0

<p>DLAA Parameters</p>
Recommended: 0.7 | Pure AA, no sharpening: 0.0 | Aggressive sharpness: 1.0

Edge AA Parameters
Recommended: 0.08 | Only hard edges: 0.2 | Almost everything: 0.02

blur_radius
Recommended: 1 | Softer: 2–3 | Very soft: 4–5

Control
Normal: False | On scene change: True (for 1 frame)

<p>Safe starting preset:</p>
taa_strength      = 0.9
taa_history_alpha = 0.1
jitter_scale      = 0.5
dlaa_strength     = 0.7
edge_threshold    = 0.08
blur_radius       = 1
reset_history     = False

# Pro Tips
Ghosting Issues: If you see trailing shadows behind moving objects, slightly increase the edge_threshold or decrease taa_strength.
For Animation: Use a lower jitter_scale (0.5 - 1.0) for cleaner, more stable motion lines.
Upscale Workflow: Place this node immediately after your Upscale node. The jittering will help integrate the new pixel data created by the upscaler.
Batching: If you experience "Out of Memory" errors on high-resolution renders, set your batch_size to 1 or 2.

# Changelog
v0.1.1: Added History Clamping (Ghosting fix), Halton Sequence jittering, and improved edge-aware blending logic.
v0.1.0: Initial release of the adaptive anti-aliasing node.