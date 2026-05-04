DLAA Model Training Guide

You can use these tools to train new DLAA models from scratch.

Recommended hardware:
NVIDIA RTX 3090 or better with 24 GB VRAM.

1. Prepare the dataset

Place your training images in the dataset folder.

Recommended dataset size:
5,000 - 10,000 HD images

Run Clean_Dataset.bat to filter the images.

The cleaner checks for weak, broken, low-detail, document-like, or duplicate images.
Rejected images are moved to the _rejected folder instead of being deleted.

2. Run a smoke test

Run the smoke test first before starting a full training run.

If everything works correctly, continue with the actual training.

3. Train the models

Run Trainer_DLAA.bat to train the main DLAA model.

Run Trainer_Texture.bat to train the optional texture refinement model.

4. Output files

DLAANet.safetensors:
Main anti-aliasing refinement model for cleaner edges.

DLAATexture.safetensors:
Optional texture refinement model for fine detail and micro-texture.

Both model files will be saved in the runs folder.

Best checkpoints are saved automatically during training.
If training stops midway, you can resume from the latest checkpoint.

Final output files:

DLAANet.safetensors
DLAATexture.safetensors