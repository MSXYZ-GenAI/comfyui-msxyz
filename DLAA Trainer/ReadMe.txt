You can use this tutorial to train a new DLAA model from scratch.

First, load your images into the dataset folder. (5,000-10,000 HD images)
Run Clean_Dataset.bat to filter the images. This will move low-quality images to the 'reject' folder.

Run a smoke test first. If everything goes well, proceed with the actual training by running Trainer_DLAA.bat or Trainer_Texture.bat.

DLAANet.safetensors — main anti-aliasing refinement model for cleaner edges.
DLAATexture.safetensors — optional texture refinement model for fine detail and micro-texture.

The both model files will appear in the 'runs' folder. 
If the process stops midway, you can resume it. The best checkpoints are saved automatically. 

The final output files will be DLAANet.safetensor and DLAATexture.safetensors.