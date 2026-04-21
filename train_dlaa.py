# Created by MSXYZ

# How to train your DLAA model?
# I have included the `train_dlaa.py` script for those who want to fine-tune the model. 
# 1. Create a folder named `dataset` next to the script.
# 2. Put 50-100 high-quality images (landscapes, portraits, geometric images) inside it.
# 3. Open the terminal in hte folder Run "python train_dlaa.py" and wait for 100 or 500 epochs.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob

# Do not change this architecture (must be the same as in VideoTAADLAA.py)
class _DLAANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("sobel_x", torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("sobel_y", torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("jitter_offsets", torch.tensor([[0.25, 0.25], [-0.25, -0.25], [-0.25, 0.25], [0.25, -0.25]], dtype=torch.float32))

        self.block = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 16, 3, padding=1)
        )
        
        self.extract = nn.Conv2d(3, 16, 3, padding=1)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.extract_feature.weight)
        nn.init.orthogonal_(self.refiner.weight)
        nn.init.dirac_(self.reconstructor.weight) 

    def forward(self, x):
        features = F.leaky_relu(self.extract(x), 0.2)
        residual = self.block(features)
        return x + residual

# Dataset
class DLAADataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = glob.glob(os.path.join(image_dir, '*.*'))
        
        # transform
        self.target_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        
        self.input_transform = transforms.Compose([
            transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        target_img = self.target_transform(img)
        input_img = self.input_transform(img)
        return input_img, target_img

# 3. Epoch
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training is starting... Device used: {device}")

    # Model
    model = _DLAANet().to(device)
    
    # Dataset
    dataset = DLAADataset("dataset")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Optimization
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=4e-3)

    epochs = 100

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

    # Save
    torch.save(model.state_dict(), "dlaanet.pth")
    print("Excellent! 'dlaanet.pth' has been successfully created and saved.")

if __name__ == "__main__":
    train()