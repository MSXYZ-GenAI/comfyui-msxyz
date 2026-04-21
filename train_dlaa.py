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

        self.extract_feature = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.refiner = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 16, 3, padding=1, bias=False),
        )
        self.reconstructor = nn.Conv2d(16, 3, 3, padding=1, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.extract_feature.weight)

        for m in self.refiner.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

        nn.init.dirac_(self.reconstructor.weight)

    def forward(self, x):
        features = F.leaky_relu(self.extract_feature(x), 0.2)
        refined = self.refiner(features)

        residual = self.reconstructor(refined)
        return x + residual


# Dataset
class DLAADataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = glob.glob(os.path.join(image_dir, '*.*'))

        self.target_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

        self.input_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),

            transforms.RandomApply([
                transforms.GaussianBlur(3, sigma=(0.1, 1.5))
            ], p=0.5),

            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')

        target_img = self.target_transform(img)
        input_img = self.input_transform(img)

        return input_img, target_img

# Epoch
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training is starting... Device used: {device}")

    model = _DLAANet().to(device)
    
    dataset = DLAADataset("dataset")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

    epochs = 100

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            if epoch == 0 and batch_idx == 0:
                print("CUDA:", torch.cuda.is_available())
                print("Device:", device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "dlaanet.pth")
    print("Excellent! 'dlaanet.pth' has been successfully created and saved.")


if __name__ == "__main__":
    train()