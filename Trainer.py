# Created by MSXYZ
# Trainer — v0.1.1
 
# Steps:
# 1. Create a folder named "dataset" next to this script.
# 2. Add 50–100 high-quality images (landscapes, portraits, geometric patterns).
# 3. Run: python Trainer.py
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
 
 
# Do NOT change this architecture
class _DLAANet(nn.Module):
    def __init__(self):
        super().__init__()
 
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1, -2, -1],
                          [ 0,  0,  0],
                          [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
        self.register_buffer(
            "jitter_offsets",
            torch.tensor([
                [ 0.25,  0.25],
                [-0.25, -0.25],
                [-0.25,  0.25],
                [ 0.25, -0.25],
            ])
        )
 
        self.extract_feature = nn.Conv2d(3, 32, 3, padding=1, bias=False)
 
        self.refiner = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2),
 
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2),
 
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
        )
 
        self.reconstructor = nn.Conv2d(32, 3, 3, padding=1, bias=False)
 
        self._init_weights()
 
    def _init_weights(self):
        # orthogonal
        nn.init.orthogonal_(self.extract_feature.weight)
 
        # kaiming
        for m in self.refiner.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
 
        # zero-init
        nn.init.zeros_(self.reconstructor.weight)
 
    def forward(self, x):
        f = F.leaky_relu(self.extract_feature(x), 0.2)
        f = self.refiner(f)
        r = self.reconstructor(f)
        return x + r


# Edge-Aware
class EdgeAwareLoss(nn.Module):

    def __init__(self, edge_weight: float = 0.3):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.edge_weight = edge_weight
 
        sobel_x = torch.tensor([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
        self._sobel_x = sobel_x
        self._sobel_y = sobel_y
 
    def _edge_map(self, x: torch.Tensor) -> torch.Tensor:
        sx = self._sobel_x.to(x.device)
        sy = self._sobel_y.to(x.device)
        gray = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        ex = F.conv2d(gray, sx, padding=1)
        ey = F.conv2d(gray, sy, padding=1)
        return torch.sqrt(ex ** 2 + ey ** 2 + 1e-6)
 
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pixel_loss = self.l1(pred, target)
        edge_loss  = self.l1(self._edge_map(pred), self._edge_map(target))
        return pixel_loss + self.edge_weight * edge_loss
 

# Dataset
class DLAADataset(Dataset):
    SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
 
    def __init__(self, image_dir: str, size: int = 512):
        self.image_paths = [
            p for p in glob.glob(os.path.join(image_dir, "*.*"))
            if os.path.splitext(p)[1].lower() in self.SUPPORTED
        ]
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in '{image_dir}'.")
 
        alias_size = int(size * 0.75)
 
        self.target_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
 
        self.input_transform = transforms.Compose([
            transforms.Resize((size, size),       interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Resize((alias_size, alias_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.Resize((size, size),       interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
 
    def __len__(self) -> int:
        return len(self.image_paths)
 
    def __getitem__(self, idx: int):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception as e:
            print(f"[WARN] Could not open {self.image_paths[idx]}: {e}")
            img = Image.new("RGB", (512, 512), (128, 128, 128))
 
        return self.input_transform(img), self.target_transform(img)
 

# Early Stopping
class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-5):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter   = 0
 
    def step(self, loss: float) -> bool:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter   = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
 

# Training
def train(
    dataset_dir : str   = "dataset",
    output_path : str   = "DLAANet.pth",
    image_size  : int   = 512,
    batch_size  : int   = 2,
    epochs      : int   = 60,
    lr          : float = 3e-4,
    edge_weight : float = 0.3,
    patience    : int   = 8,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DLAA] Device: {device}")
 
    dataset    = DLAADataset(dataset_dir, size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )
    print(f"[DLAA] Dataset: {len(dataset)} images, {len(dataloader)} batches/epoch")
 
    # Model
    model     = _DLAANet().to(device)
    criterion = EdgeAwareLoss(edge_weight=edge_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    stopper   = EarlyStopping(patience=patience)
 
    best_loss  = float("inf")
    best_state = None
 
    # Loop 
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
 
        for inputs, targets in dataloader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
 
            optimizer.zero_grad()
            outputs = torch.clamp(model(inputs), 0.0, 1.0)
            loss    = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 
            optimizer.step()
            total_loss += loss.item()
 
        scheduler.step()
 
        avg_loss = total_loss / len(dataloader)
        lr_now   = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1:>3}/{epochs}]  Loss: {avg_loss:.5f}  LR: {lr_now:.2e}")
 
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
 
        # Early stopping
        if stopper.step(avg_loss):
            print(f"[DLAA] Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
 
    # Save 
    if best_state is not None:
        torch.save(best_state, output_path)
        print(f"\n[DLAA] ✓ Best model saved → {output_path}  (loss: {best_loss:.5f})")
    else:
        torch.save(model.state_dict(), output_path)
        print(f"\n[DLAA] ✓ Model saved → {output_path}")
 
 
if __name__ == "__main__":
    train(
        dataset_dir  = "dataset",
        output_path  = "DLAANet.pth",
        image_size   = 512,
        batch_size   = 2,       
        epochs       = 60, #PSNR: 40.48 dB
        lr           = 3e-4,
        edge_weight  = 0.3,
        patience     = 8,
    )