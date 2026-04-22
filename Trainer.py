# Created by MSXYZ (AI-assisted)
# DLAA Model Trainer — v0.1.1
# Perceptual Loss, Luma Loss, Smooth Loss, JPEG Augmentation

# Steps:
# 1. Create a folder named "dataset" next to this script.
# 2. Add 50–200 high-quality images (landscapes, portraits, geometric patterns).
# 3. Run: python Trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image, ImageFilter
import random
import io
import os
import glob


# Model
class DLAANet(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer("sobel_x",
            torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer("sobel_y",
            torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer("jitter_offsets",
            torch.tensor([[0.25,0.25],[-0.25,-0.25],[-0.25,0.25],[0.25,-0.25]], dtype=torch.float32))

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.reconstructor = nn.Conv2d(32, 3, 3, padding=1, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.zeros_(self.reconstructor.weight)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b  = self.bottleneck(e2)
        d  = self.dec(torch.cat([b, e1], dim=1))
        r  = self.reconstructor(d)
        return x + r


# Perceptual Loss
class _PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg  = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.l1  = nn.L1Loss()

        # VGG normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        p = self.vgg(self._normalize(pred))
        t = self.vgg(self._normalize(target))
        return self.l1(p, t)


# Combined Loss
class DLAALoss(nn.Module):
    def __init__(self, device,
                 w_pixel=1.0, w_edge=0.5, w_freq=0.1,
                 w_luma=0.5,  w_smooth=0.3, w_perceptual=0.2):
        super().__init__()
        self.l1          = nn.L1Loss()
        self.perceptual  = _PerceptualLoss(device)
        self.w_pixel     = w_pixel
        self.w_edge      = w_edge
        self.w_freq      = w_freq
        self.w_luma      = w_luma
        self.w_smooth    = w_smooth
        self.w_perceptual = w_perceptual

        sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self._sx = sx
        self._sy = sy

    def _luma(self, x):
        return 0.2126*x[:,0:1] + 0.7152*x[:,1:2] + 0.0722*x[:,2:3]

    def _edge_map(self, x):
        sx = self._sx.to(x.device)
        sy = self._sy.to(x.device)
        gray = self._luma(x)
        ex   = F.conv2d(gray, sx, padding=1)
        ey   = F.conv2d(gray, sy, padding=1)
        return torch.sqrt(ex**2 + ey**2 + 1e-6)

    def _freq(self, x):
        return torch.abs(torch.fft.fft2(self._luma(x)))

    def forward(self, pred, target):
        # Pixel
        pixel = self.l1(pred, target)

        # Edge
        edge  = self.l1(self._edge_map(pred), self._edge_map(target))

        # Frequency
        freq  = self.l1(self._freq(pred), self._freq(target))

        # Luma
        luma  = self.l1(self._luma(pred), self._luma(target))

        # Smooth
        edge_map    = self._edge_map(target).detach()
        smooth_mask = 1.0 - torch.sigmoid((edge_map - 0.1) * 20.0)
        smooth      = self.l1(pred * smooth_mask, target * smooth_mask)

        # Perceptual
        perceptual  = self.perceptual(pred, target)

        total = (
            self.w_pixel      * pixel      +
            self.w_edge       * edge       +
            self.w_freq       * freq       +
            self.w_luma       * luma       +
            self.w_smooth     * smooth     +
            self.w_perceptual * perceptual
        )
        return total, {
            "pixel": pixel.item(),
            "edge":  edge.item(),
            "luma":  luma.item(),
            "smooth": smooth.item(),
            "perceptual": perceptual.item(),
        }


# Dataset
class DLAADataset(Dataset):
    SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(self, image_dir: str, size: int = 512):
        self.image_paths = [
            p for p in glob.glob(os.path.join(image_dir, "*.*"))
            if os.path.splitext(p)[1].lower() in self.SUPPORTED
        ]
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in '{image_dir}'.")

        self.size     = size
        self.to_tensor = transforms.ToTensor()
        self.clean_tf  = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS),
        ])

    def __len__(self):
        return len(self.image_paths)

    def _alias_nearest(self, img, factor):
        s = int(self.size * factor)
        return img.resize((s, s), Image.NEAREST).resize((self.size, self.size), Image.NEAREST)

    def _alias_bilinear_blur(self, img, factor):
        s = int(self.size * factor)
        img = img.resize((s, s), Image.BILINEAR).resize((self.size, self.size), Image.BILINEAR)
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.9)))

    def _alias_checkerboard(self, img):
        t = self.to_tensor(img)
        t[:, ::2, 1::2] = 0
        t[:, 1::2, ::2] = 0
        t = F.avg_pool2d(t.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
        return transforms.ToPILImage()(t.clamp(0, 1))

    def _alias_mixed(self, img, factor):
        s = int(self.size * factor)
        img = img.resize((s, s), Image.NEAREST).resize((self.size, self.size), Image.BILINEAR)
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

    def _alias_jpeg(self, img, quality):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.size, self.size), (128, 128, 128))

        clean  = self.clean_tf(img)
        mode   = random.randint(0, 4)
        factor = random.uniform(0.25, 0.6)

        if   mode == 0: aliased = self._alias_nearest(clean, factor)
        elif mode == 1: aliased = self._alias_bilinear_blur(clean, factor)
        elif mode == 2: aliased = self._alias_checkerboard(clean)
        elif mode == 3: aliased = self._alias_mixed(clean, factor)
        else:           aliased = self._alias_jpeg(clean, random.randint(40, 75))

        return self.to_tensor(aliased), self.to_tensor(clean)


# Early Stopping
class EarlyStopping:
    def __init__(self, patience=12, min_delta=1e-5):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.counter   = 0

    def step(self, loss):
        if loss < self.best - self.min_delta:
            self.best    = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# Training
def train(
    dataset_dir    = "dataset",
    output_path    = "DLAANet.pth",
    image_size     = 512,
    batch_size     = 4,
    epochs         = 100,
    lr             = 3e-4,
    patience       = 12,
    # loss weights
    w_pixel        = 1.0,
    w_edge         = 0.5,
    w_freq         = 0.1,
    w_luma         = 0.5,
    w_smooth       = 0.3,
    w_perceptual   = 0.2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DLAA] Device       : {device}")

    dataset    = DLAADataset(dataset_dir, size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=(device.type == "cuda"))
    print(f"[DLAA] Dataset      : {len(dataset)} images | {len(dataloader)} batches/epoch")

    model    = DLAANet().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[DLAA] Parameters   : {n_params:,}")

    criterion = DLAALoss(device,
                         w_pixel=w_pixel, w_edge=w_edge, w_freq=w_freq,
                         w_luma=w_luma,   w_smooth=w_smooth, w_perceptual=w_perceptual)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-6)
    stopper   = EarlyStopping(patience=patience)

    best_loss  = float("inf")
    best_state = None

    print(f"\n{'='*62}")
    print(f"  DLAA Training  v0.4.0  |  {epochs} epochs  |  LR {lr:.0e}")
    print(f"  Loss: pixel={w_pixel} edge={w_edge} freq={w_freq} luma={w_luma} smooth={w_smooth} perceptual={w_perceptual}")
    print(f"{'='*62}")

    for epoch in range(epochs):
        model.train()
        total_loss   = 0.0
        sub_totals   = {"pixel": 0., "edge": 0., "luma": 0., "smooth": 0., "perceptual": 0.}

        for inputs, targets in dataloader:
            inputs  = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs        = torch.clamp(model(inputs), 0.0, 1.0)
            loss, sub_dict = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            for k in sub_totals:
                sub_totals[k] += sub_dict[k]

        scheduler.step()

        n      = len(dataloader)
        avg    = total_loss / n
        lr_now = scheduler.get_last_lr()[0]
        marker = " ★" if avg < best_loss else ""

        # Detailed loss breakdown every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            breakdown = "  |  " + "  ".join(
                f"{k}:{sub_totals[k]/n:.4f}" for k in sub_totals
            )
        else:
            breakdown = ""

        print(f"Epoch [{epoch+1:>3}/{epochs}]  Loss: {avg:.5f}  LR: {lr_now:.2e}{marker}{breakdown}")

        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if stopper.step(avg):
            print(f"\n[DLAA] Early stop at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    save_state = best_state if best_state else model.state_dict()
    torch.save(save_state, output_path)
    print(f"\n[DLAA] ✓ Best model saved → {output_path}  (loss: {best_loss:.5f})")
    print(f"[DLAA] Replace DLAANet.pth in your ComfyUI node folder.")


if __name__ == "__main__":
    print("=" * 62)
    print("         DLAA Model Training  v0.4.0")
    print("=" * 62)

    dataset_dir = "dataset"
    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] '{dataset_dir}' folder not found.")
        print("Create a 'dataset' folder and add images, then re-run.")
        exit(1)

    n_imgs = len([
        f for f in glob.glob(os.path.join(dataset_dir, "*.*"))
        if os.path.splitext(f)[1].lower() in DLAADataset.SUPPORTED
    ])
    print(f"[INFO] Checking the dataset...")
    if n_imgs < 20:
        print(f"[WARN] Only {n_imgs} images found. 50+ recommended.")
    else:
        print(f"[INFO] {n_imgs} images found. Starting training...")

    train(
        dataset_dir  = "dataset",
        output_path  = "DLAANet.pth",
        image_size   = 512,
        batch_size   = 4,       # VRAM'a göre artır: 8, 16...
        epochs       = 100,
        lr           = 3e-4,
        patience     = 12,
        w_pixel      = 1.0,
        w_edge       = 0.5,
        w_freq       = 0.1,
        w_luma       = 0.5,
        w_smooth     = 0.3,
        w_perceptual = 0.2,
    )