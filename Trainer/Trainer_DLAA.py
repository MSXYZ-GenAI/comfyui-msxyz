# DLAANet Model Trainer
# Trains the 1x DLAA refinement model used by comfyui-msxyz.
# Exports DLAANet.pth and DLAANet.safetensors.


import argparse
import io
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image


try:
    from torchvision.models import vgg16, VGG16_Weights
    HAS_VGG = True
except Exception:
    HAS_VGG = False

try:
    from torch.amp import GradScaler, autocast
except Exception:
    from torch.cuda.amp import GradScaler, autocast


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


class DLAANet(nn.Module):
    def __init__(self, base_channels=192):
        super().__init__()

        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            # Stored in the model weights for use during inference (TAA jittering).
            # Not used during the training forward pass.
            "jitter_offsets",
            torch.tensor([
                [0.0000, -0.1667],
                [-0.2500, 0.1667],
                [0.2500, -0.3889],
                [-0.3750, -0.0556],
                [0.1250, 0.2778],
                [-0.1250, -0.2778],
                [0.3750, 0.0556],
                [-0.4375, 0.3889],
                [0.0625, -0.4630],
                [-0.3125, 0.1296],
                [0.1875, -0.2963],
                [-0.4375, -0.0370],
                [0.3125, 0.2407],
                [-0.0625, -0.2037],
                [0.4375, 0.0185],
                [-0.4688, 0.3148],
            ], dtype=torch.float32),
        )

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.reconstructor = nn.Conv2d(64, 3, 3, padding=1, bias=False)
        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="leaky_relu")
            elif isinstance(layer, nn.GroupNorm):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
        nn.init.zeros_(self.reconstructor.weight)

    def forward(self, x):
        feat_low = self.enc1(x)
        feat_high = self.enc2(feat_low)
        context = self.bottleneck(feat_high)
        decoded = self.dec(torch.cat([context, feat_low], dim=1))
        residual = self.reconstructor(decoded)
        return torch.clamp(x + residual, 0.0, 1.0)


class TargetOnlyDataset(Dataset):
    def __init__(self, root, split="train", patch_size=192, augment=True, val_ratio=0.10, seed=42):
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.augment = augment and split == "train"

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.root}")

        files = sorted([p for p in self.root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        if not files:
            raise RuntimeError(f"No images found in: {self.root}")

        rng = random.Random(seed)
        rng.shuffle(files)

        val_count = max(1, int(len(files) * val_ratio)) if len(files) > 1 else 0
        val_files = files[:val_count]
        train_files = files[val_count:] if val_count > 0 else files

        self.files = val_files if split == "val" else train_files
        if not self.files:
            self.files = files

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def _load_rgb(self, path):
        return Image.open(path).convert("RGB")

    def _resize_if_needed(self, img):
        w, h = img.size
        ps = self.patch_size
        if w < ps or h < ps:
            scale = max(ps / max(w, 1), ps / max(h, 1))
            nw = max(ps, int(w * scale + 0.5))
            nh = max(ps, int(h * scale + 0.5))
            img = img.resize((nw, nh), Image.BICUBIC)
        return img

    def _random_crop(self, img):
        img = self._resize_if_needed(img)
        w, h = img.size
        ps = self.patch_size
        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)
        return img.crop((x, y, x + ps, y + ps))

    def _center_crop(self, img):
        img = self._resize_if_needed(img)
        w, h = img.size
        ps = self.patch_size
        x = max(0, (w - ps) // 2)
        y = max(0, (h - ps) // 2)
        return img.crop((x, y, x + ps, y + ps))

    def __getitem__(self, idx):
        path = self.files[idx]
        target = self._load_rgb(path)

        if self.split == "train":
            target = self._random_crop(target)
        else:
            target = self._center_crop(target)

        if self.augment:
            if random.random() < 0.5:
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                target = target.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() < 0.25:
                target = target.rotate(random.choice([90, 180, 270]), expand=True)
                target = self._random_crop(target)

        target_t = self.to_tensor(target)
        input_a = degrade_input(target_t.unsqueeze(0), strong=True).squeeze(0)
        input_b = degrade_input(target_t.unsqueeze(0), strong=False).squeeze(0)

        return input_a, input_b, target_t, path.name


def _jpeg_roundtrip_tensor(x, quality):
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    out = []
    for img in x:
        pil = to_pil(img.detach().cpu().clamp(0, 1))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out.append(to_tensor(Image.open(buf).convert("RGB")))
    return torch.stack(out, dim=0).to(x.device)


def degrade_input(x, strong=True):
    b, c, h, w = x.shape
    device = x.device

    if strong:
        scale = random.choices(
            [0.50, 0.58, 0.625, 0.75, 0.875],
            weights=[0.22, 0.20, 0.22, 0.24, 0.12],
            k=1
        )[0]
    else:
        scale = random.choices(
            [0.75, 0.875, 0.95],
            weights=[0.35, 0.40, 0.25],
            k=1
        )[0]

    nh = max(16, int(h * scale))
    nw = max(16, int(w * scale))

    if strong:
        down_mode = random.choices(
            ["nearest", "bilinear", "bicubic"],
            weights=[0.45, 0.35, 0.20],
            k=1
        )[0]
        up_mode = random.choices(
            ["nearest", "bilinear", "bicubic"],
            weights=[0.30, 0.35, 0.35],
            k=1
        )[0]
    else:
        down_mode = random.choices(
            ["nearest", "bilinear", "bicubic"],
            weights=[0.20, 0.45, 0.35],
            k=1
        )[0]
        up_mode = random.choices(
            ["nearest", "bilinear", "bicubic"],
            weights=[0.15, 0.45, 0.40],
            k=1
        )[0]

    if down_mode == "nearest":
        y = F.interpolate(x, size=(nh, nw), mode="nearest")
    else:
        y = F.interpolate(x, size=(nh, nw), mode=down_mode, align_corners=False)

    if up_mode == "nearest":
        y = F.interpolate(y, size=(h, w), mode="nearest")
    else:
        y = F.interpolate(y, size=(h, w), mode=up_mode, align_corners=False)

    shift_prob = 0.80 if strong else 0.60
    shift_amount = 0.70 if strong else 0.35

    if random.random() < shift_prob:
        shift_x = random.uniform(-shift_amount, shift_amount)
        shift_y = random.uniform(-shift_amount, shift_amount)

        theta = torch.eye(2, 3, device=device).unsqueeze(0).repeat(b, 1, 1)
        theta[:, 0, 2] = shift_x / max(w, 1)
        theta[:, 1, 2] = shift_y / max(h, 1)

        grid = F.affine_grid(theta, y.shape, align_corners=False)
        y = F.grid_sample(
            y,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=False
        )

    checker_prob = 0.15 if strong else 0.06

    if random.random() < checker_prob:
        checker = torch.ones_like(y)
        checker[:, :, ::2, 1::2] *= random.uniform(0.80, 0.95)
        checker[:, :, 1::2, ::2] *= random.uniform(0.80, 0.95)

        y = torch.clamp(y * checker, 0.0, 1.0)
        y = F.avg_pool2d(
            F.pad(y, (1, 1, 1, 1), mode="reflect"),
            kernel_size=3,
            stride=1
        )

    sharpen_prob = 0.32 if strong else 0.20
    sharpen_strength = (0.18, 0.40) if strong else (0.10, 0.22)

    if random.random() < sharpen_prob:
        blur = F.avg_pool2d(
            F.pad(y, (1, 1, 1, 1), mode="reflect"),
            kernel_size=3,
            stride=1
        )
        detail = y - blur
        y = torch.clamp(y + detail * random.uniform(*sharpen_strength), 0.0, 1.0)

    noise_prob = 0.10 if strong else 0.05
    noise_sigma = (0.0015, 0.0050) if strong else (0.0010, 0.0030)

    if random.random() < noise_prob:
        noise = torch.randn_like(y) * random.uniform(*noise_sigma)
        y = torch.clamp(y + noise, 0.0, 1.0)

    quant_prob = 0.10 if strong else 0.06

    if random.random() < quant_prob:
        levels = random.choice([128, 160, 192, 224])
        y = torch.round(y * levels) / levels

    jpeg_prob = 0.10 if strong else 0.06
    jpeg_quality = (65, 88) if strong else (75, 92)

    if random.random() < jpeg_prob:
        quality = random.randint(*jpeg_quality)
        y = _jpeg_roundtrip_tensor(y, quality)

    return torch.clamp(y, 0.0, 1.0)


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        if not HAS_VGG:
            self.vgg = None
            return
        try:
            vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:8].to(device).eval()
        except Exception:
            self.vgg = None
            return
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def forward(self, pred, target):
        if self.vgg is None:
            return pred.new_tensor(0.0)
        pred_n = (pred - self.mean) / self.std
        tgt_n = (target - self.mean) / self.std
        return F.l1_loss(self.vgg(pred_n), self.vgg(tgt_n))


class LossPack(nn.Module):
    def __init__(self, device, use_perceptual=True):
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sx", sx)
        self.register_buffer("sy", sy)
        self.register_buffer("lap", lap)
        self.perceptual = PerceptualLoss(device) if use_perceptual else None

    def luma(self, x):
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def edge_map(self, x):
        y = self.luma(x)
        ex = F.conv2d(y, self.sx, padding=1)
        ey = F.conv2d(y, self.sy, padding=1)
        return torch.sqrt(ex * ex + ey * ey + 1e-6)

    def lap_map(self, x):
        return F.conv2d(self.luma(x), self.lap, padding=1)

    def charbonnier(self, pred, target, eps=1e-3):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + eps * eps))

    def frequency_loss(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        tgt_fft = torch.fft.rfft2(target, norm="ortho")
        return F.l1_loss(torch.log1p(torch.abs(pred_fft)), torch.log1p(torch.abs(tgt_fft)))

    def smooth_loss(self, x):
        dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return dx + dy

    def forward(self, pred_a, pred_b, target, args):
        l1 = F.l1_loss(pred_a, target)
        edge = F.l1_loss(self.edge_map(pred_a), self.edge_map(target))
        lap = F.l1_loss(self.lap_map(pred_a), self.lap_map(target))
        freq = self.frequency_loss(pred_a, target)
        charb = self.charbonnier(pred_a, target)
        smooth = self.smooth_loss(pred_a)
        temporal = F.l1_loss(pred_a, pred_b)

        if self.perceptual is not None:
            perceptual = self.perceptual(pred_a, target)
        else:
            perceptual = pred_a.new_tensor(0.0)

        total = (
            args.w_l1 * l1 +
            args.w_edge * edge +
            args.w_lap * lap +
            args.w_freq * freq +
            args.w_charb * charb +
            args.w_smooth * smooth +
            args.w_temporal * temporal +
            args.w_perceptual * perceptual
        )

        parts = {
            "total": float(total.detach().cpu()),
            "l1": float(l1.detach().cpu()),
            "edge": float(edge.detach().cpu()),
            "lap": float(lap.detach().cpu()),
            "freq": float(freq.detach().cpu()),
            "charb": float(charb.detach().cpu()),
            "smooth": float(smooth.detach().cpu()),
            "temporal": float(temporal.detach().cpu()),
            "perceptual": float(perceptual.detach().cpu()),
        }
        return total, parts


def make_fp16_state_dict(state_dict):
    packed = {}
    for k, v in state_dict.items():
        if torch.is_floating_point(v):
            packed[k] = v.detach().cpu().half()
        else:
            packed[k] = v.detach().cpu()
    return packed


@torch.no_grad()
def validate(model, loader, loss_fn, device, args, max_batches=None):
    model.eval()
    total = 0.0
    count = 0
    for bi, (inp_a, inp_b, tgt, _) in enumerate(loader):
        inp_a = inp_a.to(device, non_blocking=True)
        inp_b = inp_b.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        pred_a = model(inp_a)
        pred_b = model(inp_b)
        loss, _ = loss_fn(pred_a, pred_b, tgt, args)
        total += float(loss.detach().cpu())
        count += 1
        if max_batches is not None and bi + 1 >= max_batches:
            break
    model.train()
    return total / max(count, 1)


@torch.no_grad()
def save_preview(model, loader, device, out_dir, step, max_items=4):
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inp_a, inp_b, tgt, _ = next(iter(loader))
    inp_a = inp_a.to(device)[:max_items]
    inp_b = inp_b.to(device)[:max_items]
    tgt = tgt.to(device)[:max_items]
    pred = model(inp_a)
    grid = torch.cat([inp_a, inp_b, pred, tgt], dim=0)
    save_image(grid, out_dir / f"preview_{step:06d}.png", nrow=max_items)
    model.train()


def model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def save_checkpoint(model, optimizer, scheduler, epoch, step, val_loss, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    pth_path = out_dir / f"{name}.pth"
    torch.save(payload, pth_path)
    save_file(model.state_dict(), str(out_dir / f"{name}.safetensors"))
    return model_size_mb(pth_path)


def save_best_snapshot(model, optimizer, scheduler, epoch, step, val_loss, out_dir, export_fp16=True):
    archive_dir = Path(out_dir) / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    clean_val = f"{val_loss:.6f}".replace(".", "_")
    name = f"best_epoch_{epoch:03d}_val_{clean_val}"
    payload = {
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(payload, archive_dir / f"{name}.pth")
    state = make_fp16_state_dict(model.state_dict()) if export_fp16 else model.state_dict()
    save_file(state, str(archive_dir / f"{name}.safetensors"))
    return name


def save_inference_weights(model, out_dir, name="DLAANet", export_fp16=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = make_fp16_state_dict(model.state_dict()) if export_fp16 else model.state_dict()

    pth_path = out_dir / f"{name}.pth"
    safe_path = out_dir / f"{name}.safetensors"

    torch.save(state, pth_path)
    save_file(state, str(safe_path))

    return model_size_mb(pth_path), model_size_mb(safe_path)


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    preview_dir = out_dir / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    train_set = TargetOnlyDataset(args.dataset, "train", args.patch_size, augment=True, val_ratio=args.val_ratio, seed=args.seed)
    val_set = TargetOnlyDataset(args.dataset, "val", args.patch_size, augment=False, val_ratio=args.val_ratio, seed=args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.workers // 2),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = DLAANet(base_channels=args.base_channels).to(device)
    loss_fn = LossPack(device, use_perceptual=not args.no_perceptual).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    best_val = math.inf
    step = 0
    start_epoch = 1
    log_data = []
    log_path = out_dir / "training_log.json"

    if log_path.exists():
        try:
            log_data = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            log_data = []

    if args.resume is not None:
        resume_path = Path(args.resume)

        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=device)

        model.load_state_dict(checkpoint["model"], strict=False)

        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        step = int(checkpoint.get("step", 0))
        best_val = float(checkpoint.get("val_loss", math.inf))

        best_path = ckpt_dir / "best.pth"
        if best_path.exists():
            try:
                best_checkpoint = torch.load(best_path, map_location="cpu")
                best_val = float(best_checkpoint.get("val_loss", best_val))
            except Exception:
                pass

        print("==============================================================")
        print(f"Resuming from : {resume_path}")
        print(f"Start epoch   : {start_epoch}")
        print(f"Step          : {step}")
        print(f"Best val      : {best_val:.6f}")
        print("==============================================================")

    print("==============================================================")
    print("DLAA Trainer")
    print("==============================================================")
    print(f"Device      : {device}")
    print(f"Dataset     : {len(train_set)} train | {len(val_set)} val")
    print(f"Patch       : {args.patch_size}")
    print(f"Batch       : {args.batch_size}")
    print(f"Epochs      : {args.epochs}")
    print(f"Parameters  : {n_params / 1e6:.2f}M")
    print(f"Export      : {'FP16' if args.export_fp16 else 'FP32'}")
    print("==============================================================")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for inp_a, inp_b, tgt, _ in train_loader:
            inp_a = inp_a.to(device, non_blocking=True)
            inp_b = inp_b.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=args.amp and device.type == "cuda"):
                pred_a = model(inp_a)
                pred_b = model(inp_b)
                loss, parts = loss_fn(pred_a, pred_b, tgt, args)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach().cpu())
            step += 1

            if step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[E{epoch:03d} S{step:06d}] loss={parts['total']:.6f} "
                    f"l1={parts['l1']:.5f} edge={parts['edge']:.5f} "
                    f"freq={parts['freq']:.5f} temp={parts['temporal']:.5f} lr={lr:.2e}"
                )

        scheduler.step()

        train_loss = epoch_loss / max(len(train_loader), 1)
        val_loss = validate(model, val_loader, loss_fn, device, args, max_batches=args.val_batches)

        if epoch % args.preview_every == 0 or epoch == 1:
            save_preview(model, val_loader, device, preview_dir, step)

        last_size = save_checkpoint(model, optimizer, scheduler, epoch, step, val_loss, ckpt_dir, "last")

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, step, val_loss, ckpt_dir, "best")
            snapshot_name = save_best_snapshot(model, optimizer, scheduler, epoch, step, val_loss, ckpt_dir, export_fp16=args.export_fp16)
            
            pth_size, safe_size = save_inference_weights(
                model,
                ckpt_dir,
                "DLAANet",
                export_fp16=args.export_fp16
            )
            
            if pth_size > args.max_model_mb:
                print(f"[WARN] DLAANet.pth is {pth_size:.2f} MB, above limit {args.max_model_mb:.2f} MB")

            if safe_size > args.max_model_mb:
                print(f"[WARN] DLAANet.safetensors is {safe_size:.2f} MB, above limit {args.max_model_mb:.2f} MB")

            print(
                f"[BEST] val={best_val:.6f} | "
                f"DLAANet.pth={pth_size:.2f} MB | "
                f"DLAANet.safetensors={safe_size:.2f} MB | "
                f"archived={snapshot_name}"
            )

        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val": best_val,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
            "last_checkpoint_mb": last_size,
        }
        log_data.append(row)
        log_path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")

        best_mark = " ★" if is_best else ""
        print(
            f"[E{epoch:03d}] train={train_loss:.6f} "
            f"val={val_loss:.6f} best={best_val:.6f} "
            f"time={elapsed:.1f}s{best_mark}"
        )

    print("Training complete.")
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Output: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="DLAA Trainer")
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=192)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--preview-every", type=int, default=1)
    parser.add_argument("--val-batches", type=int, default=None)
    parser.add_argument("--max-model-mb", type=float, default=9.0)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-fp16", action="store_true", default=True)
    parser.add_argument("--no-perceptual", action="store_true")
    parser.add_argument("--out-dir", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--w-l1", type=float, default=1.00)
    parser.add_argument("--w-edge", type=float, default=0.50)
    parser.add_argument("--w-lap", type=float, default=0.30)
    parser.add_argument("--w-freq", type=float, default=0.22)
    parser.add_argument("--w-charb", type=float, default=0.12)
    parser.add_argument("--w-smooth", type=float, default=0.01)
    parser.add_argument("--w-temporal", type=float, default=0.18)
    parser.add_argument("--w-perceptual", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
