# Video Cinematic FX
# v0.1.0
# SOLRICKS


import math
import torch
import torch.nn.functional as F


try:
    import comfy.model_management as mm
except ImportError:
    mm = None


try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None


MIN_EFFECT_STRENGTH = 1e-5
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


CINEMATIC_PRESETS = {
    "Cinematic": {
        "temperature": 0.02,
        "tint": 0.00,
        "exposure": 0.00,
        "contrast": 0.18,
        "saturation": -0.02,
        "lift": -0.005,
        "bloom": 0.36,
        "bloom_threshold": 0.66,
        "halation": 0.14,
        "streak": 0.08,
        "streak_warmth": 0.25,
        "vignette": 0.34,
        "grain": 0.016,
        "chromatic": 0.18,
        "soften": 0.00,
        "bloom_color": (1.00, 0.93, 0.82),
    },
    "Luxury Ad": {
        "temperature": 0.06,
        "tint": 0.01,
        "exposure": 0.03,
        "contrast": 0.12,
        "saturation": 0.10,
        "lift": 0.000,
        "bloom": 0.55,
        "bloom_threshold": 0.60,
        "halation": 0.28,
        "streak": 0.30,
        "streak_warmth": 0.55,
        "vignette": 0.22,
        "grain": 0.004,
        "chromatic": 0.08,
        "soften": 0.03,
        "bloom_color": (1.00, 0.88, 0.68),
    },
    "Soft Beauty": {
        "temperature": 0.05,
        "tint": 0.02,
        "exposure": 0.02,
        "contrast": -0.04,
        "saturation": 0.04,
        "lift": 0.004,
        "bloom": 0.42,
        "bloom_threshold": 0.62,
        "halation": 0.18,
        "streak": 0.02,
        "streak_warmth": 0.40,
        "vignette": 0.14,
        "grain": 0.002,
        "chromatic": 0.00,
        "soften": 0.10,
        "bloom_color": (1.00, 0.90, 0.82),
    },
    "Neon Night": {
        "temperature": -0.06,
        "tint": 0.05,
        "exposure": -0.01,
        "contrast": 0.22,
        "saturation": 0.25,
        "lift": -0.010,
        "bloom": 0.78,
        "bloom_threshold": 0.48,
        "halation": 0.08,
        "streak": 0.58,
        "streak_warmth": -0.30,
        "vignette": 0.42,
        "grain": 0.012,
        "chromatic": 0.45,
        "soften": 0.00,
        "bloom_color": (0.70, 0.86, 1.00),
    },
    "Cool Tech": {
        "temperature": -0.08,
        "tint": -0.02,
        "exposure": 0.00,
        "contrast": 0.16,
        "saturation": 0.06,
        "lift": -0.004,
        "bloom": 0.38,
        "bloom_threshold": 0.58,
        "halation": 0.04,
        "streak": 0.24,
        "streak_warmth": -0.50,
        "vignette": 0.26,
        "grain": 0.004,
        "chromatic": 0.20,
        "soften": 0.00,
        "bloom_color": (0.70, 0.92, 1.00),
    },
    "Warm Film": {
        "temperature": 0.10,
        "tint": 0.00,
        "exposure": 0.00,
        "contrast": 0.08,
        "saturation": -0.04,
        "lift": 0.002,
        "bloom": 0.30,
        "bloom_threshold": 0.64,
        "halation": 0.36,
        "streak": 0.04,
        "streak_warmth": 0.70,
        "vignette": 0.30,
        "grain": 0.026,
        "chromatic": 0.08,
        "soften": 0.02,
        "bloom_color": (1.00, 0.78, 0.55),
    },
    "Dramatic Contrast": {
        "temperature": -0.01,
        "tint": 0.00,
        "exposure": -0.02,
        "contrast": 0.34,
        "saturation": -0.06,
        "lift": -0.018,
        "bloom": 0.22,
        "bloom_threshold": 0.70,
        "halation": 0.06,
        "streak": 0.06,
        "streak_warmth": 0.00,
        "vignette": 0.50,
        "grain": 0.014,
        "chromatic": 0.12,
        "soften": 0.00,
        "bloom_color": (0.95, 0.95, 1.00),
    },
    "Performance": {
        "temperature": 0.02,
        "tint": 0.00,
        "exposure": 0.00,
        "contrast": 0.10,
        "saturation": 0.02,
        "lift": -0.004,
        "bloom": 0.14,
        "bloom_threshold": 0.68,
        "halation": 0.04,
        "streak": 0.00,
        "streak_warmth": 0.00,
        "vignette": 0.18,
        "grain": 0.000,
        "chromatic": 0.00,
        "soften": 0.00,
        "bloom_color": (1.00, 0.95, 0.88),
    },
}


class VideoCinematicFX:
    def __init__(self):
        self._vignette_cache = {}
        self._grid_cache = {}
        self._horizontal_kernel_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "preset": ([
                    "Cinematic",
                    "Luxury Ad",
                    "Soft Beauty",
                    "Neon Night",
                    "Cool Tech",
                    "Warm Film",
                    "Dramatic Contrast",
                    "Performance",
                ],),
                "look_strength": (
                    "FLOAT",
                    {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.05},
                ),
                "glow_strength": (
                    "FLOAT",
                    {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.05},
                ),
                "contrast_strength": (
                    "FLOAT",
                    {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.05},
                ),
                "grain_strength": (
                    "FLOAT",
                    {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.05},
                ),
                "vignette_strength": (
                    "FLOAT",
                    {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "CustomPostProcess"

    def _get_device(self):
        if mm is not None:
            return mm.get_torch_device()
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _clamp_strength(value):
        return max(0.0, min(float(value), 2.0))

    @staticmethod
    def _rgb_luma(image):
        r, g, b = LUMA_WEIGHTS
        return r * image[:, 0:1] + g * image[:, 1:2] + b * image[:, 2:3]

    @staticmethod
    def _safe_odd_kernel(size, height, width, horizontal=False):
        size = int(size)
        if horizontal:
            max_size = max(1, int(width) - 1)
        else:
            max_size = max(1, min(int(height), int(width)) - 1)

        size = min(size, max_size)
        if size % 2 == 0:
            size -= 1
        return max(1, size)

    def _avg_blur(self, image, kernel_size):
        _, _, h, w = image.shape
        kernel_size = self._safe_odd_kernel(kernel_size, h, w)
        if kernel_size <= 1:
            return image
        pad = kernel_size // 2
        return F.avg_pool2d(
            F.pad(image, [pad, pad, pad, pad], mode="reflect"),
            kernel_size,
            stride=1,
        )

    def _horizontal_blur(self, image, kernel_size):
        b, c, h, w = image.shape
        kernel_size = self._safe_odd_kernel(kernel_size, h, w, horizontal=True)
        if kernel_size <= 1:
            return image

        cache_key = (c, kernel_size, str(image.device), image.dtype)
        if cache_key not in self._horizontal_kernel_cache:
            kernel = torch.ones(
                c,
                1,
                1,
                kernel_size,
                device=image.device,
                dtype=image.dtype,
            ) / float(kernel_size)
            self._horizontal_kernel_cache[cache_key] = kernel

        pad = kernel_size // 2
        padded = F.pad(image, [pad, pad, 0, 0], mode="reflect")
        return F.conv2d(padded, self._horizontal_kernel_cache[cache_key], groups=c)

    def _color_grade(self, image, cfg, look_strength, contrast_strength):
        look = look_strength
        contrast_scale = contrast_strength

        out = image

        exposure = cfg["exposure"] * look
        if abs(exposure) > MIN_EFFECT_STRENGTH:
            out = out * (2.0 ** exposure)

        temp = cfg["temperature"] * look
        tint = cfg["tint"] * look
        gains = torch.tensor(
            [
                1.0 + temp + tint * 0.20,
                1.0 - abs(tint) * 0.25,
                1.0 - temp + tint * 0.20,
            ],
            device=out.device,
            dtype=out.dtype,
        ).view(1, 3, 1, 1)
        out = out * gains

        lift = cfg["lift"] * look
        if abs(lift) > MIN_EFFECT_STRENGTH:
            out = out + lift

        contrast = cfg["contrast"] * contrast_scale * look
        if abs(contrast) > MIN_EFFECT_STRENGTH:
            pivot = 0.50
            out = (out - pivot) * (1.0 + contrast) + pivot

        # Soft filmic highlight rolloff. Keeps bright areas from looking clipped after contrast.
        luma = self._rgb_luma(out.clamp(0.0, 1.0))
        highlight_mask = torch.sigmoid((luma - 0.74) * 10.0)
        filmic = out / (out + 0.28)
        filmic = filmic * 1.22
        out = torch.lerp(out, filmic, highlight_mask * 0.18 * look)

        out = out.clamp(0.0, 1.0)

        saturation = 1.0 + cfg["saturation"] * look
        luma = self._rgb_luma(out)
        out = luma + (out - luma) * saturation

        return out.clamp(0.0, 1.0)

    def _bloom_and_halation(self, image, cfg, glow_strength):
        glow = glow_strength
        bloom_amount = cfg["bloom"] * glow
        halation_amount = cfg["halation"] * glow
        streak_amount = cfg["streak"] * glow

        if (
            bloom_amount <= MIN_EFFECT_STRENGTH and
            halation_amount <= MIN_EFFECT_STRENGTH and
            streak_amount <= MIN_EFFECT_STRENGTH
        ):
            return image

        luma = self._rgb_luma(image)
        threshold = cfg["bloom_threshold"]
        highlight_mask = torch.sigmoid((luma - threshold) * 12.0)
        highlights = image * highlight_mask

        blur_small = self._avg_blur(highlights, 9)
        blur_mid = self._avg_blur(highlights, 19)
        blur_large = self._avg_blur(highlights, 35)
        bloom = blur_small * 0.45 + blur_mid * 0.35 + blur_large * 0.20

        bloom_color = torch.tensor(
            cfg["bloom_color"],
            device=image.device,
            dtype=image.dtype,
        ).view(1, 3, 1, 1)
        bloom = bloom * bloom_color

        out = image + bloom * bloom_amount * 0.55

        if halation_amount > MIN_EFFECT_STRENGTH:
            halo = self._avg_blur(highlights, 23)
            warm = torch.tensor(
                [1.0, 0.42, 0.24],
                device=image.device,
                dtype=image.dtype,
            ).view(1, 3, 1, 1)
            out = out + halo * warm * halation_amount * 0.28

        if streak_amount > MIN_EFFECT_STRENGTH:
            streak = self._horizontal_blur(highlights, 71)
            warmth = cfg["streak_warmth"]
            if warmth >= 0.0:
                streak_color = (1.0, 0.86 + warmth * 0.10, 0.62 + warmth * 0.10)
            else:
                streak_color = (0.58, 0.78, 1.0)
            streak_color = torch.tensor(
                streak_color,
                device=image.device,
                dtype=image.dtype,
            ).view(1, 3, 1, 1)
            out = out + streak * streak_color * streak_amount * 0.42

        return out.clamp(0.0, 1.0)

    def _soft_beauty_blend(self, image, cfg, look_strength):
        soften = cfg["soften"] * look_strength
        if soften <= MIN_EFFECT_STRENGTH:
            return image

        blurred = self._avg_blur(image, 5)
        luma = self._rgb_luma(image)
        protect_edges = self._avg_blur(torch.abs(image - blurred).mean(dim=1, keepdim=True), 3)
        edge_guard = (1.0 - torch.clamp(protect_edges / 0.075, 0.0, 1.0)).clamp(0.0, 1.0)
        highlight_guard = 1.0 - torch.sigmoid((luma - 0.82) * 12.0) * 0.60
        blend = (edge_guard * highlight_guard * soften).clamp(0.0, 0.35)
        return torch.lerp(image, blurred, blend).clamp(0.0, 1.0)

    def _vignette_mask(self, h, w, device, dtype):
        cache_key = (h, w, str(device), dtype)
        if cache_key not in self._vignette_cache:
            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
                torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
                indexing="ij",
            )
            # Slightly oval vignette looks more cinematic for widescreen and vertical content.
            dist = torch.sqrt((xx * 0.92) ** 2 + (yy * 1.08) ** 2)
            mask = ((dist - 0.26) / 0.86).clamp(0.0, 1.0)
            mask = mask * mask * (3.0 - 2.0 * mask)
            self._vignette_cache[cache_key] = mask.view(1, 1, h, w)
        return self._vignette_cache[cache_key]

    def _apply_vignette(self, image, cfg, vignette_strength):
        strength = cfg["vignette"] * vignette_strength
        if strength <= MIN_EFFECT_STRENGTH:
            return image
        _, _, h, w = image.shape
        mask = self._vignette_mask(h, w, image.device, image.dtype)
        falloff = 1.0 - mask * strength * 0.65
        return (image * falloff).clamp(0.0, 1.0)

    def _base_grid(self, b, h, w, device, dtype):
        cache_key = (b, h, w, str(device), dtype)
        if cache_key not in self._grid_cache:
            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
                torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
                indexing="ij",
            )
            grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
            self._grid_cache[cache_key] = grid
        return self._grid_cache[cache_key]

    def _apply_chromatic_aberration(self, image, cfg, look_strength):
        amount = cfg["chromatic"] * look_strength
        if amount <= MIN_EFFECT_STRENGTH:
            return image

        b, c, h, w = image.shape
        if c != 3 or w <= 2:
            return image

        px = min(amount * 1.35, 2.0)
        dx = px * (2.0 / max(w - 1, 1))
        base = self._base_grid(b, h, w, image.device, image.dtype)

        grid_r = base.clone()
        grid_b = base.clone()
        grid_r[:, :, :, 0] = grid_r[:, :, :, 0] - dx
        grid_b[:, :, :, 0] = grid_b[:, :, :, 0] + dx

        red = F.grid_sample(
            image[:, 0:1],
            grid_r,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        green = image[:, 1:2]
        blue = F.grid_sample(
            image[:, 2:3],
            grid_b,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        shifted = torch.cat((red, green, blue), dim=1)

        # Keep the center cleaner and push the effect toward the edges.
        edge_mask = self._vignette_mask(h, w, image.device, image.dtype).clamp(0.0, 1.0)
        blend = (edge_mask * min(amount, 1.0) * 0.75).clamp(0.0, 1.0)
        return torch.lerp(image, shifted, blend).clamp(0.0, 1.0)

    def _film_grain(self, image, cfg, grain_strength, frame_index):
        amount = cfg["grain"] * grain_strength
        if amount <= MIN_EFFECT_STRENGTH:
            return image

        b, c, h, w = image.shape
        gh = max(8, h // 2)
        gw = max(8, w // 2)

        try:
            generator = torch.Generator(device=image.device)
            generator.manual_seed(9173 + int(frame_index) * 37)
            noise = torch.randn(
                (b, 1, gh, gw),
                generator=generator,
                device=image.device,
                dtype=image.dtype,
            )
        except Exception:
            noise = torch.randn((b, 1, gh, gw), device=image.device, dtype=image.dtype)

        noise = F.interpolate(noise, size=(h, w), mode="bilinear", align_corners=False)
        noise = noise.clamp(-2.5, 2.5) / 2.5

        luma = self._rgb_luma(image)
        shadow_bias = (1.0 - luma).clamp(0.0, 1.0) * 0.55 + 0.45
        grain = noise * amount * shadow_bias

        # Mostly luma grain, with a very small color separation for film feel.
        chroma = torch.cat((grain * 0.20, -grain * 0.08, grain * 0.10), dim=1)
        return (image + grain + chroma).clamp(0.0, 1.0)

    def _process_frame(
        self,
        rgb,
        cfg,
        look_strength,
        glow_strength,
        contrast_strength,
        grain_strength,
        vignette_strength,
        frame_index,
    ):
        out = rgb.clamp(0.0, 1.0)
        out = self._color_grade(out, cfg, look_strength, contrast_strength)
        out = self._bloom_and_halation(out, cfg, glow_strength)
        out = self._soft_beauty_blend(out, cfg, look_strength)
        out = self._apply_chromatic_aberration(out, cfg, look_strength)
        out = self._apply_vignette(out, cfg, vignette_strength)
        out = self._film_grain(out, cfg, grain_strength, frame_index)
        return out.clamp(0.0, 1.0)

    def execute(
        self,
        images,
        preset,
        look_strength=1.0,
        glow_strength=1.0,
        contrast_strength=1.0,
        grain_strength=1.0,
        vignette_strength=1.0,
    ):
        if len(images.shape) != 4:
            raise ValueError("VideoCinematicFX expects an IMAGE tensor with shape [B, H, W, C].")

        frame_count, height, width, channels = images.shape
        if channels < 3:
            raise ValueError("VideoCinematicFX expects RGB or RGBA images.")

        cfg = CINEMATIC_PRESETS.get(preset, CINEMATIC_PRESETS["Cinematic"])
        look_strength = self._clamp_strength(look_strength)
        glow_strength = self._clamp_strength(glow_strength)
        contrast_strength = self._clamp_strength(contrast_strength)
        grain_strength = self._clamp_strength(grain_strength)
        vignette_strength = self._clamp_strength(vignette_strength)

        device = self._get_device()

        extra_channels = None
        if channels > 3:
            extra_channels = images[:, :, :, 3:].detach().cpu()

        out_channels = 3 if extra_channels is None else 3 + extra_channels.shape[-1]
        out_tensor = torch.empty((frame_count, height, width, out_channels), dtype=images.dtype, device="cpu")

        progress = ProgressBar(frame_count) if ProgressBar is not None else None

        with torch.inference_mode():
            for i in range(frame_count):
                if mm is not None and hasattr(mm, "throw_exception_if_processing_interrupted"):
                    mm.throw_exception_if_processing_interrupted()

                rgb = images[i:i + 1].to(device).permute(0, 3, 1, 2).float()[:, :3]

                fx = self._process_frame(
                    rgb,
                    cfg,
                    look_strength,
                    glow_strength,
                    contrast_strength,
                    grain_strength,
                    vignette_strength,
                    i,
                )

                frame_out = fx.permute(0, 2, 3, 1).detach().cpu()
                if frame_out.dtype != out_tensor.dtype:
                    frame_out = frame_out.to(out_tensor.dtype)

                if extra_channels is not None:
                    frame_extra = extra_channels[i:i + 1]
                    if frame_extra.dtype != frame_out.dtype:
                        frame_extra = frame_extra.to(frame_out.dtype)
                    frame_out = torch.cat((frame_out, frame_extra), dim=-1)

                out_tensor[i:i + 1].copy_(frame_out)

                if progress is not None:
                    progress.update(1)

                if mm is not None and i > 0 and i % 50 == 0:
                    mm.soft_empty_cache()

        return (out_tensor,)


NODE_CLASS_MAPPINGS = {
    "VideoCinematicFX": VideoCinematicFX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCinematicFX": "🎬 Video Cinematic FX",
}
