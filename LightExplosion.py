import torch

class LightExplosion:
    """
    Bu node parlaklık, kontrast, doygunluk ve gamma ayarları yapar.
    Sadece PyTorch kullanarak GPU üzerinde hızla çalışır.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "Bloom": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "Glow": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "Explosion": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "Rays": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust"
    CATEGORY = "image/color"

    def adjust(self, image, brightness, contrast, saturation, gamma):
        # 1. Kontrast ve Parlaklık
        img = (image - 0.5) * contrast + 0.5 + brightness
        
        # 2. Doygunluk (Saturation)
        # Gri tonlama değerini hesapla (Luminance)
        # R*0.299 + G*0.587 + B*0.114
        grayscale = img[..., 0:1] * 0.299 + img[..., 1:2] * 0.587 + img[..., 2:3] * 0.114
        img = img * saturation + grayscale * (1 - saturation)
        
        # 3. Gamma Düzeltme
        # Gamma işlemi için değerlerin 0'dan büyük olması gerekir (clamp)
        img = torch.clamp(img, 1e-6, 1.0)
        img = torch.pow(img, 1.0 / gamma)
        
        # 4. Son Kırpma
        return (torch.clamp(img, 0.0, 1.0),)

# Node'un üzerinde fareyle beklendiğinde görünecek açıklama
NODE_DESCRIPTION = "Görüntü üzerinde Parlaklık, Kontrast, Doygunluk ve Gamma düzeltmeleri yapar."
