# Created By MSXYZ and Claude Opus 4.6
# GPU başında ciddi işler döndürüyoruz. 🚀
# TAA (Temporal Anti-Aliasing) + DLAA (Deep Learning Anti-Aliasing) Adaptation
# v0.1.1 - Anti-Ghosting + Stability Improvements

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import comfy.model_management as mm

logger = logging.getLogger("VideoTAADLAA")

# =========================
# DLAA CORE (Uzamsal Filtre & Keskinleştirme)
# =========================
class _DLAANet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 3 Katmanlı basit bir Evrişimli Sinir Ağı (CNN).
        # Bu ağ, görüntüyü analiz edip detayları kurtarmak/keskinleştirmek için kullanılır.
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1, bias=False),
        )
        
        # Edge Detection (Kenar Tespiti) işlemleri için kullanılacak X ve Y yönlü Sobel filtreleri
        self.register_buffer("sobel_x", torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("sobel_y", torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        # Sub-pixel kaydırma (Jitter) için 4 yönlü offset (kayma) koordinatları
        self.register_buffer("jitter_offsets", torch.tensor([[0.25, 0.25], [-0.25, -0.25], [-0.25, 0.25], [0.25, -0.25]], dtype=torch.float32))
        self._init()

    def _init(self):
        convs = [m for m in self.conv if isinstance(m, nn.Conv2d)]
        
        # 1. Katman: Sadece ilk 3 kanalı (RGB) geçirip kalan 13 kanalı sıfırlarız. (Kaiming YERİNE)
        with torch.no_grad():
            convs[0].weight.zero_()
            for c in range(3):
                convs[0].weight[c, c, 1, 1] = 1.0 # RGB Identity
                
        # 2. Katman: Identity (Zaten doğru yapmışsın)
        with torch.no_grad():
            convs[1].weight.zero_()
            for c in range(min(convs[1].weight.shape[0], convs[1].weight.shape[1])):
                convs[1].weight[c, c, 1, 1] = 1.0
        
        # 3. Katman: Sharpening (Sadece ilk 3 kanalı kullanarak)
        nn.init.zeros_(convs[2].weight)
        sharpen = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]], dtype=torch.float32)
        with torch.no_grad():
            for out_c in range(3):
                # Girdinin sadece kendisine ait kanalını keskinleştirir (Cross-channel karışıklığını önler)
                convs[2].weight[out_c, out_c] = sharpen * 0.1

    def forward(self, x):
        
        # Residual connection: Orijinal görüntüye, ağdan çıkan keskinleştirilmiş detayları ekliyoruz
        return torch.clamp(x + self.conv(x) * 0.12, 0.0, 1.0)

# =========================
# TEMPORAL STATE (ZAMANSAL DURUM & ANTİ-GHOSTİNG)
# =========================
class _TAAState:
    def __init__(self):
        self.history = None # Önceki kareyi hafızada tutar
        self.frame_id = 0   # Jitter döngüsü için kare sayacı

    def reset(self):
        
        # Video değiştiğinde veya kullanıcı sıfırlamak istediğinde geçmişi siler
        self.history = None
        self.frame_id = 0

    def update(self, frame, alpha, sensitivity):
        
        # İlk kareyse veya çözünürlük değiştiyse direkt mevcut kareyi geçmiş olarak kaydet
        if self.history is None or self.history.shape != frame.shape:
            self.history = frame.clone()
            self.frame_id += 1
            return frame

        # GHOSTING'İ ÖLDÜREN KISIM: AGRESİF NEIGHBORHOOD CLAMPING
        # 3x3 komşulukta min/max sınırlarını alıyoruz.
        # Min değeri direkt bulmak yerine max_pool2d'yi ters işaretle kullanmak daha pratik oluyor.
        local_min = -F.max_pool2d(-frame, kernel_size=3, stride=1, padding=1)
        local_max = F.max_pool2d(frame, kernel_size=3, stride=1, padding=1)

        # Geçmiş kareyi mevcut frame’in renk sınırları içine çekiyoruz.
        # Bu, ghosting etkisini ciddi şekilde azaltıyor.
        history_clipped = torch.clamp(self.history, local_min, local_max)

        # HAREKET REDDİ (Motion Masking)
        # Mevcut kare ile kırpılmış geçmiş kare arasındaki farkı bulur.
        diff = torch.abs(frame - history_clipped).mean(dim=1, keepdim=True)
        
        # Hassasiyet eşiğine göre hareketli bölgelerde TAA etkisini azaltmak için bir maske oluşturur (0.0 - 1.0 arası)
        motion_mask = torch.sigmoid((diff - sensitivity) * 20.0)
        
        # Hareketin çok olduğu yerlerde geçmişin ağırlığını (alpha) düşürür
        dynamic_alpha = alpha * (1.0 - motion_mask)

        # Mevcut kare ile geçmiş kareyi dinamik alpha değerine göre harmanlar (Blend)
        out = torch.lerp(frame, history_clipped, dynamic_alpha)
        
        # Yeni geçmişi hafızaya kaydet ve sayacı artır (OOM yememek için detach() önemli)
        self.history = out.detach()
        self.frame_id += 1
        return out

# =========================
# MAIN NODE (COMFYUI ARAYÜZÜ)
# =========================
class VideoTAADLAA:
    def __init__(self):
        self.net_cache = {} # Farklı GPU cihazları için model önbelleği.
        self.taa = _TAAState()

    @classmethod
    def INPUT_TYPES(cls):
        
        # ComfyUI arayüzünde görünecek ayarlar
        return {
            "required": {
                "images": ("IMAGE",),
                "taa_strength": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.05}), 
                "taa_alpha": ("FLOAT", {"default": 0.6, "min": 0, "max": 0.95, "step": 0.01}),   
                "motion_sensitivity": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.5, "step": 0.01}),
                "jitter_scale": ("FLOAT", {"default": 0.05, "min": 0, "max": 1, "step": 0.01}),
                "dlaa_strength": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05}),
                "edge_threshold": ("FLOAT", {"default": 0.25, "min": 0, "max": 1, "step": 0.01}),
                "blur_radius": ("INT", {"default": 1, "min": 0, "max": 5, "step": 1}),
                "reset_history": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "CustomPostProcess"

    def _net(self, device):
    
        # Ağı ilgili cihaza (CPU/GPU) yükler ve bellekte tutar
        if device not in self.net_cache:
            self.net_cache[device] = _DLAANet().to(device).eval()
        return self.net_cache[device]

    def jitter(self, x, idx, scale, net):
        
        # Sub-pixel Jittering: Görüntüyü mikroskobik düzeyde kaydırarak tırtıkları yumuşatır.
        if scale < 1e-5: return x
        off = net.jitter_offsets[idx % 4]
        B, C, H, W = x.shape
        
        # Affine dönüşüm matrisi oluşturuluyor
        theta = torch.eye(2, 3, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        theta[:, 0, 2], theta[:, 1, 2] = off[0] * scale / W, off[1] * scale / H
        
        # Bilinear interpolasyon ile görüntüyü kaydırır (Kaba piksel kaydırmasından çok daha kalitelidir)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    def edge_aa(self, x, thr, blur, net):
        
        # Kenar Yumuşatma: Yalnızca keskin kenarları bulup hafifçe bulanıklaştırır
        if blur <= 0: return x
        
        # Görüntüyü gri tonlamaya (Grayscale) çevirir (İnsan gözünün Luma algısına göre ağırlıklandırılmış)
        gray = 0.299*x[:, 0:1] + 0.587*x[:, 1:2] + 0.114*x[:, 2:3]
        
        # Sobel filtreleri ile X ve Y eksenindeki farklılıklar (kenarlar) tespit edilir
        sx, sy = F.conv2d(gray, net.sobel_x, padding=1), F.conv2d(gray, net.sobel_y, padding=1)
        
        # Hipotenüs hesabı ile net kenar şiddeti bulunur
        # NaN oluşumunu önlemek için küçük epsilon eklenir
        edge = torch.sqrt(sx*sx + sy*sy + 1e-6)
        
        # Belirlenen eşik (threshold) değerine göre maske oluşturulur
        mask = torch.sigmoid((edge - thr) * 12.0)
        
        # Yansıtmalı dolgu (reflection pad) ile bulanıklaştırma yapılır
        blurred = F.avg_pool2d(F.pad(x, [blur]*4, mode="reflect"), blur*2+1, stride=1)
        
        # Sadece kenar olan yerlere bulanıklaştırılmış versiyon, diğer yerlere orijinal görüntü verilir
        return x*(1-mask) + blurred*mask

    def execute(self, images, taa_strength, taa_alpha, motion_sensitivity, jitter_scale, dlaa_strength, edge_threshold, blur_radius, reset_history=True):
        device = mm.get_torch_device()
        if reset_history: self.taa.reset()
        net = self._net(device)
        B, H, W, C = images.shape
        out_list = []
        
        with torch.no_grad():
            
            # Tüm videoyu tek seferde GPU'ya alıp OOM (Bellek Hatası) yememek için kareleri tek tek işliyoruz
            for i in range(B):
                
                # ComfyUI [B, H, W, C] formatını PyTorch [B, C, H, W] formatına çevir
                img = images[i:i+1].to(device).permute(0, 3, 1, 2).float()
                rgb = img[:, :3]

                # 1. Aşama: Uzamsal Kaydırma (Jitter) ve Kenar Yumuşatma (Edge)
                rgb = self.jitter(rgb, self.taa.frame_id, jitter_scale, net)
                rgb = self.edge_aa(rgb, edge_threshold, blur_radius, net)

                # 2. Aşama: TAA (Zamansal Anti-Aliasing - Nuclear Anti-Ghosting)
                taa_out = self.taa.update(rgb, taa_alpha, motion_sensitivity)
                rgb = torch.lerp(rgb, taa_out, taa_strength)

                # 3. Aşama: DLAA (Derin Öğrenme Tabanlı Keskinleştirme/Toparlama)
                if dlaa_strength > 0:
                    rgb = torch.lerp(rgb, net(rgb), dlaa_strength)
                
                # İşlenen kareyi RAM'e geri al ve tekrar ComfyUI formatına [B, H, W, C] çevir
                out_list.append(rgb.permute(0, 2, 3, 1).cpu())
                
                # Her 10 karede bir GPU belleğini temizleyerek kararlılığı artır
                if i % 10 == 0: mm.soft_empty_cache()
        
        # İşlenen tüm kareleri tek bir batch halinde ComfyUI'a döndür
        return (torch.cat(out_list, dim=0),)

# ComfyUI Node Kayıt İşlemleri
NODE_CLASS_MAPPINGS = {
    "VideoTAADLAA": VideoTAADLAA
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoTAADLAA": "🎮 Video TAA + DLAA Anti-Aliasing"
}