# models/cnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional

# ---------- Norm factory ----------
def norm2d(num_channels: int, kind: Literal["bn", "gn"] = "bn", gn_groups: int = 8):
    if kind == "bn":
        return nn.BatchNorm2d(num_channels)
    elif kind == "gn":
        # ασφαλές clamp για να μη σκάσει σε μικρά C
        g = max(1, min(gn_groups, num_channels))
        return nn.GroupNorm(g, num_channels)
    else:
        raise ValueError(f"Unsupported norm kind: {kind}")

# ---------- SE block (optional) ----------
class SEBlock(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        hidden = max(8, c // r)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w

# ---------- Conv-Norm-Act ----------
class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, norm="bn", gn_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.norm = norm2d(out_c, kind=norm, gn_groups=gn_groups)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

# ---------- Upgraded block ----------
class EnhancedBlock(nn.Module):
    """
    Διπλό 3×3 conv, configurable downsample μέσω stride=2 στο πρώτο conv,
    optional SE, optional residual όταν ταιριάζουν τα dims.
    """
    def __init__(
        self,
        in_c: int,
        out_c: int,
        downsample: bool = False,
        use_se: bool = True,
        norm: Literal["bn", "gn"] = "bn",
        gn_groups: int = 8,
        residual: bool = True,
    ):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = ConvBNAct(in_c, out_c, k=3, s=stride, p=1, norm=norm, gn_groups=gn_groups)
        self.conv2 = ConvBNAct(out_c, out_c, k=3, s=1, p=1, norm=norm, gn_groups=gn_groups)
        self.se     = SEBlock(out_c) if use_se else nn.Identity()

        # Προαιρετικό skip (1×1 conv) όταν αλλάζει C ή γίνεται downsample
        self.use_skip = residual and (in_c != out_c or downsample)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                norm2d(out_c, kind=norm, gn_groups=gn_groups),
            )
            if self.use_skip else nn.Identity()
        )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        if self.use_skip:
            identity = self.skip(identity)
        return F.relu(x + identity) if isinstance(self.skip, nn.Module) else x

# ---------- The upgraded CNN ----------
class CustomCNNPlus(nn.Module):
    """
    VGG-like αλλά:
    - Downsample με conv(stride=2) αντί για MaxPool
    - 2×3×3 conv ανά στάδιο
    - SE attention (optional)
    - Επιλογή BatchNorm ή GroupNorm
    - Προαιρετικό residual για ευκολότερη βελτιστοποίηση
    """
    def __init__(
        self,
        num_classes: int = 4,
        dropout: float = 0.4,
        channels=(32, 64, 128, 256),
        use_se: bool = True,
        norm: Literal["bn", "gn"] = "bn",
        gn_groups: int = 8,
        residual: bool = True,
    ):
        super().__init__()
        c1, c2, c3, c4 = channels

        # Στάδιο 1: χωρίς downsample στην αρχή, μετά κάθε στάδιο downsample= True
        self.stem = EnhancedBlock(3, c1, downsample=False, use_se=use_se, norm=norm, gn_groups=gn_groups, residual=residual)
        self.stage2 = EnhancedBlock(c1, c2, downsample=True,  use_se=use_se, norm=norm, gn_groups=gn_groups, residual=residual)
        self.stage3 = EnhancedBlock(c2, c3, downsample=True,  use_se=use_se, norm=norm, gn_groups=gn_groups, residual=residual)
        self.stage4 = EnhancedBlock(c3, c4, downsample=True,  use_se=use_se, norm=norm, gn_groups=gn_groups, residual=residual)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)   # B, c1, H, W
        x = self.stage2(x) # B, c2, H/2, W/2
        x = self.stage3(x) # B, c3, H/4, W/4
        x = self.stage4(x) # B, c4, H/8, W/8
        x = self.gap(x)    # B, c4, 1, 1
        x = self.classifier(x)  # B, num_classes
        return x

# ---------- Public factory ----------
def get_model(
    num_classes: int = 4,
    backbone: str = "custom",
    **kwargs
):
    if backbone == "custom":
        # παλιό baseline για reference
        dropout = kwargs.get("dropout", 0.4)
        return CustomCNN(num_classes=num_classes, dropout=dropout)
    elif backbone in {"custom+", "custom_plus"}:
        return CustomCNNPlus(
            num_classes=num_classes,
            dropout=kwargs.get("dropout", 0.4),
            channels=kwargs.get("channels", (32, 64, 128, 256)),
            use_se=kwargs.get("use_se", True),
            norm=kwargs.get("norm", "bn"),          # "bn" ή "gn"
            gn_groups=kwargs.get("gn_groups", 8),
            residual=kwargs.get("residual", True),
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

# ---------- Original baseline kept for compatibility ----------
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.pool(F.relu(self.bn(self.conv(x))))
        return x

class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            BasicBlock(3, 32, pool=True),
            BasicBlock(32, 64, pool=True),
            BasicBlock(64, 128, pool=True),
            BasicBlock(128, 256, pool=True)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
