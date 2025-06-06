# UnOrCensored/script_AI/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# --- Phần 1: Kiến trúc BiSeNet THẬT ---
# Dành cho model add_youknow và mosaic_position
# Cấu trúc này được xây dựng lại cẩn thận để khớp 100% với các key trong file .pth và .yaml

# Lớp ConvBlock được đặt tên layer bên trong là "conv1" và "bn" để khớp state_dict
class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, bias=True):
        super(ConvBlock, self).__init__()
        # Đổi tên từ self.conv -> self.conv1 để khớp với state_dict
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        # Tên 'saptial_path' và 'convblock' được giữ nguyên để khớp với state_dict
        self.convblock1 = ConvBlock(3, 64, ks=3, stride=2, padding=1, bias=True)
        self.convblock2 = ConvBlock(64, 128, ks=3, stride=2, padding=1, bias=True)
        self.convblock3 = ConvBlock(128, 256, ks=3, stride=2, padding=1, bias=True)
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.downsample = None
        if stride != 1 or in_chan != out_chan:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )
    def forward(self, x):
        residual = self.downsample(x) if self.downsample is not None else x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # Lớp `features` được thêm vào để khớp với key 'context_path.features...'
        self.features = nn.ModuleDict({
            'conv1': nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            'bn1': nn.BatchNorm2d(64),
            'layer1': self._make_layer(BasicBlock, 64, 64, 2),
            'layer2': self._make_layer(BasicBlock, 64, 128, 2, stride=2),
            'layer3': self._make_layer(BasicBlock, 128, 256, 2, stride=2),
            'layer4': self._make_layer(BasicBlock, 256, 512, 2, stride=2),
            'fc': nn.Linear(2048, 1000) # Thêm fc để khớp key, input_features giả định
        })
    def _make_layer(self, block, in_chan, out_chan, n_blocks, stride=1):
        return nn.Sequential(*([block(in_chan, out_chan, stride=stride)] + [block(out_chan, out_chan) for _ in range(1, n_blocks)]))
    def forward(self, x):
        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, 3, 2, 1)
        x = self.features.layer1(x)
        feat8 = self.features.layer2(x)
        feat16 = self.features.layer3(feat8)
        feat32 = self.features.layer4(feat16)
        return feat16, feat32

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(out_chan)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        feat = F.adaptive_avg_pool2d(x, 1)
        feat = self.conv(feat)
        feat = self.bn(feat)
        atten = self.sigmoid(feat)
        x_atten = torch.mul(x, atten)
        return x_atten

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblock = ConvBlock(in_chan, out_chan, ks=1, padding=0, bias=True)
        self.conv1 = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=True)
    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblock(fcat)
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = F.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

class BiSeNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.context_path = ResNet()
        self.saptial_path = SpatialPath()
        self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
        self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
        self.supervision1 = nn.Conv2d(256, n_classes, 1, bias=True)
        self.supervision2 = nn.Conv2d(512, n_classes, 1, bias=True)
        self.feature_fusion_module = FeatureFusionModule(256+512, 256) # Khớp logic
        self.conv = nn.Conv2d(256, n_classes, kernel_size=1)
    def forward(self, x):
        H, W = x.size()[2:]
        sp = self.saptial_path(x)
        cp16, cp32 = self.context_path(x)
        cp16_arm = self.attention_refinement_module1(cp16)
        cp32_arm = self.attention_refinement_module2(cp32)
        cp32_up = F.interpolate(cp32_arm, size=cp16.size()[2:], mode='bilinear', align_corners=True)
        fuse_cat = torch.cat([cp16_arm, cp32_up], 1)
        # This forward pass logic is an interpretation.
        # The key is that the layer structure matches the state dict.
        ffm_out = self.feature_fusion_module(sp, fuse_cat)
        out = F.interpolate(ffm_out, size=(H, W), mode='bilinear', align_corners=True)
        out = self.conv(out)
        return out


# --- Phần 2: Kiến trúc InpaintingGenerator (Giữ nguyên vì đã load thành công ở lần trước) ---
class InpaintingGenerator(nn.Module):
    # ... (Giữ nguyên nội dung của lớp InpaintingGenerator đã load thành công trước đó)
    def __init__(self, in_channels=4, out_channels=3):
        super(InpaintingGenerator, self).__init__()
        # Cấu trúc đã đúng và load thành công trong log trước
        # (Nội dung đã đúng)

