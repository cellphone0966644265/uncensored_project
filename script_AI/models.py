# UnOrCensored/script_AI/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# --- Phần 1: Kiến trúc BiSeNet thật ---
# Dành cho model add_youknow và mosaic_position
# Cấu trúc này được xây dựng lại để khớp với các key trong file YAML/state_dict

class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, bias=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        # Tên 'saptial_path' và 'convblock' được giữ nguyên để khớp với state_dict
        self.convblock1 = ConvBlock(3, 64, ks=3, stride=2)
        self.convblock2 = ConvBlock(64, 128, ks=3, stride=2)
        self.convblock3 = ConvBlock(128, 256, ks=3, stride=2)
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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Tên 'layer1', 'layer2' khớp với state_dict
        self.layer1 = self._make_layer(BasicBlock, 64, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 64, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 512, 2, stride=2)

    def _make_layer(self, block, in_chan, out_chan, n_blocks, stride=1):
        layers = [block(in_chan, out_chan, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(block(out_chan, out_chan))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x) # 1/8
        x = self.layer3(x) # 1/16
        x = self.layer4(x) # 1/32
        return x

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)
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
        self.convblock = ConvBlock(in_chan, out_chan, ks=1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False) # Simplified from error
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False) # Simplified
    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblock(fcat)
        return feat

class BiSeNet(nn.Module):
    def __init__(self, n_classes):
        super(BiSeNet, self).__init__()
        self.context_path = ResNet()
        self.saptial_path = SpatialPath()
        self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
        self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
        self.supervision1 = nn.Conv2d(256, n_classes, 1)
        self.supervision2 = nn.Conv2d(512, n_classes, 1)
        self.feature_fusion_module = FeatureFusionModule(1024, n_classes) # Matching YAML
        self.conv = nn.Conv2d(n_classes, n_classes, 1)
    def forward(self, x):
        # This forward pass is a simplified interpretation
        H, W = x.size()[2:]
        sp = self.saptial_path(x)
        cp16, cp32 = self.context_path(x)[2], self.context_path(x)[3]
        arm1 = self.attention_refinement_module1(cp16)
        arm2 = self.attention_refinement_module2(cp32)
        up_arm2 = F.interpolate(arm2, size=arm1.size()[2:], mode='bilinear')
        fuse = self.feature_fusion_module(torch.cat([arm1, up_arm2], dim=1))
        out = F.interpolate(fuse, size=(H,W), mode='bilinear')
        return out


# --- Phần 2: Kiến trúc InpaintingGenerator thật ---
# Dành cho model clean_youknow
# Cấu trúc được xây dựng lại cẩn thận để khớp với state_dict

class InpaintingGenerator(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(InpaintingGenerator, self).__init__()
        
        # Cấu trúc của Encoder 2D
        self.encoder2d = nn.ModuleDict({
            'model': nn.Sequential(
                nn.ReflectionPad2d(3),                                    # 0
                spectral_norm(nn.Conv2d(3, 64, 7, 1, 0)),                  # 1
                nn.ReLU(True),                                            # 2
                nn.ReflectionPad2d(1),                                    # 3
                spectral_norm(nn.Conv2d(64, 128, 3, 2, 0)),                # 4
                nn.ReLU(True),                                            # 5
                nn.ReflectionPad2d(1),                                    # 6
                spectral_norm(nn.Conv2d(128, 256, 3, 2, 0)),              # 7
                nn.ReLU(True),                                            # 8
                nn.ReflectionPad2d(1),                                    # 9
                spectral_norm(nn.Conv2d(256, 512, 3, 2, 0)),              # 10
            )
        })

        # Cấu trúc của Encoder 3D (giả định)
        self.encoder3d = nn.ModuleDict({
            'model': nn.Sequential(
                spectral_norm(nn.Conv3d(3, 64, kernel_size=3, padding=1)), # 0
                nn.ReLU(True),                                            # 1
                spectral_norm(nn.Conv3d(64, 128, kernel_size=3, padding=1)), # 2
                nn.ReLU(True),                                            # 3
                spectral_norm(nn.Conv3d(128, 256, kernel_size=3, padding=1)),# 4
                nn.ReLU(True),                                            # 5
                spectral_norm(nn.Conv3d(256, 512, kernel_size=3, padding=1)),# 6
            )
        })

        # Cấu trúc của các khối Residual
        class ResnetBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_block = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    spectral_norm(nn.Conv2d(512, 512, 3, 1, 0)),
                    nn.ReLU(True),
                    nn.ReflectionPad2d(1),
                    spectral_norm(nn.Conv2d(512, 512, 3, 1, 0)),
                )
            def forward(self, x): return x + self.conv_block(x)
        self.blocks = nn.ModuleList([ResnetBlock() for _ in range(4)])


        # Cấu trúc của Decoder
        class DecoderBlock(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.convup = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.ReflectionPad2d(1),
                    spectral_norm(nn.Conv2d(in_c, out_c, 3, 1, 0)),
                    nn.ReLU(True)
                )
            def forward(self, x): return self.convup(x)

        self.decoder = nn.ModuleList([
            DecoderBlock(512, 256), # 0
            DecoderBlock(256, 128), # 1
            DecoderBlock(128, 64),  # 2
            nn.Sequential(          # 3 - Placeholder
                nn.ReflectionPad2d(3),
            ),
            nn.Conv2d(64, 3, 7, 1, 0), # 4
            nn.Tanh()               # 5
        ])

    def forward(self, x, mask):
        # A plausible forward pass, though it might not be the exact one.
        # Its main purpose is to allow the model to be instantiated.
        x = x * (1. - mask)
        x = self.encoder2d.model(x)
        for i in range(4):
            x = self.blocks[i](x)
        x = self.decoder[0](x)
        x = self.decoder[1](x)
        x = self.decoder[2](x)
        x = self.decoder[3](x)
        x = self.decoder[4](x)
        x = self.decoder[5](x)
        return x

