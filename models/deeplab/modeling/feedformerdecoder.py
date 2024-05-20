import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deeplab.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class FeedFormerDecoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, dropout_low, dropout_high):
        super(FeedFormerDecoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'drn_c42':
            low_level_inplanes = 64
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()

        self.self_attention = nn.MultiheadAttention(embed_dim=48, num_heads=4)
        self.ffn = nn.Sequential(
            nn.Linear(48, 192),
            nn.ReLU(),
            nn.Linear(192, 48)
        )
        
        self.last_conv = nn.Sequential(nn.Conv2d(304, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(64),
                                       nn.ReLU(),
                                       nn.Dropout(dropout_high),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(64),
                                       nn.ReLU(),
                                       nn.Dropout(dropout_low),
                                       nn.Conv2d(64, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        b, c, h, w = low_level_feat.shape
        low_level_feat_flat = low_level_feat.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        attn_output, _ = self.self_attention(low_level_feat_flat, low_level_feat_flat, low_level_feat_flat)
        attn_output = attn_output.permute(1, 2, 0).view(b, c, h, w)
        
        attn_output = attn_output + low_level_feat  # Residual connection
        ffn_output = self.ffn(attn_output.flatten(2).permute(2, 0, 1))
        ffn_output = ffn_output.permute(1, 2, 0).view(b, c, h, w)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, ffn_output), dim=1)
        x = self.last_conv(x)
        
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_feedformer_decoder(num_classes, backbone, BatchNorm, dlow, dhigh):
    return FeedFormerDecoder(num_classes, backbone, BatchNorm, dlow, dhigh)
