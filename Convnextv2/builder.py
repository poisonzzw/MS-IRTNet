from Convnextv2.convnextv2 import ConvNeXtV2
import torch.nn.functional as F
import torch.nn as nn
from Convnextv2.MFT import transformer
from Convnextv2.VIT import react


class Convnextv2(nn.Module):
    def __init__(self, in_chans=3, num_classes=9, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs):
        super().__init__()
        self.num_stages = 4
        self.dec_outChannels=768
        self.encoder_rgb = ConvNeXtV2(in_chans=in_chans,depths=depths, dims=dims, pretrained = True, **kwargs)
        self.encoder_thermal = ConvNeXtV2(in_chans=in_chans,depths=depths, dims=dims, pretrained = True, **kwargs)
        # GWIM: gate-weighted interaction module
        self.react = nn.ModuleList([
            react(dim=dims[0], reduction=1),
            react(dim=dims[1], reduction=1),
            react(dim=dims[2], reduction=1),
            react(dim=dims[3], reduction=1)
        ])
        # FIIM: feature information interaction module
        self.transformer = nn.ModuleList([
            transformer(dim=dims[0], reduction=1, num_heads=1, norm_layer=nn.BatchNorm2d),
            transformer(dim=dims[1], reduction=1, num_heads=2, norm_layer=nn.BatchNorm2d),
            transformer(dim=dims[2], reduction=1, num_heads=4, norm_layer=nn.BatchNorm2d),
            transformer(dim=dims[3], reduction=1, num_heads=8, norm_layer=nn.BatchNorm2d)])

        from Convnextv2.MLPDecoder import DecoderHead
        self.decoder = DecoderHead(in_channels=[96, 192, 384, 768], num_classes=num_classes, norm_layer=nn.BatchNorm2d,
                                   embed_dim=512)

    def forward(self, rgb, thermal):
        raw_rgb = rgb

        enc_rgb = self.encoder_rgb(rgb)
        enc_thermal = self.encoder_thermal(thermal)
        enc_feats = []

        for i in range(self.num_stages):

            vi, ir = self.react[i](enc_rgb[i], enc_thermal[i])
            x_fused = self.transformer[i](vi, ir)
            enc_feats.append(x_fused)

        dec_out = self.decoder(enc_feats)

        output = F.interpolate(dec_out, size=raw_rgb.size()[-2:], mode='bilinear',
                               align_corners=True)
        return output
