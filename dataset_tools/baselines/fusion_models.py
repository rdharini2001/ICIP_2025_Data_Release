"""
Reference mid-level (feature) fusion module for AVID: two independent
backbone streams (RGB, IR), combined via a channel + spatial attention gate
in the spirit of CBAM/SE-style fusion, before a shared detection head.

Scope note: this is a compact *reference* implementation meant to give
researchers a correct, minimal starting point for the fusion architecture
this dataset is designed to stress-test -- it is not a reproduction of any
specific production detector (YOLOv8/v11, DETR, etc.). For a stronger,
production-grade baseline, two paths are both legitimate and simpler than
reimplementing a full detector:

1. Early fusion: patch an existing pretrained detector's first conv layer to
   accept a 4-channel (RGB+IR-gray) or 6-channel (RGB + IR-as-3ch) input; see
   `patch_yolo_early_fusion()` below.
2. Late fusion: train two independent single-modality detectors and combine
   their outputs with `dataset_tools/baselines/late_fusion_wbf.py`.

NOTE ON TEST STATUS: same caveat as dataset.py -- this needs `torch`, not
available in the sandbox this was authored in. Validate with:

    python -m dataset_tools.baselines.fusion_models --self-test

which runs a forward pass on random tensors of the real (256, 320) input
size and checks output shapes, plus one gradient step to confirm the graph
is fully differentiable end-to-end.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = object


class ConvBlock(nn.Module if torch else object):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Backbone(nn.Module if torch else object):
    """A small 5-stage strided-conv backbone (stride 32 total), stream-agnostic
    (works for 3-channel RGB or 1-channel IR by setting in_ch)."""

    def __init__(self, in_ch, widths=(32, 64, 128, 256, 256)):
        super().__init__()
        layers = []
        c_in = in_ch
        for c_out in widths:
            layers.append(ConvBlock(c_in, c_out))
            c_in = c_out
        self.stages = nn.ModuleList(layers)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x  # (B, widths[-1], H/32, W/32)


class CBAMFusion(nn.Module if torch else object):
    """Channel + spatial attention gate fusing two same-shape feature maps
    (RGB-stream, IR-stream) into one, following the CBAM attention formulation
    (Woo et al., ECCV 2018) applied to the concatenated dual-stream features."""

    def __init__(self, channels_per_stream, reduction=8):
        super().__init__()
        c = channels_per_stream * 2
        self.channel_mlp = nn.Sequential(
            nn.Linear(c, max(c // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(c // reduction, 4), c),
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.project = nn.Conv2d(c, channels_per_stream, kernel_size=1)

    def forward(self, feat_rgb, feat_ir):
        x = torch.cat([feat_rgb, feat_ir], dim=1)  # (B, 2C, H, W)

        avg_pool = x.mean(dim=(2, 3))
        max_pool = x.amax(dim=(2, 3))
        channel_att = torch.sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool))
        x = x * channel_att.unsqueeze(-1).unsqueeze(-1)

        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.spatial_conv(torch.cat([avg_map, max_map], dim=1)))
        x = x * spatial_att

        return self.project(x)  # back to channels_per_stream for the shared head


class DetectionHead(nn.Module if torch else object):
    """Minimal anchor-free head: per grid cell, predicts (objectness, class
    logits, box regression as (dx, dy, dw, dh) relative to the cell)."""

    def __init__(self, in_ch, n_classes=2):
        super().__init__()
        self.n_classes = n_classes
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1)
        self.act = nn.SiLU(inplace=True)
        self.out = nn.Conv2d(in_ch, 5 + n_classes, kernel_size=1)  # obj + 4 box + classes

    def forward(self, x):
        x = self.act(self.conv(x))
        return self.out(x)  # (B, 5+n_classes, H/32, W/32)


class MidFusionDetector(nn.Module if torch else object):
    """RGB-stream + IR-stream backbones -> CBAM feature fusion -> shared head."""

    def __init__(self, n_classes=2, backbone_width=256):
        super().__init__()
        self.rgb_backbone = Backbone(in_ch=3, widths=(32, 64, 128, 256, backbone_width))
        self.ir_backbone = Backbone(in_ch=1, widths=(32, 64, 128, 256, backbone_width))
        self.fusion = CBAMFusion(channels_per_stream=backbone_width)
        self.head = DetectionHead(backbone_width, n_classes=n_classes)

    def forward(self, rgb, ir):
        f_rgb = self.rgb_backbone(rgb)
        f_ir = self.ir_backbone(ir)
        fused = self.fusion(f_rgb, f_ir)
        return self.head(fused)


class SingleModalityDetector(nn.Module if torch else object):
    """RGB-only or IR-only baseline sharing the same backbone/head design as
    MidFusionDetector so comparisons isolate the effect of fusion, not
    architecture differences."""

    def __init__(self, in_ch, n_classes=2, backbone_width=256):
        super().__init__()
        self.backbone = Backbone(in_ch=in_ch, widths=(32, 64, 128, 256, backbone_width))
        self.head = DetectionHead(backbone_width, n_classes=n_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


def patch_yolo_early_fusion(model, extra_channels=1):
    """
    Patch an Ultralytics YOLO model (`model.model[0].conv`, the first conv
    layer) to accept `3 + extra_channels` input channels for early RGB-IR
    fusion (extra_channels=1 for RGB+IR-gray concat, 3 for RGB+IR-as-3ch,
    matching what several real 2025-challenge teams did with channel
    concatenation). Pretrained RGB weights are kept for the first 3 input
    channels; the new channels are initialized as the mean of the RGB
    kernels so the extra channel starts as "another vote from the existing
    filters" rather than random noise.

    Usage (requires `ultralytics` installed separately, not a dependency of
    this repo):

        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        patch_yolo_early_fusion(model.model, extra_channels=1)
        # then fine-tune on 4-channel AVIDPairDataset(..., fusion_mode="early")
    """
    if torch is None:
        raise ImportError("patch_yolo_early_fusion requires PyTorch: pip install torch")

    first_conv = model.model[0].conv  # ultralytics Conv wraps nn.Conv2d as .conv
    old_weight = first_conv.weight.data  # (out_ch, 3, k, k)
    new_in_ch = 3 + extra_channels
    new_conv = nn.Conv2d(
        new_in_ch,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = old_weight
        mean_kernel = old_weight.mean(dim=1, keepdim=True)  # (out_ch, 1, k, k)
        for c in range(extra_channels):
            new_conv.weight[:, 3 + c : 3 + c + 1] = mean_kernel
        if first_conv.bias is not None:
            new_conv.bias[:] = first_conv.bias
    model.model[0].conv = new_conv
    return model


def _self_test():
    assert torch is not None, "install torch to run this self-test: pip install torch"
    b, h, w = 2, 256, 320
    rgb = torch.randn(b, 3, h, w)
    ir = torch.randn(b, 1, h, w)

    mid_model = MidFusionDetector(n_classes=2)
    out = mid_model(rgb, ir)
    expected_hw = (h // 32, w // 32)
    assert out.shape == (b, 5 + 2, *expected_hw), out.shape

    rgb_only = SingleModalityDetector(in_ch=3, n_classes=2)
    out_rgb = rgb_only(rgb)
    assert out_rgb.shape == (b, 5 + 2, *expected_hw), out_rgb.shape

    ir_only = SingleModalityDetector(in_ch=1, n_classes=2)
    out_ir = ir_only(ir)
    assert out_ir.shape == (b, 5 + 2, *expected_hw), out_ir.shape

    # one optimization step to confirm the fused graph is fully
    # differentiable end to end (RGB stream, IR stream, and fusion gate all
    # receive gradients).
    opt = torch.optim.SGD(mid_model.parameters(), lr=1e-3)
    loss = out.pow(2).mean()
    loss.backward()
    grads_ok = all(p.grad is not None for p in mid_model.parameters() if p.requires_grad)
    assert grads_ok, "some parameters received no gradient"
    opt.step()

    print("fusion_models self-test passed:")
    print("  MidFusionDetector output:", out.shape)
    print("  SingleModalityDetector (rgb) output:", out_rgb.shape)
    print("  SingleModalityDetector (ir) output:", out_ir.shape)
    print("  backward pass: all parameters received gradients")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
