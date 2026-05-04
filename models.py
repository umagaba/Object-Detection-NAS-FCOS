import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.checkpoint import checkpoint

FPN_OPS = ["sep_conv_3x3", "sep_conv_3x3_d3", "sep_conv_5x5_d6", "skip", "deform_3x3"]
HEAD_OPS = FPN_OPS + ["conv1x1", "conv3x3"]
AGG_OPS = ["sum", "concat"]
N_FPN_BLOCKS, N_HEAD_OPS = 7, 6

class SepConv(nn.Module):
    def __init__(self, C, kernel_size=3, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.op = nn.Sequential(
            nn.Conv2d(C, C, kernel_size, padding=pad, dilation=dilation, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.GroupNorm(min(8, C), C),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.op(x)

class DeformConvApprox(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, C), C),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.op(x)

class Identity(nn.Module):
    def forward(self, x): return x

class Conv1x1(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C, C, 1, bias=False),
            nn.GroupNorm(min(8, C), C),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.op(x)

def build_op(op_name, C):
    if op_name == "sep_conv_3x3":    return SepConv(C, 3, 1)
    if op_name == "sep_conv_3x3_d3": return SepConv(C, 3, 3)
    if op_name == "sep_conv_5x5_d6": return SepConv(C, 5, 6)
    if op_name == "skip":            return Identity()
    if op_name == "deform_3x3":      return DeformConvApprox(C)
    if op_name == "conv1x1":         return Conv1x1(C)
    if op_name == "conv3x3":         return SepConv(C, 3, 1)
    raise ValueError(f"Unknown op: {op_name}")

class Backbone(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        resnet = torchvision.models.resnet50(weights="DEFAULT")
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.proj = nn.ModuleDict({
            "c3": nn.Conv2d(512,  out_channels, 1),
            "c4": nn.Conv2d(1024, out_channels, 1),
            "c5": nn.Conv2d(2048, out_channels, 1),
        })

    def forward(self, x, use_checkpoint=False):
        if use_checkpoint:
            x  = checkpoint(self.stem, x, use_reentrant=False)
            x  = checkpoint(self.layer1, x, use_reentrant=False)
            c3 = checkpoint(self.layer2, x, use_reentrant=False)
            c4 = checkpoint(self.layer3, c3, use_reentrant=False)
            c5 = checkpoint(self.layer4, c4, use_reentrant=False)
        else:
            x = self.layer1(self.stem(x))
            c3 = self.layer2(x)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)
        return {"c3": self.proj["c3"](c3), "c4": self.proj["c4"](c4), "c5": self.proj["c5"](c5)}

class ConcatFuse(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.fuse = nn.Conv2d(C * 2, C, 1, bias=False)
    def forward(self, a, b):
        if a.shape[-2:] != b.shape[-2:]:
            b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([a, b], dim=1))

class SumFuse(nn.Module):
    def forward(self, a, b):
        if a.shape[-2:] != b.shape[-2:]:
            b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return a + b

class NASFPNBlock(nn.Module):
    def __init__(self, C, op1_name, op2_name, agg_name, id1, id2):
        super().__init__()
        self.id1, self.id2 = id1, id2
        self.op1, self.op2 = build_op(op1_name, C), build_op(op2_name, C)
        self.agg = ConcatFuse(C) if agg_name == "concat" else SumFuse()

    def forward(self, pool):
        return self.agg(self.op1(pool[self.id1]), self.op2(pool[self.id2]))

class NASFPN(nn.Module):
    def __init__(self, C, fpn_arch):
        super().__init__()
        self.blocks = nn.ModuleList([
            NASFPNBlock(C, b["op1"], b["op2"], b["agg"], b["id1"], b["id2"]) for b in fpn_arch
        ])
        self.p6_conv = nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False)
        self.p7_conv = nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False)

    def forward(self, backbone_feats):
        pool = [backbone_feats["c3"], backbone_feats["c4"], backbone_feats["c5"]]
        block_outputs = []
        for block in self.blocks:
            x_t = block(pool)
            pool.append(x_t)
            block_outputs.append(x_t)

        p3, p4, p5 = block_outputs[-3], block_outputs[-2], block_outputs[-1]
        sampled_ids = {block.id1 for block in self.blocks} | {block.id2 for block in self.blocks}
        output_pool_indices = {len(pool)-3, len(pool)-2, len(pool)-1}

        for t in range(N_FPN_BLOCKS):
            pool_idx = 3 + t
            if pool_idx not in output_pool_indices and pool_idx not in sampled_ids:
                dang = pool[pool_idx]
                def _align(src, tgt):
                    return F.interpolate(src, tgt.shape[-2:], mode="bilinear", align_corners=False) if src.shape[-2:] != tgt.shape[-2:] else src
                p3, p4, p5 = p3 + _align(dang, p3), p4 + _align(dang, p4), p5 + _align(dang, p5)

        return [p3, p4, p5, self.p6_conv(p5), self.p7_conv(self.p6_conv(p5))]

class NASHead(nn.Module):
    def __init__(self, channels, num_classes, head_arch, share_from=0, n_levels=5):
        super().__init__()
        self.share_from = share_from
        self.n_levels = n_levels
        
        self.indep_ops = nn.ModuleList([
            nn.ModuleList([build_op(head_arch[i], channels) for i in range(share_from)])
            for _ in range(n_levels)
        ]) if share_from > 0 else nn.ModuleList()

        n_shared = len(head_arch) - share_from
        self.shared_ops = nn.ModuleList([
            build_op(head_arch[share_from + i], channels) for i in range(n_shared)
        ]) if n_shared > 0 else nn.ModuleList()

        self.cls = nn.Conv2d(channels, num_classes, 1)
        self.reg = nn.Conv2d(channels, 4, 1)
        self.ctr = nn.Conv2d(channels, 1, 1)

    def forward(self, pyramid):
        cls_out, reg_out, ctr_out = [], [], []
        for lvl, feat in enumerate(pyramid):
            x = feat
            if self.share_from > 0:
                for op in self.indep_ops[lvl % self.n_levels]: x = op(x)
            for op in self.shared_ops: x = op(x)
            cls_out.append(self.cls(x))
            reg_out.append(self.reg(x))
            ctr_out.append(self.ctr(x))
        return cls_out, reg_out, ctr_out

class NASFCOSDetector(nn.Module):
    def __init__(self, fpn_arch, head_arch, channels=256, num_classes=80, share_from=0):
        super().__init__()
        self.backbone = Backbone(out_channels=channels)
        self.fpn = NASFPN(channels, fpn_arch)
        self.head = NASHead(channels, num_classes, head_arch, share_from=share_from)

    def forward(self, x, use_checkpoint=False):
        feats = self.backbone(x, use_checkpoint=use_checkpoint)
        return self.head(self.fpn(feats))