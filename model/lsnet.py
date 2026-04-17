import torch
import itertools

from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite
import numpy as np
import itertools

from mmcv_custom import load_checkpoint, _load_checkpoint, load_state_dict
from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES
from torch.nn.modules.batchnorm import _BatchNorm
from .ska import SKA

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Conv2d_BN(dim, h, ks=1)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0))
        self.dw = Conv2d_BN(nh_kd, nh_kd, 3, 1, 1, groups=nh_kd)
        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, -1, H, W).split([self.nh_kd, self.nh_kd, self.dh], dim=1)
        q = self.dw(q)
        q, k, v = q.view(B, self.num_heads, -1, N), k.view(B, self.num_heads, -1, N), v.view(B, self.num_heads, -1, N)
        attn = (q.transpose(-2, -1) @ k) * self.scale

        bias = self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        bias = torch.nn.functional.interpolate(bias.unsqueeze(0), size=(attn.size(-2), attn.size(-1)), mode='bicubic')
        attn = attn + bias
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).reshape(B, -1, H, W)
        x = self.proj(x)
        return x

class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
    
    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x
    
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv

import torch.nn as nn

class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        
        self.sks = sks
        self.groups = groups
        self.dim = dim
        
    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w

class LSConv(nn.Module):
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        return output


class Conv_DAB(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_prelu(output)
        return output


class DABModule_7(nn.Module):
    """DABModule with 7×1/1×7 asymmetric convolutions (instead of 3×1/1×3)
    and 7×1(D)/1×7(D) dilated convolutions (instead of 3×1(D)/1×3(D))."""
    def __init__(self, nIn, d=1):
        super().__init__()
        self.bn_relu_1 = BNPReLU(nIn)
        self.conv3x3 = Conv_DAB(nIn, nIn // 2, 3, 1, padding=1, bn_acti=True)

        self.dconv7x1 = Conv_DAB(nIn // 2, nIn // 2, (7, 1), 1,
                                  padding=(3, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x7 = Conv_DAB(nIn // 2, nIn // 2, (1, 7), 1,
                                  padding=(0, 3), groups=nIn // 2, bn_acti=True)
        self.ddconv7x1 = Conv_DAB(nIn // 2, nIn // 2, (7, 1), 1,
                                   padding=(3 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x7 = Conv_DAB(nIn // 2, nIn // 2, (1, 7), 1,
                                   padding=(0, 3 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv_DAB(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)

        br1 = self.dconv7x1(output)
        br1 = self.dconv1x7(br1)
        br2 = self.ddconv7x1(output)
        br2 = self.ddconv1x7(br2)

        output = br1 + br2
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output + input


class DASmall(DABModule_7):
    """DABModule_7 with fixed dilation=2 — stages 0-1, local context."""
    def __init__(self, nIn):
        super().__init__(nIn, d=2)


class DALarge(DABModule_7):
    """DABModule_7 with variable dilation — stage 2, wide multi-scale context."""
    def __init__(self, nIn, d):
        super().__init__(nIn, d=d)


class Block(torch.nn.Module):
    def __init__(self,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 stage=-1, depth=-1, dilation=2):
        super().__init__()

        if depth % 2 == 0:
            self.mixer = RepVGGDW(ed)
            self.se = SqueezeExcite(ed, 0.25)
        else:
            self.se = torch.nn.Identity()
            if stage == 3:
                self.mixer = Residual(Attention(ed, kd, nh, ar, resolution=14))
            elif stage < 2:
                self.mixer = DASmall(ed)
            else:
                self.mixer = DALarge(ed, d=dilation)

        self.ffn = Residual(FFN(ed, int(ed * 2)))

    def forward(self, x):
        return self.ffn(self.se(self.mixer(x)))

class LSNet(torch.nn.Module):
    def __init__(self, img_size=224,
                 patch_size=16,
                 frozen_stages = 0,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[64, 128, 192, 256],
                 key_dim=[16, 16, 16, 16],
                 depth=[1, 2, 3, 4],
                 num_heads=[4, 4, 4, 4],
                 pretrained=None,
                 distillation=False,):
        super().__init__()

        resolution = img_size
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim[0] // 4, 3, 2, 1), torch.nn.ReLU(),
                                Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1), torch.nn.ReLU(),
                                Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1)
                           )

        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1 = nn.Sequential()
        self.blocks2 = nn.Sequential()
        self.blocks3 = nn.Sequential()
        self.blocks4 = nn.Sequential()
        blocks = [self.blocks1, self.blocks2, self.blocks3, self.blocks4]
        
        # DAB Block 1 (stages 0-1): fixed dilation=2
        # DAB Block 2 (stage 2): cycling dilations following [4, 4, 8, 8, 16, 16]
        dab_block2_dilations = [4, 4, 8, 8, 16, 16]
        for i, (ed, kd, dpth, nh, ar) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio)):
            for d in range(dpth):
                if i < 2:
                    dilation = 2
                elif i == 2:
                    dilation = dab_block2_dilations[d % len(dab_block2_dilations)]
                else:
                    dilation = 2  # stage 3 uses Attention; dilation unused
                blocks[i].add_module(f'block_{d}', Block(ed, kd, nh, ar, resolution, stage=i, depth=d, dilation=dilation))
            
            if i != len(depth) - 1:
                blk = blocks[i+1]
                resolution_ = (resolution - 1) // 2 + 1
                blk.add_module(f'downsample_dw_{i}', Conv2d_BN(embed_dim[i], embed_dim[i], ks=3, stride=2, pad=1, groups=embed_dim[i]))
                blk.add_module(f'downsample_dw_{i}', Conv2d_BN(embed_dim[i], embed_dim[i+1], ks=1, stride=1, pad=0))
                resolution = resolution_

        self.frozen_stages = frozen_stages # freeze the patch embedding
        self._freeze_stages()

        if pretrained is not None:
            self.init_weights(pretrained=pretrained)
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            assert(False)
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if isinstance(pretrained, str):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(pretrained, map_location='cpu')

            if not isinstance(checkpoint, dict):
                raise RuntimeError(
                    f'No state_dict found in checkpoint file {pretrained}')
            # get state_dict from checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            model_state_dict = self.state_dict()
            # bicubic interpolate attention_biases if not match

            rpe_idx_keys = [
                k for k in state_dict.keys() if "attention_bias_idxs" in k]
            for k in rpe_idx_keys:
                print("deleting key: ", k)
                del state_dict[k]

            relative_position_bias_table_keys = [
                k for k in state_dict.keys() if "attention_biases" in k]
            for k in relative_position_bias_table_keys:
                if k in model_state_dict:
                    relative_position_bias_table_pretrained = state_dict[k]
                    relative_position_bias_table_current = model_state_dict[k]
                    nH1, L1 = relative_position_bias_table_pretrained.size()
                    nH2, L2 = relative_position_bias_table_current.size()
                    if nH1 != nH2:
                        logger.warning(f"Error in loading {k} due to different number of heads")
                    else:
                        if L1 != L2:
                            print("resizing key {} from {} * {} to {} * {}".format(k, L1, L1, L2, L2))
                            # bicubic interpolate relative_position_bias_table if not match
                            S1 = int(L1 ** 0.5)
                            S2 = int(L2 ** 0.5)
                            relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                                relative_position_bias_table_pretrained.view(1, nH1, S1, S1), size=(S2, S2),
                                mode='bicubic')
                            state_dict[k] = relative_position_bias_table_pretrained_resized.view(
                                nH2, L2)

                else:
                    # Nếu không có trong model thì bỏ qua, không báo lỗi nữa
                    print(f"Skipping key {k} as it is not in current model")

            load_state_dict(self, state_dict, strict=False, logger=logger)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(LSNet, self).train(mode)
        self._freeze_stages()

    def forward(self, x):
        x = self.patch_embed(x)
        outs = []
        x = self.blocks1(x)
        outs.append(x)
        x = self.blocks2(x)
        outs.append(x)
        x = self.blocks3(x)
        outs.append(x)
        x = self.blocks4(x)
        outs.append(x)
        return tuple(outs)

@BACKBONES.register_module()
def lsnet_t(num_classes=1000, distillation=False, pretrained=False, frozen_stages=0, **kwargs):
    model = LSNet(num_classes=num_classes, 
                  distillation=distillation, 
                  img_size=224,
                  patch_size=8,
                  embed_dim=[64, 128, 256, 384],
                  depth=[0, 2, 8, 10],
                  num_heads=[3, 3, 3, 4],
                  pretrained=pretrained,
                  frozen_stages=frozen_stages
                  )
    return model

@BACKBONES.register_module()
def lsnet_s(num_classes=1000, distillation=False, pretrained=False, frozen_stages=0, **kwargs):
    model = LSNet(num_classes=num_classes, 
                  distillation=distillation,
                  img_size=224,
                  patch_size=8,
                  embed_dim=[96, 192, 320, 448],
                  depth=[1, 2, 8, 10],
                  num_heads=[3, 3, 3, 4],
                  pretrained=pretrained,
                  frozen_stages=frozen_stages
                  )
    return model

@BACKBONES.register_module()
def lsnet_b(num_classes=1000, distillation=False, pretrained=False, frozen_stages=0, **kwargs):
    model = LSNet(num_classes=num_classes, 
                  distillation=distillation,
                  img_size=224,
                  patch_size=8,
                  embed_dim=[128, 256, 384, 512],
                  depth=[4, 6, 8, 10],
                  num_heads=[3, 3, 3, 4],
                  pretrained=pretrained,
                  frozen_stages=frozen_stages
                  )
    return model
