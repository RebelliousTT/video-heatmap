import time
from ops.transforms1 import *
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['mkaresnet0']
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 封装一个3*3的卷积函数
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

# 封装一个1*1的卷积函数
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SiLU(torch.nn.Module):
    """
    [https://arxiv.org/pdf/1710.05941.pdf]
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())

class CotLayer(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CotLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size
        self.key_embed = nn.Sequential(
            Rearrange('(b n) c h w -> b c n h w', n=8),
            nn.Conv3d(dim, dim, (self.kernel_size, 1, 1), stride=1, padding=(self.kernel_size//2, 0, 0), groups=4, bias=False),
            Rearrange('b c n h w -> (b n) c h w'),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        self.share_planes = 4
        factor = 2

        self.embed = nn.Sequential(
            nn.Conv2d(2 * dim, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // factor, kernel_size * dim // self.share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // self.share_planes, num_channels=kernel_size * dim // self.share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.bn = nn.BatchNorm2d(dim)
        self.act = SiLU(inplace=True)

        reduction_factor = 2
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, self.radix * dim, 1))

    def forward(self, x):

        k = self.key_embed(x)
        qk = torch.cat([x, k], dim=1)
        b, c, qk_hh, qk_ww = qk.size()

        w = self.embed(qk)
        w = w.view(b, -1, 1, self.kernel_size, qk_hh, qk_ww)

        x = self.conv1x1(x)
        x = rearrange(x, '(b n) c h w -> b c n h w', n=8)
        x = F.pad(x, [0, 0, 0, 0, (self.kernel_size - 1) // 2, (self.kernel_size - 1) - ((self.kernel_size - 1) // 2)])
        x = x.unfold(2, self.kernel_size, 1)
        x = rearrange(x,'b (c1 c2) n h w k -> (b n) c1 c2 h w k', c2=self.share_planes)
        x = torch.einsum('bnckhw,bndhwk -> bndhw', w, x).reshape(b, -1, qk_hh, qk_ww)

        x = self.bn(x)
        x = self.act(x)

        x = rearrange(x, '(b n) c h w -> b c n h w', n=8)

        B, C, N, H, W = x.shape
        k = rearrange(k, '(b n) c h w -> b c n h w', n=8)
        x = x.view(B, C, 1, N, H, W)
        k = k.view(B, C, 1, N, H, W)
        x = torch.cat([x, k], dim=2)
        x_gap = x.sum(dim=2)

        x_gap = x_gap.mean((2, 3, 4), keepdim=True)
        x_gap = rearrange(x_gap, 'b c n h w -> (b n) c h w')
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1, 1))).sum(dim=2)
        out = rearrange(out, 'b c n h w -> (b n) c h w', n=8)

        return out.contiguous()

class AttentionModule(nn.Module):
    def __init__(self, dim=64, num_segments=8, typee=1):
        super(AttentionModule, self).__init__()

        self.conv3 = CotLayer(dim, kernel_size=5)#######################################################################

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        attn = self.conv3(x)

        return attn

class MKAttention(nn.Module):
    def __init__(self, dim=64, dim_mid=64, num_segments=8, typee=1):
        super().__init__()

        self.proj_1 = nn.Conv2d(dim, dim_mid, 1)
        self.activation = nn.ReLU(inplace=True)
        self.spatial_temporal_unit = AttentionModule(dim=dim_mid, num_segments=num_segments, typee=typee)
        self.proj_2 = nn.Conv2d(dim_mid, dim, 1)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        shorcut = x.clone()

        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_temporal_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class Shift(nn.Module):
    def __init__(self, dim=64, num_segments=8, shift_div=8, shift_grad=False):
        super(Shift, self).__init__()
        self.num_segments = num_segments
        self.fold = dim // shift_div


        if shift_grad:
            self.action_shift = nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=dim, bias=False)
            self.action_shift.weight.requires_grad = shift_grad
            self.action_shift.weight.data.zero_()
            self.action_shift.weight.data.normal_(0, 0.02)
        else:
            self.action_shift = No_param_shift(dim=dim, num_segments=num_segments, shift_div=shift_div)

    def forward(self, x):
        x = rearrange(x, '(b n) c h w -> b c n h w', n=self.num_segments)
        out = self.action_shift(x)  # (n_batch*h*w, c, n_segment)
        out = rearrange(out, 'b c n h w -> (b n) c h w')
        return out

class No_param_shift(nn.Module):
    def __init__(self, dim=64, num_segments=8, shift_div=8):
        super(No_param_shift, self).__init__()
        self.num_segments = num_segments
        self.fold = dim // shift_div

    def forward(self, x):
        out = torch.zeros_like(x)
        out[:, :self.fold, :-1] = x[:, :self.fold, 1:]  # shift left
        out[:, self.fold: 2 * self.fold, 1:] = x[:, self.fold: 2 * self.fold, :-1]  # shift right
        out[:, 2 * self.fold:, :] = x[:, 2 * self.fold:, :]  # not shift
        return out

# 定义了50 101 152 等深层resnet的残差模块，由1*1，3*3 ，1*1的卷积堆叠而成
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, num_segments=8, downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer=None, shift=False, shift_grad=False, mka=False, mka_typee=1, mka_squeeze=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.num_segments = num_segments
        if shift:
            self.shift = Shift(dim=inplanes, num_segments=num_segments, shift_div=8, shift_grad=shift_grad)
        else:
            self.shift = nn.Identity()
        if mka:
            self.mka = MKAttention(dim=inplanes, dim_mid=inplanes//mka_squeeze if inplanes>64 else 16, num_segments=8, typee=mka_typee)
        else:
            self.mka = nn.Identity()
    def forward(self, x):
        x = self.mka(x)
        identity = x

        out = self.shift(x)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Model(nn.Module):

    def __init__(self, num_class, num_segments, dropout, block=Bottleneck, layers=[3, 4, 6, 3], zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, mka_squeeze=4):
        super(Model, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.Norm_layer = norm_layer
        self.num_segments = num_segments
        self.modality = 'RGB'
        self.new_length = 1
        self.input_size = 224
        self.dropout = nn.Dropout(p=dropout)
        self.fc_lr5 = True
        self.target_transforms = {86: 87, 87: 86, 93: 94, 94: 93, 166: 167, 167: 166}


        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       shift=False, shift_grad=False, mka=False, mka_typee=1, mka_squeeze=mka_squeeze)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                       shift=False, shift_grad=False, mka=False, mka_typee=1, mka_squeeze=mka_squeeze)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                       shift=False, shift_grad=False, mka=False, mka_typee=2, mka_squeeze=mka_squeeze)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                       shift=False, shift_grad=False, mka=False, mka_typee=3, mka_squeeze=mka_squeeze)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, shift=True, shift_grad=False, mka=True, mka_typee=1, mka_squeeze=1):
        norm_layer = self.Norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                # Rearrange('(b n) c h w -> b c n h w', n=self.num_segments),
                norm_layer(planes * block.expansion),
                # Rearrange('b c n h w -> (b n) c h w', n=self.num_segments),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, self.num_segments, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, shift=shift, shift_grad=shift_grad,
                            mka=mka, mka_typee=mka_typee, mka_squeeze=mka_squeeze))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_segments=self.num_segments, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer,
                                shift=shift, shift_grad=shift_grad, mka=mka, mka_typee=mka_typee+1, mka_squeeze=mka_squeeze))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = rearrange(x, 'b (n c) h w -> (b n) c h w', n=self.num_segments)
        # See note [TorchScript super()]
        x = self.conv1(x)
        # x = rearrange(x, '(b n) c h w -> b c n h w', n=self.num_segments)
        x = self.bn1(x)
        # x = rearrange(x, 'b c n h w -> (b n) c h w', n=self.num_segments)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)


        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.dropout(x)

        x = self.fc(x)

        x = rearrange(x, '(b n) c -> b n c', n=self.num_segments)

        x = x.mean(dim=1)
        return x

    def forward(self, x):
        return self._forward_impl(x)

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if self.modality == 'RGB':
            if flip:
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
            else:
                print('#' * 20, 'sth FLIP!!!')
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip_sth(self.target_transforms)])


def mkaresnet0(num_classes=174, num_segments=16, drop_rate=0., drop_path_rate=0., **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    model = Model(num_classes, num_segments, drop_rate, block=Bottleneck, layers=[3, 4, 6, 3],
                  zero_init_residual=False,
                  groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, mka_squeeze=16)
    return model

########################################################################################################################

if __name__ == "__main__":
    import torchvision.models as models
    from torchsummary import summary

    net = mkaresnet0(num_classes=174, num_segments=8, drop_rate=0., drop_path_rate=0.).cuda()

    # summary(net, (24, 224, 224))

    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(net, (24, 224, 224), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # a = net.state_dict()
    end1 = time.time()
    x = torch.randn(2, 24, 224, 224).cuda()
    X = net(x)
    end2 = time.time()
    print(end2 - end1)
