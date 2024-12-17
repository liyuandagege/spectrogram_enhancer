import math
from functools import partial
from math import log2
from typing import List
# import numpy.nn as nn
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from kornia.filters import filter2d

from nemo.collections.tts.parts.utils.helpers import mask_sequence_tensor

# Helper function to build the activation layer
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

# Efficient up-convolution block (EUCB)
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x 

def build_act_layer(act_type):
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()

# Element-wise scaling layer
class ElementScale(nn.Module):
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

# Multi-order Depthwise 1D Convolution
class MultiOrderDWConv1D(nn.Module):
    def __init__(self, embed_dims, dw_dilation=[1, 2, 3], channel_split=[1, 2, 2]):  # 1, 3, 4
        super(MultiOrderDWConv1D, self).__init__()
        
        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert embed_dims % sum(channel_split) == 0

        # Basic DW conv with 1D convolutions
        self.DW_conv0 = nn.Conv1d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=2 * dw_dilation[0],  # Adjust padding for 1D
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv1d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=2 * dw_dilation[1],  # Adjust padding for 1D
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv1d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=3 * dw_dilation[2],  # Adjust padding for 1D
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2],
        )
        # Pointwise convolution to aggregate channel information
        self.PW_conv = nn.Conv1d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1
        )

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0 + self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims - self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x

# Multi-order Gated Aggregation with Conv1D
class MultiOrderGatedAggregation1D(nn.Module):
    def __init__(self, embed_dims, attn_dw_dilation=[1, 2, 3], attn_channel_split=[1, 3, 4], attn_act_type='SiLU', attn_force_fp32=False): # SiLU
        super(MultiOrderGatedAggregation1D, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv1d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv1d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv1D(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv1d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # Activation functions for gating and value branches
        self.act_value = build_act_layer(attn_act_type)
        self.act_gate = build_act_layer(attn_act_type)

        # Element-wise scaling layer for decomposed feature
        self.sigma = ElementScale(embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, W] -> [B, C, 1] using 1D adaptive average pooling
        x_d = F.adaptive_avg_pool1d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        # Ensure gating is in float32 for numerical stability if specified
        g = g.to(torch.float32)
        v = v.to(torch.float32)
        return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()
        # Feature decomposition
        x = self.feat_decompose(x)
        # Gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # Aggregation with gating
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut  # Residual connection
        return x

def normalize_2nd_moment(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

class PixelShuffleUpsampleBlock(nn.Module):
    def __init__(self, in_channels, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        return self.pixel_shuffle(x)

class Blur(torch.nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer("f", f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)


class EqualLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=1, bias=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


class ExtendedMappingNetwork(nn.Module):
    def __init__(
        self,
        z_dim: int,
        conditional: bool = True,
        num_layers: int = 2,
        activation: str = 'lrelu',
        lr_multiplier: float = 0.01,
        x_avg_beta: float = 0.995,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.x_avg_beta = x_avg_beta
        self.num_ws = None
        self.training = True
        self.mlp = MLP([z_dim] * (num_layers + 1), activation=activation,
                       lr_multiplier=lr_multiplier, linear_out=True)

        if conditional:
            self.audio_clip = AudioCLIP(pretrained=True)
            self.c_dim = self.audio_clip.embed_dim  # AudioCLIP embedding dimension
            # Freeze AudioCLIP weights (no training)
            for param in self.audio_clip.parameters():
                param.requires_grad = False
            self.audio_clip.eval()  # Set AudioCLIP to evaluation mode

            # Add Transformer Encoder to process the AudioCLIP embeddings
            self.text_transformer = TransformerEncoder(dim=self.c_dim, num_heads=8, num_layers=4)
        else:
            self.c_dim = 0

        self.w_dim = self.c_dim + self.z_dim
        self.fusion_layer = nn.Linear(self.w_dim, self.z_dim)
        self.register_buffer('x_avg', torch.zeros([self.z_dim]))

    def encode_text(self, c, device):
        if self.c_dim > 0:
            self.audio_clip = self.audio_clip.to(device)
            embedding = self.audio_clip.encode_text(c)  # Returns tensor of shape [batch_size, embed_dim]

            # Pass through Transformer Encoder
            embedding = embedding.unsqueeze(1)  # Add sequence length dimension
            embedding = self.text_transformer(embedding)  # Process with Transformer
            embedding = embedding.squeeze(1)  # Remove sequence length dimension

            return embedding
        return None

    def forward(
        self,
        z: torch.Tensor, # random
        c: Optional[torch.Tensor] = None,  # text_embedding
        truncation_psi: float = 0.5,
    ) -> torch.Tensor:
        assert z.shape[1] == self.z_dim
        device = z.device

        x = self.mlp(F.normalize(z, dim=1))
        self.x_avg = self.x_avg.to(x.dtype)

        if self.x_avg_beta is not None and self.training:
            self.x_avg.copy_(x.detach().mean(0).lerp(self.x_avg, self.x_avg_beta))

        if truncation_psi != 1:
            x = self.x_avg.lerp(x, truncation_psi)
            if self.c_dim > 0 and c is not None:
                # c = self.encode_text(c, device)
                w = torch.cat([x, c], 1)
                w = self.fusion_layer(w)
            else:
                w = x
        return w



class RGBBlock(torch.nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, channels=3):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = torch.nn.Linear(latent_dim, input_channel)

        out_filters = channels
        self.conv = Conv2DModulated(input_channel, out_filters, 1, demod=False)

        self.upsample = (
            # torch.nn.Sequential(torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), Blur(),)
            torch.nn.Sequential(PixelShuffleUpsampleBlock(out_filters), Blur(),)
            if upsample
            else None
        )

    def forward(self, x, prev_rgb, istyle):
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if prev_rgb is not None:
            x = x + prev_rgb

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class Conv2DModulated(torch.nn.Module):
    """
    Modulated convolution.
    For details refer to [1]
    [1] Karras et. al. - Analyzing and Improving the Image Quality of StyleGAN (https://arxiv.org/abs/1912.04958)
    """

    def __init__(
        self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps=1e-8, **kwargs,
    ):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = torch.nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        torch.nn.init.kaiming_normal_(self.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x


class GeneratorBlock(torch.nn.Module):
    def __init__(
        self, ind, latent_dim, input_channels, filters, upsample=True, upsample_rgb=True, channels=1, attention=True
    ):
        super().__init__()
        self.use_attention = attention
        if ind == 0:
            embed_dims = 10
        elif ind == 1:
            embed_dims = 20
        elif ind == 2:
            embed_dims = 40
        else:  # ind == 3 or ind == 4
            embed_dims = 80
        if self.use_attention:
            self.attention = MultiOrderGatedAggregation1D(embed_dims=embed_dims) 
        # if self.use_attention:
        #     self.attention = LinearAttention(filters)
        # print("value of filters", input_channels, filters)
        # self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) if upsample else None
        # self.upsample = PixelShuffleUpsampleBlock(input_channels) if upsample else None
        self.upsample = EUCB(input_channels, input_channels) if upsample else None
        self.to_style1 = torch.nn.Linear(latent_dim, input_channels)
        self.to_noise1 = torch.nn.Linear(1, filters)
        self.conv1 = Conv2DModulated(input_channels, filters, 3)

        self.to_style2 = torch.nn.Linear(latent_dim, filters)
        self.to_noise2 = torch.nn.Linear(1, filters)
        self.conv2 = Conv2DModulated(filters, filters, 3)

        self.activation = torch.nn.LeakyReLU(0.2, inplace=True)
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, channels)

    def forward(self, x, prev_rgb, istyle, inoise):
        # print('shape of prev_rgb', x.shape, prev_rgb.shape, istyle.shape, inoise.shape)
        if self.upsample is not None:
            x = self.upsample(x)
            
        inoise = inoise[:, : x.shape[2], : x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 1, 2))
        noise2 = self.to_noise2(inoise).permute((0, 3, 1, 2))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)
        # print('shape of attention x', x.shape)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        # print('shape of rgb x', rgb.shape, x.shape)
        if self.use_attention:
            rgb = self.attention(rgb.squeeze(1)).unsqueeze(1)
            # rgb = rgb + rgb_moga
        return x, rgb


class DiscriminatorBlock(torch.nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = torch.nn.Conv2d(input_channels, filters, 1, stride=(2 if downsample else 1))

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, filters, 3, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(filters, filters, 3, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )

        self.downsample = (
            torch.nn.Sequential(Blur(), torch.nn.Conv2d(filters, filters, 3, padding=1, stride=2))
            if downsample
            else None
        )

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x


class Generator(torch.nn.Module):
    def __init__(
        self, n_bands, latent_dim, style_depth, network_capacity=16, channels=1, fmap_max=512, start_from_zero=True
    ):
        super().__init__()
        self.image_size = n_bands
        self.latent_dim = latent_dim
        self.num_layers = int(log2(n_bands) - 1)
        self.style_depth = style_depth

        self.style_mapping = ExtendedMappingNetwork(z_dim=self.latent_dim, num_layers=self.style_depth, lr_multiplier=0.1)

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])
        # print("len of pairs", len(in_out_pairs))
        self.initial_conv = torch.nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = torch.nn.ModuleList([])
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs): # 0, 1, 2, 3, 4
            not_first = ind != 0
            use_attention = ind > 1
            not_last = ind != (self.num_layers - 1)
            block = GeneratorBlock(
                ind, latent_dim, in_chan, out_chan, upsample=not_first, upsample_rgb=not_last, channels=channels, attention = use_attention
            )
            self.blocks.append(block)
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
        for block in self.blocks:
            torch.nn.init.zeros_(block.to_noise1.weight)
            torch.nn.init.zeros_(block.to_noise1.bias)
            torch.nn.init.zeros_(block.to_noise2.weight)
            torch.nn.init.zeros_(block.to_noise2.bias)

        initial_block_size = n_bands // self.upsample_factor, 1
        self.initial_block = torch.nn.Parameter(
            torch.randn((1, init_channels, *initial_block_size)), requires_grad=False
        )
        if start_from_zero:
            self.initial_block.data.zero_()

    def add_scaled_condition(self, target: torch.Tensor, condition: torch.Tensor, condition_lengths: torch.Tensor):
        *_, target_height, _ = target.shape
        *_, height, _ = condition.shape

        scale = height // target_height

        # scale appropriately
        condition = F.interpolate(condition, size=target.shape[-2:], mode="bilinear")

        # add and mask
        result = (target + condition) / 2
        result = mask_sequence_tensor(result, (condition_lengths / scale).ceil().long())

        return result

    @property
    def upsample_factor(self):
        return 2 ** sum(1 for block in self.blocks if block.upsample)

    def forward(self, condition: torch.Tensor, lengths: torch.Tensor, ws: List[torch.Tensor], noise: torch.Tensor):
        batch_size, _, _, max_length = condition.shape
        x = self.initial_block.expand(batch_size, -1, -1, max_length // self.upsample_factor)

        rgb = None
        x = self.initial_conv(x)

        for style, block in zip(ws, self.blocks):
            x, rgb = block(x, rgb, style, noise)
            # print('shap of x and rgb', x.shape, rgb.shape)
            x = self.add_scaled_condition(x, condition, lengths)
            rgb = self.add_scaled_condition(rgb, condition, lengths)

        return rgb


class Discriminator(torch.nn.Module):
    def __init__(
        self, n_bands, network_capacity=16, channels=1, fmap_max=512,
    ):
        super().__init__()
        num_layers = int(log2(n_bands) - 1)
        num_init_filters = channels

        blocks = []
        filters = [num_init_filters] + [(network_capacity * 4) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last)
            blocks.append(block)

        self.blocks = torch.nn.ModuleList(blocks)

        channel_last = filters[-1]
        latent_dim = channel_last

        self.final_conv = torch.nn.Conv2d(channel_last, channel_last, 3, padding=1)
        self.to_logit = torch.nn.Linear(latent_dim, 1)

        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x, condition: torch.Tensor, lengths: torch.Tensor):
        for block in self.blocks:
            x = block(x)
            scale = condition.shape[-1] // x.shape[-1]
            x = mask_sequence_tensor(x, (lengths / scale).ceil().long())

        x = self.final_conv(x)

        scale = condition.shape[-1] // x.shape[-1]
        x = mask_sequence_tensor(x, (lengths / scale).ceil().long())

        x = x.mean(axis=-2)
        x = (x / rearrange(lengths / scale, "b -> b 1 1")).sum(axis=-1)
        x = self.to_logit(x)
        return x.squeeze()
