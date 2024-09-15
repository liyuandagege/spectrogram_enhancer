import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
from nemo.collections.tts.modules.shared import MLP

import math
from functools import partial
from math import log2
from typing import List
from einops import rearrange
from kornia.filters import filter2d
from nemo.collections.tts.parts.utils.helpers import mask_sequence_tensor
from nemo.collections.tts.losses.spectrogram_enhancer_losses import PercepContxLoss
from nemo.collections.tts.modules.clip import CLIP
from typing import Union, Any, Optional


def is_list_of_strings(arr: Any) -> bool:
    if arr is None:
        return False
    is_list = isinstance(arr, list) or isinstance(arr, np.ndarray) or isinstance(arr, tuple)
    entry_is_str = isinstance(arr[0], str)
    return is_list and entry_is_str


class PixelShuffleUpsampleBlock(nn.Module):
    def __init__(self, in_channels, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        return self.pixel_shuffle(x)


class ExtendedMappingNetwork(nn.Module):
    def __init__(
        self,
        z_dim: int,
        conditional: bool = False,
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
            self.clip = CLIP()
            del self.clip.model.visual  # Only using the text encoder
            self.c_dim = self.clip.txt_dim
        else:
            self.c_dim = 0

        self.w_dim = self.c_dim + self.z_dim
        self.fusion_layer = nn.Linear(self.w_dim, self.z_dim)
        self.register_buffer('x_avg', torch.zeros([self.z_dim]))

    def forward(
        self,
        z: torch.Tensor,
        c: Union[None, torch.Tensor, List[str]],
        truncation_psi: float = 0.5,
    ) -> torch.Tensor:
        assert z.shape[1] == self.z_dim

        # Forward pass
        x = self.mlp(F.normalize(z, dim=1))
        
        # Check if x contains inf or NaN
        self.x_avg = self.x_avg.to(x.dtype)
        
        # Update moving average
        if self.x_avg_beta is not None and self.training:
            self.x_avg.copy_(x.detach().mean(0).lerp(self.x_avg, self.x_avg_beta))

        # Apply truncation
        if truncation_psi != 1:
            assert self.x_avg_beta is not None
            x = self.x_avg.lerp(x, truncation_psi)  # Lower truncation_psi values make generated samples closer to x_avg, reducing diversity but improving sample quality.
        
        # Build latent vector
        if self.c_dim > 0:
            assert c is not None
            c = self.clip.encode_text(c) if is_list_of_strings(c) else c
            w = torch.cat([x, c], 1)
            w = self.fusion_layer(w)
        else:
            w = x
        
        # Broadcast latent vector
        return w


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


# class StyleMapping(torch.nn.Module):
#     def __init__(self, emb, depth, lr_mul=0.1):
#         super().__init__()

#         layers = []
#         for _ in range(depth):
#             layers.extend([EqualLinear(emb, emb, lr_mul), torch.nn.LeakyReLU(0.2, inplace=True)])

#         self.net = torch.nn.Sequential(*layers)

#     def forward(self, x):
#         x = F.normalize(x, dim=1)
#         return self.net(x)


class RGBBlock(torch.nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, channels=3):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = torch.nn.Linear(latent_dim, input_channel)

        out_filters = channels
        self.conv = Conv2DModulated(input_channel, out_filters, 1, demod=False)

        self.upsample = (
            torch.nn.Sequential(PixelShuffleUpsampleBlock(out_filters), Blur(),)
            # torch.nn.Sequential(torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), Blur(),)
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
        self, latent_dim, input_channels, filters, upsample=True, upsample_rgb=True, channels=1,
    ):
        super().__init__()
        # self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) if upsample else None
        self.upsample = PixelShuffleUpsampleBlock(input_channels) if upsample else None
        self.to_style1 = torch.nn.Linear(latent_dim, input_channels)
        self.to_noise1 = torch.nn.Linear(1, filters)
        self.conv1 = Conv2DModulated(input_channels, filters, 3)

        self.to_style2 = torch.nn.Linear(latent_dim, filters)
        self.to_noise2 = torch.nn.Linear(1, filters)
        self.conv2 = Conv2DModulated(filters, filters, 3)

        self.activation = torch.nn.LeakyReLU(0.2, inplace=True)
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, channels)

    def forward(self, x, prev_rgb, istyle, inoise):
        if self.upsample is not None:
            x = self.upsample(x)

        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 1, 2))
        noise2 = self.to_noise2(inoise).permute((0, 3, 1, 2))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
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
        # print('shape of n_bands, latent_dim, style_depth', n_bands, latent_dim, style_depth)
        super().__init__()
        self.image_size = n_bands
        self.latent_dim = latent_dim
        self.num_layers = int(log2(n_bands) - 1)
        self.style_depth = style_depth
        self.spectrogram_perceptual_loss = PercepContxLoss(loss_scale=1.0)
        self.style_mapping = ExtendedMappingNetwork(z_dim=self.latent_dim, num_layers=self.style_depth, lr_multiplier=0.1)
        # self.style_mapping = TestClass(z_dim=self.latent_dim, num_layers=self.style_depth, lr_multiplier=0.1)
        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])

        self.initial_conv = torch.nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = torch.nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)

            block = GeneratorBlock(
                (latent_dim), in_chan, out_chan, upsample=not_first, upsample_rgb=not_last, channels=channels,
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

        # Scale appropriately
        condition = F.interpolate(condition, size=target.shape[-2:], mode="bilinear")

        # Add and mask
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

            x = self.add_scaled_condition(x, condition, lengths)
            rgb = self.add_scaled_condition(rgb, condition, lengths)

        return rgb


def test_generator():
    n_bands = 32
    latent_dim = 512
    style_depth = 8
    batch_size = 2
    height = 32
    width = 32
    channels = 1

    generator = Generator(n_bands, latent_dim, style_depth)

    # Generate input tensors
    condition = torch.randn(batch_size, channels, height, width)
    lengths = torch.randint(1, width, (batch_size,))
    ws = [torch.randn(batch_size, latent_dim) for _ in range(generator.num_layers)]
    noise = torch.randn(batch_size, channels, height, width)

    # Call the forward method
    output = generator(condition, lengths, ws, noise)

    # Check output shape
    print("Output shape:", output.shape)
    assert output.shape == (batch_size, channels, height, width), "Output shape mismatch"


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
