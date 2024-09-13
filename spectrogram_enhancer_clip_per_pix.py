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
    if arr is None: return False
    is_list = isinstance(arr, list) or isinstance(arr, np.ndarray) or  isinstance(arr, tuple)
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
        self.mlp = MLP([z_dim]*(num_layers+1), activation=activation,
                       lr_multiplier=lr_multiplier, linear_out=True)

        if conditional:
            self.clip = CLIP()
            del self.clip.model.visual # only using the text encoder
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

        # 前向传播
        x = self.mlp(F.normalize(z, dim=1))
                # 检查 x 是否包含 inf 或 NaN
        self.x_avg = self.x_avg.to(x.dtype) 
        # 更新移动平均值
        
        if self.x_avg_beta is not None and self.training:
            self.x_avg.copy_(x.detach().mean(0).lerp(self.x_avg, self.x_avg_beta))

        # 应用截断
        if truncation_psi != 1:
            assert self.x_avg_beta is not None
            x = self.x_avg.lerp(x, truncation_psi)  # 较低的 truncation_psi 值会使生成样本更加接近 x_avg，从而减少多样性但提高样本质量。
        # 构建潜在向量
        if self.c_dim > 0:
            assert c is not None
            c = self.clip.encode_text(c) if is_list_of_strings(c) else c
            w = torch.cat([x, c], 1)
            w = self.fusion_layer(w)
        else:
            w = x
        # 广播潜在向量
        return w

