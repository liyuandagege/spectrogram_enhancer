# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# The following is largely based on code from https://github.com/lucidrains/stylegan2-pytorch

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import grad as torch_grad
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from torchaudio import transforms
from nemo.collections.tts.parts.utils.helpers import mask_sequence_tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from nemo.collections.tts.losses.context_loss import CSFlow
def spherical_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x * y).sum(-1).arccos().pow(2)


class TimeFrequencyLoss(torch.nn.Module):
    """
    Time-Frequency Loss to ensure the generated spectrogram is similar to the real spectrogram in both time and frequency domains.
    """

    def __init__(self, weight: float = 5.0):
        super().__init__()
        self.weight = weight

    def __call__(self, enhanced, target):
        return self.weight * F.mse_loss(enhanced, target)

class PerceptualLoss(torch.nn.Module):
    """
    Perceptual Loss to capture higher-level context information using a pre-trained network.
    """

    def __init__(self, feature_extractor, layers, weight: float = 1.0):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.layers = layers
        self.weight = weight

    def __call__(self, enhanced, target):
        enhanced_features = self._extract_features(enhanced)
        target_features = self._extract_features(target)
        loss = 0
        for ef, tf in zip(enhanced_features, target_features):
            loss += F.l1_loss(ef, tf)
        return self.weight * loss

    def _extract_features(self, x):
        features = []
        for name, layer in self.feature_extractor._modules.items():
            x = layer(x)
            if name in self.layers:
                features.append(x)
        return features

class ContextualLoss(nn.Module):
    """
    Contextual Loss to ensure the generated spectrogram captures the context of the real spectrogram.
    """

    def __init__(self, weight: float = 1.0, h: float = 0.1):
        super(ContextualLoss, self).__init__()
        self.weight = weight
        self.h = h  # Bandwidth parameter

    def forward(self, enhanced, target):
        # Assuming enhanced and target are [B, C, H, W]
        target = target.unsqueeze(1)
        # print('shape of enhanced, target', enhanced.shape, target.shape)
        assert enhanced.size() == target.size(), "Input and target must have the same size"
        
        # Normalize the enhanced and target features
        enhanced_norm = F.normalize(enhanced, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)

        # Compute cosine similarity
        cosine_sim = torch.einsum('bchw,bchw->bhw', [enhanced_norm, target_norm])
        
        # Compute contextual loss
        d = (1 - cosine_sim) / 2
        d_min = d.min(dim=-1, keepdim=True)[0]
        d_tilde = d / (d_min + 1e-5)
        w = torch.exp((1 - d_tilde) / self.h)
        contextual_loss = -torch.log(w.mean(dim=-1) + 1e-5).mean()

        return self.weight * contextual_loss

class SpectrogramPerceptualLoss(nn.Module):
    def __init__(self, loss_scale=1.0, layers_weights=None):
        super(SpectrogramPerceptualLoss, self).__init__()
        self.loss_scale = loss_scale
        vgg = models.vgg19(pretrained=True).features
        self.vgg_layers = nn.ModuleList([vgg[i] for i in range(len(vgg))])
        if layers_weights is None:
            self.layers_weights = [0.0625, 0.125, 0.25, 0.5, 1.0]
            # 示例权重，调整以匹配任务需要
            # self.layers_weights = [0.0625, 0.125, 0.25, 0.5, 1.0]
            # self.layers_weights = [0.1, 0.2, 0.3, 0.2, 0.2]
            # self.layers_weights = [0.1, 0.125, 0.25, 0.5, 1.0]
        else:
            self.layers_weights = layers_weights
    def forward(self, spect_predicted, spect_tgt):
        # 确保输入的频谱图具有正确的范围和形状
        spect_predicted = spect_predicted.repeat(1, 3, 1, 1)  # (B, C, H, W)
        spect_tgt = spect_tgt.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, C, H, W)
        # 标准化频谱图以匹配 VGG19 的输入范围
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(spect_predicted.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(spect_predicted.device)
        spect_predicted = (spect_predicted - mean) / std
        spect_tgt = (spect_tgt - mean) / std
        loss = 0.0
        for i, weight in enumerate(self.layers_weights):
            with torch.no_grad():
                # print("spect_tgt before", spect_tgt.shape)

                spect_tgt = self.vgg_layers[i](spect_tgt)
                # print("spect_tgt after", spect_tgt.shape)
                # spect_tgt = self.adjust_channels[i](spect_tgt)  # 调整通道数
            spect_predicted = self.vgg_layers[i](spect_predicted)
            # spect_predicted = self.adjust_channels[i](spect_predicted)  # 调整通道数
            loss += weight * F.l1_loss(spect_predicted, spect_tgt)

        return loss * self.loss_scale

class SpeechLoss(nn.Module):
    def __init__(self, kernel_size, stride):
        super(SpeechLoss, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.mse_loss = nn.L1Loss() #nn.MSELoss()
        # self.filter_length = filter_length
        # self.hop_length = hop_length
        # self.ri_stft = REALSTFT(filter_length=filter_length, hop_length=hop_length).cuda()

    def forward(self, enhanced_audio, target_audio):
        # real_lab, imag_lab = self.ri_stft.transform(target_audio[1].cuda())
        # real_lab, imag_lab = real_lab.permute(0, 2, 1).contiguous(), imag_lab.permute(0, 2, 1).contiguous()
        # target_audio = torch.stack((real_lab, imag_lab), dim=3).permute(0, 3, 1, 2)
        # real_est, imag_est = self.ri_stft.transform(enhanced_audio.cuda())
        # real_est, imag_est = real_est.permute(0, 2, 1).contiguous(), imag_est.permute(0, 2, 1).contiguous()
        # enhanced_audio = torch.stack((real_est, imag_est), dim=3).permute(0, 3, 1, 2)

        batch_size, num_channels, time, frequency = enhanced_audio.shape

        enhanced_audio = (enhanced_audio - enhanced_audio.mean()) / enhanced_audio.std()
        target_audio = (target_audio - target_audio.mean()) / target_audio.std()
        # 计算时间维度填充大小
        time_padding = self.kernel_size - (time % self.kernel_size)
        if time_padding == self.kernel_size:
            time_padding = 0

        # 计算频率维度填充大小
        frequency_padding = self.kernel_size - (frequency % self.kernel_size)
        if frequency_padding == self.kernel_size:
            frequency_padding = 0

        # 填充增强语音和目标语音
        enhanced_padded = F.pad(enhanced_audio, (0, time_padding, 0, frequency_padding))
        target_padded = F.pad(target_audio, (0, time_padding, 0, frequency_padding))

        # 将填充后的增强语音和目标语音切分成小块
        unfolded_enhanced = enhanced_padded.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        unfolded_target = target_padded.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        # 计算小块之间的相关性
        correlation = self.compute_correlation(unfolded_enhanced)
        correlation_target = self.compute_correlation(unfolded_target)

        # 计算相关性损失
        correlation_loss = self.mse_loss(correlation, correlation_target)

        # 综合损失
        loss = correlation_loss * 0.01
        #print(correlation_loss)
        return loss

    def compute_correlation(self, x):
        #print(x.shape)
        batch,channels,time_patchnumber,fre_patchnumber,patchheigth,patchweith = x.shape
        correlation_x = x.reshape(batch,channels,time_patchnumber,fre_patchnumber,-1)
        correlation_x_1 = x.reshape(batch,channels,-1,patchweith*patchheigth)
        correlation_x_2 = x.reshape(batch,channels,patchweith*patchheigth,-1)
        correlation_metrix = correlation_x_1.matmul(correlation_x_2)
        #print(correlation_metrix.shape)
        return correlation_metrix



def contextual_loss(enhanced, target, h=0.5):
    # Compute cosine similarity
    cosine_sim = torch.einsum('bchw,bchw->bhw', [enhanced, target])
    
    # Compute contextual loss
    d = (1 - cosine_sim) / 2
    d_min = d.min(dim=-1, keepdim=True)[0]
    d_tilde = d / (d_min + 1e-5)
    w = torch.exp((1 - d_tilde) / h)
    
    # # Apply frequency weighting
    # freq_weight = torch.linspace(1.0, 0.5, enhanced.size(2), device=enhanced.device)
    # w *= freq_weight.view(1, 1, -1, 1)  # Assuming frequency is on the third dimension

    # Calculate final loss
    contextual_loss = -torch.log(w.mean(dim=-1) + 1e-5).mean()

    return contextual_loss

class PercepContxLoss(nn.Module):
    def __init__(self, loss_scale=1.0, layers_weights=None):
        super(PercepContxLoss, self).__init__()
        self.loss_scale = loss_scale
        vgg = models.vgg19(pretrained=True).features
        self.vgg_layers = nn.ModuleList([vgg[i] for i in range(len(vgg))])
        if layers_weights is None:
            self.layers_weights = [0.0625, 0.125, 0.25, 0.5, 1.0]
            # 示例权重，调整以匹配任务需要
        else:
            self.layers_weights = layers_weights
    def forward(self, spect_predicted, spect_tgt):
        # 确保输入的频谱图具有正确的范围和形状
        spect_predicted = spect_predicted.repeat(1, 3, 1, 1)  # (B, C, H, W)
        spect_tgt = spect_tgt.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, C, H, W)
        # 标准化频谱图以匹配 VGG19 的输入范围
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(spect_predicted.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(spect_predicted.device)
        spect_predicted = (spect_predicted - mean) / std
        spect_tgt = (spect_tgt - mean) / std
        loss = 0.0
        for i, weight in enumerate(self.layers_weights):
            with torch.no_grad():
                spect_tgt = self.vgg_layers[i](spect_tgt)
            spect_predicted = self.vgg_layers[i](spect_predicted)
            # print('shape of spect_predicted', spect_predicted.shape, spect_tgt.shape)
            f1 = F.l1_loss(spect_predicted, spect_tgt)
            con = contextual_loss(spect_predicted, spect_tgt)
            # print('value of distance', con, f1)
            loss_for = weight * (con + f1)
            loss = loss + loss_for
        return loss * self.loss_scale

# spect_predicted = torch.randn(1, 1, 100, 64)
# spect_tgt = torch.randn(1, 100, 64)
# loss_module = PercepContxLoss()
# print("Loss:", loss_module(spect_predicted, spect_tgt).item())


# class FeatureContextLoss(nn.Module):
#     def __init__(self, loss_scale=1.0, layers_weights=None):
#         super(FeatureContextLoss, self).__init__()
#         self.loss_scale = loss_scale
#         vgg = models.vgg19(pretrained=True).features
#         self.vgg_layers = nn.ModuleList([vgg[i] for i in range(len(vgg))])
#         if layers_weights is None:
#             self.layers_weights = [0.0625, 0.125, 0.25, 0.5, 1.0]
#             # 示例权重，调整以匹配任务需要
#             # self.layers_weights = [0.0625, 0.125, 0.25, 0.5, 1.0]
#             # self.layers_weights = [0.1, 0.2, 0.3, 0.2, 0.2]
#             # self.layers_weights = [0.1, 0.125, 0.25, 0.5, 1.0]
#         else:
#             self.layers_weights = layers_weights
#     def forward(self, spect_predicted, spect_tgt):
#         # 确保输入的频谱图具有正确的范围和形状
#         spect_predicted = spect_predicted.repeat(1, 3, 1, 1)  # (B, C, H, W)
#         spect_tgt = spect_tgt.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, C, H, W)
#         # 标准化频谱图以匹配 VGG19 的输入范围
#         mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(spect_predicted.device)
#         std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(spect_predicted.device)
#         spect_predicted = (spect_predicted - mean) / std
#         spect_tgt = (spect_tgt - mean) / std
#         loss = 0.0
#         for i, weight in enumerate(self.layers_weights):
#             with torch.no_grad():
#                 # print("spect_tgt before", spect_tgt.shape)

#                 spect_tgt = self.vgg_layers[i](spect_tgt)
#                 # print("spect_tgt after", spect_tgt.shape)
#                 # spect_tgt = self.adjust_channels[i](spect_tgt)  # 调整通道数
#             spect_predicted = self.vgg_layers[i](spect_predicted)
#             # spect_predicted = self.adjust_channels[i](spect_predicted)  # 调整通道数
#             # loss += weight * F.l1_loss(spect_predicted, spect_tgt)
#             loss += weight * CX_loss(spect_predicted, spect_tgt)

#         return loss * self.loss_scale

class SpectrogramPerceptualLossFreq(nn.Module):
    def __init__(self, loss_scale=1.0, layers_weights=None, bands=None):
        super(SpectrogramPerceptualLossFreq, self).__init__()
        self.loss_scale = loss_scale
        vgg = models.vgg19(pretrained=True).features
        self.vgg_layers = nn.ModuleList([vgg[i] for i in range(len(vgg))])
        if layers_weights is None:
            # 示例权重，调整以匹配任务需要
            self.layers_weights = [0.0625, 0.125, 0.25, 0.5, 1.0]
        else:
            self.layers_weights = layers_weights
        if bands is None:
            # 默认将频谱图分成两部分：低频和高频
            self.bands = [(0, 40), (40, 80)]  # 这是假设频谱图有80个频带
        else:
            self.bands = bands

    def forward(self, spect_predicted, spect_tgt):
        loss = 0.0
        for band in self.bands:
            band_start, band_end = band
            # print('shape of spec_predicted_band', spect_predicted.shape)
            # print('shape of spec_predicted_band', spect_tgt.shape)
            spect_predicted_band = spect_predicted[:, :, band_start:band_end, :]
            spect_tgt_band = spect_tgt.unsqueeze(1)[:, :, band_start:band_end, :]
            spect_predicted_band = spect_predicted_band.repeat(1, 3, 1, 1)  # (B, 3, H, W)
            spect_tgt_band = spect_tgt_band.repeat(1, 3, 1, 1)  # (B, 3, H, W)

            # 标准化频谱图以匹配 VGG19 的输入范围
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(spect_predicted.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(spect_predicted.device)
            spect_predicted_band = (spect_predicted_band - mean) / std
            spect_tgt_band = (spect_tgt_band - mean) / std
            # print('after of spec_predicted_band', spect_predicted_band.shape)
            # print('after of spec_predicted_band', spect_tgt_band.shape)
            
            for i, weight in enumerate(self.layers_weights):
                with torch.no_grad():
                    spect_tgt_band = self.vgg_layers[i](spect_tgt_band)
                spect_predicted_band = self.vgg_layers[i](spect_predicted_band)
                loss += weight * F.l1_loss(spect_tgt_band, spect_predicted_band)

        return loss * self.loss_scale


class SpectrogramPerceptualLossWav(nn.Module):
    def __init__(self, loss_scale=1.0, layers_weights=None, bands=None):
        super(SpectrogramPerceptualLossWav, self).__init__()
        self.loss_scale = loss_scale
        self.processor = Wav2Vec2Processor.from_pretrained("/Data/LiYuan/Wetts/fastspeech2/memo/premodel/wav2vec")
        self.wav2vec = Wav2Vec2Model.from_pretrained("/Data/LiYuan/Wetts/fastspeech2/memo/premodel/wav2vec")
        self.wav2vec.eval()  # Set to evaluation mode
        
        if layers_weights is None:
            # 示例权重，调整以匹配任务需要
            self.layers_weights = [1.0]  # 因为 wav2vec 是一个整体模型，没有分层权重
        else:
            self.layers_weights = layers_weights
        
        if bands is None:
            # 默认将频谱图分成两部分：低频和高频
            self.bands = [(0, 40), (40, 80)]  # 这是假设频谱图有80个频带
        else:
            self.bands = bands
    def melspec_to_waveform_griffinlim(self, melspec, sample_rate=22050, n_mel_channels=80, n_window_size=158, n_window_stride=79, n_iter=32):
        """
        将梅尔频谱图转换为波形，使用 Griffin-Lim 算法。

        Args:
            melspec (Tensor): 输入的梅尔频谱图，形状为 (batch_size, n_mel_channels, time_steps)
            sample_rate (int): 采样率，默认值为 22050
            n_mel_channels (int): 梅尔频带数，默认值为 80
            n_window_size (int): 窗口大小，默认值为 1024
            n_window_stride (int): 窗口滑动步长，默认值为 256
            n_iter (int): Griffin-Lim 算法的迭代次数，默认值为 32

        Returns:
            waveform (Tensor): 重建的波形，形状为 (batch_size, num_samples)
        """
        # if melspec.ndim == 3:
        #     melspec = melspec.unsqueeze(1)  # 添加通道维度以适应 Griffin-Lim 期望的输入格式 (batch_size, 1, n_mel_channels, time_steps)

        # Griffin-Lim 转换器
        griffinlim = transforms.GriffinLim(n_fft=n_window_size, win_length=n_window_size, hop_length=n_window_stride, n_iter=n_iter).to(melspec.device)

        # 使用 Griffin-Lim 算法重建波形
        waveform = griffinlim(melspec)
        return waveform
    def forward(self, spect_predicted, spect_tgt):
        loss = 0.0
        # for band in self.bands:
        #     band_start, band_end = band
        #     spect_predicted_band = spect_predicted[:, :, band_start:band_end, :]
        #     spect_tgt_band = spect_tgt[:, :, band_start:band_end, :]
        # 将频谱图转换为波形
        # print('shape of predict, tgt', spect_predicted.shape, spect_tgt.shape)
        waveform_predicted = self.melspec_to_waveform_griffinlim(spect_predicted)
        waveform_tgt = self.melspec_to_waveform_griffinlim(spect_tgt.squeeze(1))
        # 使用 wav2vec 模型提取特征
        # Ensure the waveforms are leaf tensors
        # waveform_tgt = waveform_tgt.detach().requires_grad_()
        with torch.no_grad():
            spect_tgt_features = self.wav2vec(waveform_tgt.squeeze(1).squeeze(1)).last_hidden_state
        waveform_predicted = waveform_predicted.squeeze(1).squeeze(1).detach()
        spect_predicted_features = self.wav2vec(waveform_predicted).last_hidden_state #.detach()
        loss += self.layers_weights[0] * F.l1_loss(spect_predicted_features, spect_tgt_features)
        return loss * self.loss_scale
def test_spectrogram_perceptual_loss_wav():
    # 创建随机的频谱图数据
    batch_size = 2
    channels = 1
    freq_bins = 80
    time_steps = 100
    spect_predicted = torch.rand(batch_size, channels, freq_bins, time_steps, requires_grad=True)
    spect_tgt = torch.rand(batch_size, channels, freq_bins, time_steps, requires_grad=False)

    # 初始化损失函数
    loss_fn = SpectrogramPerceptualLossWav()

    # 计算损失
    loss = loss_fn(spect_predicted, spect_tgt)

    # 打印损失值
    print(f"Calculated loss: {loss.item()}")

# 运行测试用例
test_spectrogram_perceptual_loss_wav()
class GradientPenaltyLoss(torch.nn.Module):
    """
    R1 loss from [1], used following [2]
    [1] Mescheder et. al. - Which Training Methods for GANs do actually Converge? 2018, https://arxiv.org/abs/1801.04406
    [2] Karras et. al. - A Style-Based Generator Architecture for Generative Adversarial Networks, 2018 (https://arxiv.org/abs/1812.04948)
    """

    def __init__(self, weight: float = 10.0):
        super().__init__()
        self.weight = weight

    def __call__(self, images, output):
        batch_size, *_ = images.shape
        gradients = torch_grad(
            outputs=output,
            inputs=images,
            grad_outputs=torch.ones(output.size(), device=images.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.reshape(batch_size, -1)
        return self.weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


class GeneratorLoss(torch.nn.Module):
    def __call__(self, fake_logits):
        return fake_logits.mean()


class HingeLoss(torch.nn.Module):
    def __call__(self, real_logits, fake_logits):
        return (F.relu(1 + real_logits) + F.relu(1 - fake_logits)).mean()


class ConsistencyLoss(torch.nn.Module):
    """
    Loss to keep SpectrogramEnhancer from generating extra sounds.
    L1 distance on x0.25 Mel scale (20 bins for typical 80-bin scale)
    """

    def __init__(self, weight: float = 10):
        super().__init__()
        self.weight = weight

    def __call__(self, condition, output, lengths):
        *_, w, h = condition.shape
        w, h = w // 4, h

        condition = F.interpolate(condition, size=(w, h), mode="bilinear", antialias=True)
        output = F.interpolate(output, size=(w, h), mode="bilinear", antialias=True)

        dist = (condition - output).abs()
        dist = mask_sequence_tensor(dist, lengths)
        return (dist / rearrange(lengths, "b -> b 1 1 1")).sum(dim=-1).mean() * self.weight

from transformers import BertTokenizer, BertModel

class BertTextEncoder(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased'):
        super(BertTextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    
    def forward(self, text_list):
        encoding = self.tokenizer(text_list, return_tensors='pt', padding=True, truncation=True, max_length=128)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        return embeddings

class TextImageAlignmentLoss(nn.Module):
    def __init__(self, text_encoder, weight: float = 10.0):
        super(TextImageAlignmentLoss, self).__init__()
        self.text_encoder = text_encoder
        self.weight = weight

    @staticmethod
    def spherical_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return (x * y).sum(-1).arccos().pow(2)

    def forward(self, text_list, generated_images):
        text_embeddings = self.text_encoder(text_list).to(generated_images.device)
        image_embeddings = generated_images.mean(dim=(2, 3))  # Example: Global average pooling of images
        alignment_loss = self.spherical_distance(image_embeddings, text_embeddings).mean()
        return self.weight * alignment_loss