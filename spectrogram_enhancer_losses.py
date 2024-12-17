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
            # self.layers_weights = [0.0625, 0.125, 0.25, 0.5, 1.0]
            # self.layers_weights = [0.1, 0.2, 0.3, 0.2, 0.2]
            # self.layers_weights = [0.1, 0.125, 0.25, 0.5, 1.0]
        else:
            self.layers_weights = layers_weights
    def forward(self, spect_predicted, spect_tgt):
        spect_predicted = spect_predicted.repeat(1, 3, 1, 1)  # (B, C, H, W)
        spect_tgt = spect_tgt.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, C, H, W)
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
                # spect_tgt = self.adjust_channels[i](spect_tgt)  
            spect_predicted = self.vgg_layers[i](spect_predicted)
            # spect_predicted = self.adjust_channels[i](spect_predicted)  
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
        time_padding = self.kernel_size - (time % self.kernel_size)
        if time_padding == self.kernel_size:
            time_padding = 0

        frequency_padding = self.kernel_size - (frequency % self.kernel_size)
        if frequency_padding == self.kernel_size:
            frequency_padding = 0

        enhanced_padded = F.pad(enhanced_audio, (0, time_padding, 0, frequency_padding))
        target_padded = F.pad(target_audio, (0, time_padding, 0, frequency_padding))
        unfolded_enhanced = enhanced_padded.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        unfolded_target = target_padded.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        correlation = self.compute_correlation(unfolded_enhanced)
        correlation_target = self.compute_correlation(unfolded_target)

        correlation_loss = self.mse_loss(correlation, correlation_target)

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
      
        else:
            self.layers_weights = layers_weights
    def forward(self, spect_predicted, spect_tgt):
        spect_predicted = spect_predicted.repeat(1, 3, 1, 1)  # (B, C, H, W)
        spect_tgt = spect_tgt.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, C, H, W)
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
