# Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo.collections.tts.models.aligner import AlignerModel
from nemo.collections.tts.models.fastpitch import FastPitchModel
from nemo.collections.tts.models.fastpitch_ssl import FastPitchModel_SSL
from nemo.collections.tts.models.hifigan import HifiGanModel
from nemo.collections.tts.models.mixer_tts import MixerTTSModel
from nemo.collections.tts.models.radtts import RadTTSModel
from nemo.collections.tts.models.spectrogram_enhancer import SpectrogramEnhancerModel
from nemo.collections.tts.models.spectrogram_enhancer_per import SpectrogramEnhancerModelPer
from nemo.collections.tts.models.spectrogram_enhancer_clip import SpectrogramEnhancerModelClip
from nemo.collections.tts.models.spectrogram_enhancer_clip_per import SpectrogramEnhancerModelClipPer
from nemo.collections.tts.models.spectrogram_enhancer_feature import SpectrogramEnhancerModelFeature
from nemo.collections.tts.models.spectrogram_enhancer_blur_losses import SpectrogramEnhancerModelBloss
from nemo.collections.tts.models.spectrogram_enhancer_wsm import SpectrogramEnhancerModelWSM
from nemo.collections.tts.models.se_bert_layer import SpectrogramEnhancerModelBLayer
from nemo.collections.tts.models.ssl_tts import SSLDisentangler
from nemo.collections.tts.models.tacotron2 import Tacotron2Model
from nemo.collections.tts.models.two_stages import GriffinLimModel, MelPsuedoInverseModel, TwoStagesModel
from nemo.collections.tts.models.univnet import UnivNetModel
from nemo.collections.tts.models.vits import VitsModel
from nemo.collections.tts.models.waveglow import WaveGlowModel

__all__ = [
    "AlignerModel",
    "FastPitchModel",
    "FastPitchModel_SSL",
    "SSLDisentangler",
    "GriffinLimModel",
    "HifiGanModel",
    "MelPsuedoInverseModel",
    "MixerTTSModel",
    "RadTTSModel",
    "Tacotron2Model",
    "TwoStagesModel",
    "UnivNetModel",
    "VitsModel",
    "WaveGlowModel",
    "SpectrogramEnhancerModel",
    "SpectrogramEnhancerModelClip",
    "SpectrogramEnhancerModelPer",
    "SpectrogramEnhancerModelClipPer",
    "SpectrogramEnhancerModelBloss",
    "SpectrogramEnhancerModelFeature",
    "SpectrogramEnhancerModelBLayer",
    "SpectrogramEnhancerModelWSM"
]
