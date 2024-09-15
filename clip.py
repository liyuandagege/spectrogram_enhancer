import torch
import torch.nn as nn
from torchvision.transforms import Normalize
import torch.nn.functional as F
import open_clip
from typing import List
from timm.data import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


class CLIP(nn.Module):
    def __init__(self, name='ViT-B/32', pretrained='openai'):
        super().__init__()
        self.model = open_clip.create_model(name, pretrained=pretrained)
        self.model = self.model.eval().requires_grad_(False)
        self.img_resolution = self.model.visual.image_size[0]
        self.norm = Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
        self.im_dim = self.txt_dim = self.model.ln_final.normalized_shape[0]

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def encode_image(self, images: torch.Tensor, div255: bool = False) -> torch.Tensor:
        if div255:
            images = images.to(torch.float32) / 255.
        images = F.interpolate(images, self.img_resolution, mode='bicubic', align_corners=False)
        images = self.norm(images)
        image_features = self.model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        return image_features

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        max_length = 77  # Maximum length for CLIP model
        text_embeddings = []

        for text in texts:
            # Split the text into segments of max_length
            text = text.rstrip()

            segments = [text[i:i + max_length] for i in range(0, len(text), max_length)]
            segment_embeddings = []

            for segment in segments:
                # Tokenize each segment
                tokenized_segment = open_clip.tokenize([segment]).to(self.device)

                # Encode each tokenized segment
                with torch.no_grad():
                    segment_embedding = self.model.encode_text(tokenized_segment)
                segment_embeddings.append(segment_embedding)

            # Aggregate the embeddings of all segments of the text
            text_embedding = torch.mean(torch.stack(segment_embeddings), dim=0)  # Take the mean, resulting in [1, embedding_dim]
            text_embeddings.append(text_embedding)

        # Aggregate embeddings of all texts
        text_embeddings = torch.cat(text_embeddings, dim=0)

        # Normalize the embeddings
        text_features = F.normalize(text_embeddings, dim=-1)

        return text_features


    def forward(self, images: torch.Tensor, texts: List[str], div255: bool = False) -> torch.Tensor:
        assert len(images) == len(texts)
        image_features = self.encode_image(images, div255=div255)
        text_features = self.encode_text(texts)
        joint_features = torch.cat([image_features, text_features], 1)
        return joint_features
