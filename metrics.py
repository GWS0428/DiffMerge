# clip_metrics.py

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.multimodal import clip_score
from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
import numpy as np

clip_id = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_id)
text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to("cuda")
image_processor = CLIPImageProcessor.from_pretrained(clip_id)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to("cuda")


class DirectionalSimilarity(nn.Module):
    def __init__(self, tokenizer, text_encoder, image_processor, image_encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to("cuda")}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.to("cuda")}

    def encode_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode_text(self, text):
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_directional_similarity(self, img_feat_one, img_feat_two, text_feat_one, text_feat_two):
        sim_direction = F.cosine_similarity(img_feat_two - img_feat_one, text_feat_two - text_feat_one)
        return sim_direction

    def forward(self, image_one, image_two, caption_one, caption_two):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)
        directional_similarity = self.compute_directional_similarity(
            img_feat_one, img_feat_two, text_feat_one, text_feat_two
        )
        return directional_similarity


# Partial CLIP Score function
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def calculate_clip_score(images, prompts):
    """
    Calculate CLIP score between images and prompts.

    Args:
        images (numpy.ndarray): (N, H, W, C), values in [0, 1]
        prompts (list of str): Prompts for each image

    Returns:
        float: Average CLIP score
    """
    images_int = (images * 255).astype("uint8")
    tensor_images = torch.from_numpy(images_int).permute(0, 3, 1, 2)
    score = clip_score_fn(tensor_images, prompts).detach()
    return round(float(score), 4)


def calculate_directional_similarity(input_images, edited_images, original_captions, modified_captions):
    """
    Calculate average CLIP directional similarity (CDS) score.

    Args:
        input_images: list of original PIL images
        edited_images: list of edited PIL images
        original_captions: list of str
        modified_captions: list of str

    Returns:
        float: Average directional similarity score
    """
    dir_similarity = DirectionalSimilarity(tokenizer, text_encoder, image_processor, image_encoder)
    scores = []
    for i in range(len(input_images)):
        original_image = input_images[i]
        original_caption = original_captions[i]
        edited_image = edited_images[i]
        modified_caption = modified_captions[i]

        similarity_score = dir_similarity(original_image, edited_image, original_caption, modified_caption)
        scores.append(float(similarity_score.detach().cpu()))

    return np.mean(scores)
