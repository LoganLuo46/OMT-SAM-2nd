import torch
import torch.nn as nn
from PIL import Image
from open_clip import create_model_from_pretrained
import torchvision.transforms as T


# Define cross modal attention layer
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, img_features, text_features):
        attn_output, attn_weights = self.cross_attention(img_features, text_features, text_features)
        return attn_output, attn_weights


# clip altered with adding cross-attention
class ModifiedCLIPModel(nn.Module):
    def __init__(self, clip_model, embed_dim, num_heads):
        super(ModifiedCLIPModel, self).__init__()
        self.clip_model = clip_model
        self.cross_attention = MultiHeadCrossAttention(embed_dim, num_heads)
        self.image_projection = nn.Linear(512, embed_dim)
        self.text_projection = nn.Linear(512, embed_dim)

    def forward(self, images, text_input):
        # Obtain image and text features
        image_features = self.clip_model.visual(images)  # Obtain image feature
        text_features = self.clip_model.encode_text(text_input)  # Obtain text feature
        image_features = image_features.expand_as(text_features)

        image_features = self.image_projection(image_features)

        text_features = self.text_projection(text_features)

        attn_output, attn_weights = self.cross_attention(image_features, text_features)

        return attn_output


def create_modified_clip_model():
    clip_model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    embed_dim = 256
    num_heads = 8

    clip_model, clip_preprocess = create_model_from_pretrained(clip_model_name)

    modified_clip_model = ModifiedCLIPModel(clip_model, embed_dim, num_heads)

    for param in clip_model.parameters():
        param.requires_grad = False

    return modified_clip_model


# Extract feature embedding after cross-modal attention processing
def get_clip_embeddings(images, text_inputs) -> torch.Tensor:
    clip_model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    embed_dim = 256
    num_heads = 8

    clip_model, clip_preprocess = create_model_from_pretrained(clip_model_name)

    modified_clip_model = ModifiedCLIPModel(clip_model, embed_dim, num_heads)

    with torch.no_grad():
        processed_images = []
        for image in images:
            if isinstance(image, torch.Tensor):
                image = T.ToPILImage()(image)
            processed_images.append(clip_preprocess(image).unsqueeze(0))

        clip_inputs = torch.cat(processed_images, dim=0)

    attn_output = modified_clip_model(clip_inputs, text_inputs)

    clip_image_embeddings = attn_output
    clip_prompt_embeddings = clip_image_embeddings.view(clip_image_embeddings.size(0), 1, embed_dim)

    return clip_prompt_embeddings





