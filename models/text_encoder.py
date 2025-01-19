from transformers import CLIPModel, CLIPProcessor
import torch

class TextEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode(self, text):
        """
        Encode un texte en un vecteur de caractéristiques.
        :param text: Liste de chaînes de texte
        :return: Tensor représentant les caractéristiques du texte
        """
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features
