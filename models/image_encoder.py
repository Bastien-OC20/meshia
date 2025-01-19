from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch

class ImageEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode(self, image_path):
        """
        Encode une image en vecteur de caractéristiques.
        :param image_path: Chemin vers l'image
        :return: Représentation vectorielle de l'image
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features
