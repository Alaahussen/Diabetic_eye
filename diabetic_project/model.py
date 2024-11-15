# model.py
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

class ImageClassifier:
    def __init__(self, model_name="AlaaHussien/dinov2-base-finetuned-eye"):
        # Load the processor and model for the specified model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)

    def predict(self, image_path):
        # Open the image and convert it to RGB format
        image = Image.open(image_path).convert("RGB")
        
        # Preprocess the image using the processor
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Perform inference without gradient calculations (faster)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the prediction by finding the index of the highest logit value
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        
        # Map the predicted index to the label
        return self.model.config.id2label[predicted_class_idx]

# Instantiate the classifier
classifier = ImageClassifier()
