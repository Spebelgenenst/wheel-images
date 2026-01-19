import requests
import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel

class embeds():

    def __init__(self):
        processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
        model = AlignModel.from_pretrained("kakaobrain/align-base")


    def image_embed(self, image_url):
        image = Image.open(requests.get(image_url, stream=True).raw)

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            image_embeds = model.get_image_features(
                pixel_values=inputs['pixel_values'],
            )

        return image_embeds


    def text_embed(self, text):
        inputs = processor(text=text, return_tensors="pt")

        with torch.no_grad():
            text_embeds = model.get_text_features(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids'],
            )

embeds()