import requests
import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel

processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
model = AlignModel.from_pretrained("kakaobrain/align-base")

# image embeddings
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    image_embeds = model.get_image_features(
        pixel_values=inputs['pixel_values'],
    )

# text embeddings
text = "an image of a cat"
inputs = processor(text=text, return_tensors="pt")

with torch.no_grad():
    text_embeds = model.get_text_features(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        token_type_ids=inputs['token_type_ids'],
    )


#print(text_embeds)
print(image_embeds)