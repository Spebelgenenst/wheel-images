# Example of using ALIGN for image-text similarity
from transformers import AlignProcessor, AlignModel
import torch
from PIL import Image
import requests
from io import BytesIO
import time

# Load processor and model
processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
model = AlignModel.from_pretrained("kakaobrain/align-base")

# Download image from URL
url = "https://huggingface.co/roschmid/dog-races/resolve/main/images/Golden_Retriever.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))  # Convert the downloaded bytes to a PIL Image

texts = ["a photo of a cat", "a photo of a dog"]

# Process image and text inputs
inputs = processor(images=image, text=texts, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    outputs = model(**inputs)


image_embeds = outputs.image_embeds
text_embeds = outputs.text_embeds

# Normalize embeddings for cosine similarity
image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

print(image_embeds)
print(text_embeds)

quit()

start_time = time.time()
# Calculate similarity scores
for i in range(0,16000000):
    similarity_scores = torch.matmul(text_embeds, image_embeds.T)

print(time.time()-start_time)

# Print raw scores
print("Similarity scores:", similarity_scores)

# Convert to probabilities
probs = torch.nn.functional.softmax(similarity_scores, dim=0)
print("Probabilities:", probs)

# Get the most similar text
most_similar_idx = similarity_scores.argmax().item()
print(f"Most similar text: '{texts[most_similar_idx]}'")
