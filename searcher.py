import clip
import torch
import faiss
import numpy as np
import json
import requests
from PIL import Image
from io import BytesIO

# Chargement du modèle et de l'index au démarrage
print("Chargement CLIP + index FAISS...")
model, preprocess = clip.load("ViT-B/32", device="cpu")
model.eval()
index = faiss.read_index("products.index")
with open("product_ids.json") as f:
    product_ids = json.load(f)

def get_embedding(image):
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().astype("float32")

def search_by_image_url(image_url: str, top_k: int = 10):
    response = requests.get(image_url, timeout=10)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    query = get_embedding(image)
    scores, indices = index.search(query, top_k)
    return [{"product_id": product_ids[idx], "score": float(scores[0][i])} 
            for i, idx in enumerate(indices[0]) if idx != -1]

def search_by_image_file(image_bytes: bytes, top_k: int = 10):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    query = get_embedding(image)
    scores, indices = index.search(query, top_k)
    return [{"product_id": product_ids[idx], "score": float(scores[0][i])} 
            for i, idx in enumerate(indices[0]) if idx != -1]