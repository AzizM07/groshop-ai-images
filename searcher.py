import clip
import torch
import faiss
import numpy as np
import json
import requests
from PIL import Image
from io import BytesIO

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
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
    response = requests.get(
        image_url,
        timeout=10,
        stream=True,
        allow_redirects=False,
    )
    if response.is_redirect:
        raise ValueError("Redirects are not allowed for image URLs")

    response.raise_for_status()
    content_length = response.headers.get("Content-Length")

    if content_length:
        try:
            if int(content_length) > MAX_IMAGE_SIZE:
                raise ValueError("Remote image is too large")
        except ValueError:
            raise ValueError("Invalid Content-Length")
        
    image_bytes = bytearray()

    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue

        image_bytes.extend(chunk)

        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise ValueError("Remote image is too large")

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    query = get_embedding(image)
    scores, indices = index.search(query, top_k)

    return [
        {
            "product_id": product_ids[idx],
            "score": float(scores[0][i])
        }
        for i, idx in enumerate(indices[0])
        if idx != -1
    ]


def search_by_image_file(image_bytes: bytes, top_k: int = 10):
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise ValueError("Image file is too large")

    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    query = get_embedding(image)

    scores, indices = index.search(query, top_k)

    return [
        {
            "product_id": product_ids[idx],
            "score": float(scores[0][i])
        }
        for i, idx in enumerate(indices[0])
        if idx != -1
    ]