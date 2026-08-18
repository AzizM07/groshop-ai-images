# indexer.py
import clip
import torch
import faiss
import numpy as np
import requests
import json
from PIL import Image
from io import BytesIO

# Configuration
DJANGO_API_URL = "http://localhost:8000/api/products/" # Ton URL API

model, preprocess = clip.load("ViT-B/32", device="cpu")
model.eval()

def fetch_products_from_django():
    """Récupère les produits depuis ton API Django avec gestion de pagination."""
    products_list = []
    limit = 100
    offset = 0
    
    while True:
        url = f"{DJANGO_API_URL}?limit={limit}&offset={offset}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Adapte 'data["results"]' selon la structure réelle de ton JSON
        results = data.get("results", [])
        if not results:
            break
            
        products_list.extend(results)
        offset += limit
        print(f"Récupération : {len(products_list)} produits chargés...")
        
    return products_list

def encode_image_from_url(url: str):
    """Télécharge une image depuis l'URL publique Supabase et l'encode."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().astype("float32")

def build_index():
    print("Récupération des produits depuis Django...")
    products = fetch_products_from_django()
    
    dimension = 512
    index = faiss.IndexFlatIP(dimension)
    product_ids = []
    embeddings = []

    for product in products:
        img_url = product.get("primary_image")
        if not img_url:
            continue
            
        try:
            print(f"Encoding {product['id']}...")
            emb = encode_image_from_url(img_url)
            embeddings.append(emb)
            product_ids.append(product["id"])
        except Exception as e:
            print(f"Erreur sur {product['id']}: {e}")

    if embeddings:
        matrix = np.vstack(embeddings)
        index.add(matrix)
        faiss.write_index(index, "products.index")
        with open("product_ids.json", "w") as f:
            json.dump(product_ids, f)
        print(f"\n✅ Index créé avec {len(embeddings)} produits.")

if __name__ == "__main__":
    build_index()