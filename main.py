from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import searcher # On importe la logique de recherche qu'on va créer juste après
import uvicorn

app = FastAPI(title="GROSHOP AI — Image Search", version="1.0.0")

# Autorise ton backend Django à appeler cette API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["POST"],
    allow_headers=["*"],
)

class ImageUrlRequest(BaseModel):
    image_url: str
    top_k: int = 10

@app.get("/health")
def health():
    return {"status": "ok", "service": "groshop-ai"}

@app.post("/search-by-image-url")
def search_url(request: ImageUrlRequest):
    """Recherche par URL d'image."""
    try:
        results = searcher.search_by_image_url(request.image_url, request.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-by-image-upload")
async def search_upload(file: UploadFile = File(...), top_k: int = 10):
    """Recherche par upload direct."""
    try:
        image_bytes = await file.read()
        results = searcher.search_by_image_file(image_bytes, top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)