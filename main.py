
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from urllib.parse import urlparse
import ipaddress
import socket
import os
import searcher
import uvicorn

app = FastAPI(title="GROSHOP AI — Image Search", version="1.0.0")

API_KEY = os.getenv("AI_API_KEY")

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_TOP_K = 50

ALLOWED_ORIGINS = [
    "http://localhost:8000",
    # Add your production Django/frontend URL here later.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST"],
    allow_headers=["*"],
)

class ImageUrlRequest(BaseModel):
    image_url: HttpUrl
    top_k: int = 10

def verify_api_key(x_api_key: str | None):
    if not API_KEY:
        raise RuntimeError("AI_API_KEY is not configured")

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )

def validate_image_url(url: str):
    parsed = urlparse(url)

    if parsed.scheme != "https":
        raise HTTPException(
            status_code=400,
            detail="Only HTTPS URLs are allowed"
        )

    if not parsed.hostname:
        raise HTTPException(
            status_code=400,
            detail="Invalid URL"
        )

    hostname = parsed.hostname

    try:
        addresses = socket.getaddrinfo(
            hostname,
            None,
            proto=socket.IPPROTO_TCP
        )

        for address in addresses:
            ip = address[4][0]
            ip_obj = ipaddress.ip_address(ip)

            if (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_reserved
                or ip_obj.is_multicast
            ):
                raise HTTPException(
                    status_code=400,
                    detail="URL points to a restricted address"
                )

    except socket.gaierror:
        raise HTTPException(
            status_code=400,
            detail="Could not resolve URL"
        )

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "groshop-ai"
    }

@app.post("/search-by-image-url")
def search_url(
    request: ImageUrlRequest,
    x_api_key: str | None = Header(default=None)
):
    verify_api_key(x_api_key)

    if request.top_k < 1 or request.top_k > MAX_TOP_K:
        raise HTTPException(
            status_code=400,
            detail=f"top_k must be between 1 and {MAX_TOP_K}"
        )

    validate_image_url(str(request.image_url))

    try:
        results = searcher.search_by_image_url(
            str(request.image_url),
            request.top_k
        )

        return {"results": results}

    except Exception:
        import logging
        logging.exception("Image URL search failed")

        raise HTTPException(
            status_code=500,
            detail="Image search failed"
        )

@app.post("/search-by-image-upload")
async def search_upload(
    file: UploadFile = File(...),
    top_k: int = 10,
    x_api_key: str | None = Header(default=None)
):
    verify_api_key(x_api_key)

    if top_k < 1 or top_k > MAX_TOP_K:
        raise HTTPException(
            status_code=400,
            detail=f"top_k must be between 1 and {MAX_TOP_K}"
        )
    
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type"
        )

    try:
        image_bytes = await file.read(MAX_UPLOAD_SIZE + 1)

        if len(image_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image file is too large"
            )

        results = searcher.search_by_image_file(
            image_bytes,
            top_k
        )

        return {"results": results}

    except HTTPException:
        raise

    except Exception:
        import logging
        logging.exception("Image upload search failed")

        raise HTTPException(
            status_code=500,
            detail="Image search failed"
        )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001
    )