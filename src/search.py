import clip
import torch
import os
from PIL import Image
from upstash_vector import Index
from dotenv import load_dotenv

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
index = Index(
    url=os.environ["UPSTASH_VECTOR_REST_URL"],
    token=os.environ["UPSTASH_VECTOR_REST_TOKEN"],
)


def search_by_text(query: str, top_k: int = 12, score_threshold: float = 0.0):
    tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    vec = feat.cpu().numpy()[0].tolist()
    results = index.query(vector=vec, top_k=top_k, include_metadata=True)
    return [(r.metadata, r.score) for r in results if r.score >= score_threshold]


def search_by_image(image: Image.Image, top_k: int = 12, score_threshold: float = 0.0):
    img_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img_tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    vec = feat.cpu().numpy()[0].tolist()
    results = index.query(vector=vec, top_k=top_k, include_metadata=True)
    return [(r.metadata, r.score) for r in results if r.score >= score_threshold]
