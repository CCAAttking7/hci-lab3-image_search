import clip
import torch
import numpy as np
import json
import os
from PIL import Image
from pathlib import Path
from upstash_vector import Index
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)
index = Index(
    url=os.environ["UPSTASH_VECTOR_REST_URL"],
    token=os.environ["UPSTASH_VECTOR_REST_TOKEN"],
)


def encode_and_upload(img_dir: str, batch_size=50, save_local=True):
    img_paths = sorted(Path(img_dir).glob("*.jpg"))
    print(f"Found {len(img_paths)} images in {img_dir}")

    all_vecs, all_ids, all_meta = [], [], []
    vectors_to_upsert = []

    for img_path in tqdm(img_paths, desc="Encoding & uploading"):
        try:
            img = (
                preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                feat = model.encode_image(img)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            vec = feat.cpu().numpy()[0].tolist()

            record = {
                "id": img_path.stem,
                "vector": vec,
                "metadata": {
                    "filename": img_path.name,
                    "path": str(img_path.resolve()),
                },
            }
            vectors_to_upsert.append(record)
            all_vecs.append(vec)
            all_ids.append(img_path.stem)
            all_meta.append(record["metadata"])

        except Exception as e:
            print(f"Skip {img_path.name}: {e}")

        if len(vectors_to_upsert) >= batch_size:
            index.upsert(vectors_to_upsert)
            vectors_to_upsert = []

    if vectors_to_upsert:
        index.upsert(vectors_to_upsert)

    if save_local:
        Path("cache").mkdir(exist_ok=True)
        np.save("cache/vectors.npy", np.array(all_vecs))
        with open("cache/ids.json", "w") as f:
            json.dump({"ids": all_ids, "meta": all_meta}, f)
        print("Saved to cache/")

    print(f"✅ Done: {len(all_vecs)} images uploaded to Upstash")


if __name__ == "__main__":
    encode_and_upload("data/subset")
