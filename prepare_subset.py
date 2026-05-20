# prepare_subset.py
from datasets import load_dataset
from pathlib import Path
import requests
from tqdm import tqdm

ds = load_dataset("phiyodr/coco2017", split="validation[:200]")

out_dir = Path("data/subset")
out_dir.mkdir(parents=True, exist_ok=True)

failed = 0
for i, item in enumerate(tqdm(ds)):
    url = item["coco_url"]
    img_id = item["image_id"]
    out_path = out_dir / f"{img_id}.jpg"

    if out_path.exists():
        continue

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
    except Exception as e:
        print(f"Failed {url}: {e}")
        failed += 1

print(f"Done: {200 - failed} images saved to data/subset/  ({failed} failed)")
