#! 此代码不具体在本地运行，而是配合着在 Kaggle/Colab 上的 GPU 环境使用，主要用于将 COCO 2017 验证集的图片向量化并上传到 Upstash Vector 数据库中。
# 需要提前设置好环境变量 UPSTASH_VECTOR_REST_URL 和 UPSTASH_VECTOR_REST_TOKEN 来存储你的 Upstash Vector 数据库的 URL 和 Token。
# 模型预加载等需要自行在 Kaggle/Colab 上完成，确保 encode_pil 函数可用来将 PIL 图片编码成向量。

import os
import requests
import io
import json
from PIL import Image
from datasets import load_dataset
from upstash_vector import Index

# 💡 Note: If running in Kaggle/Colab, you might use `from tqdm.notebook import tqdm` instead.
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

# ==========================================
# 🚨 Vector DB Credentials
# Read from environment to avoid hardcoding secrets in source code.
# ==========================================
MY_REAL_URL = os.environ.get("UPSTASH_VECTOR_REST_URL", "YOUR_UPSTASH_URL_HERE")
MY_REAL_TOKEN = os.environ.get("UPSTASH_VECTOR_REST_TOKEN", "YOUR_UPSTASH_TOKEN_HERE")

# Clean the strings
MY_REAL_URL = str(MY_REAL_URL).strip()
MY_REAL_TOKEN = str(MY_REAL_TOKEN).strip()

print(f"URL loaded: '{MY_REAL_URL[:20]}...'")

# Initialize Upstash Vector Index client
pure_index = Index(url=MY_REAL_URL, token=MY_REAL_TOKEN)
print("Upstash connection mapped ✅")

# ==========================================
# 开始主程序
# ==========================================
ds = load_dataset("phiyodr/coco2017", split="validation")
print(f"数据集加载完毕: {len(ds)} 张图片")


BATCH_SIZE = 50
vectors_to_upsert = []
uploaded, skipped = 0, 0

for item in tqdm(ds):
    img_id = str(item["image_id"])
    image_download_url = item["coco_url"]

    try:
        # 下载与编码
        resp = requests.get(image_download_url, timeout=15)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        # 注意，这里encode_pill是在之前的代码里定义好的一个函数，负责将PIL图片编码成向量
        """def encode_pil(img):
                t = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    f = model.encode_image(t)
                    f = f / f.norm(dim=-1, keepdim=True)
                return f.cpu().numpy()[0].tolist()"""
        vec = encode_pil(img)  # 这里调用你之前定义好的 encode_pil

        if vec is None:
            skipped += 1
            continue
    except Exception:
        skipped += 1
        continue

    # 构建极简数据
    vectors_to_upsert.append(
        {
            "id": img_id,
            "vector": vec,
            "metadata": {"image_id": img_id, "filename": f"{img_id}.jpg"},
        }
    )

    # 满 50 张上传
    if len(vectors_to_upsert) >= BATCH_SIZE:
        try:
            # 💡 注意：这里使用的是全新实例化的 pure_index！
            pure_index.upsert(vectors_to_upsert)
            uploaded += len(vectors_to_upsert)
            vectors_to_upsert = []
        except Exception as e:
            print("\n💥 【当前Cell全新连接】批量上传居然又失败了！")
            print(f"底层报错详情: {e}")
            print(f"当前 pure_index 的实际内部 URL 属性为: '{pure_index._url}'")
            raise e

# 剩余数据上传
if vectors_to_upsert:
    try:
        pure_index.upsert(vectors_to_upsert)
        uploaded += len(vectors_to_upsert)
    except Exception as e:
        print(f"尾部上传失败: {e}")

print(f"\n🎉 奇迹降临！成功上传: {uploaded}, 跳过: {skipped}")
