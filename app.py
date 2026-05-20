"""
app.py — Gradio 图像搜索界面
Five-Stage Search Framework 实现
"""

import gradio as gr
from PIL import Image
import time
import zipfile
import os
import io
import base64
from pathlib import Path
from search import search_by_text, search_by_image  # 你写的 search.py

# ─────────────────────────────────────────────
# 全局状态
# ─────────────────────────────────────────────
favorites: list[str] = []  # 收藏的图片路径列表

# ─────────────────────────────────────────────
# 核心搜索函数
# ─────────────────────────────────────────────


def do_text_search(query: str, top_k: int, min_score: float):
    """文字搜索 → 返回结果供 Gallery 显示"""
    if not query.strip():
        return [], "⚠️ Please enter a search query.", query

    start = time.time()
    results = search_by_text(query, top_k=top_k, score_threshold=min_score)
    elapsed = time.time() - start

    # 构建 Gallery 数据：(image, caption)
    gallery_data = []
    for meta, score in results:
        img_path = meta.get("path", "")
        if os.path.exists(img_path):
            caption = f"Score: {score:.3f} | {Path(img_path).name}"
            gallery_data.append((img_path, caption))

    status = (
        f"✅ Found **{len(gallery_data)}** results · Query time: **{elapsed:.2f}s**"
    )
    preview_text = f'🔍 Query: "{query}"'
    return gallery_data, status, preview_text


def do_image_search(image: Image.Image, top_k: int, min_score: float):
    """图片搜索 → 返回结果供 Gallery 显示"""
    if image is None:
        return [], "⚠️ Please upload an image.", "No image uploaded"

    start = time.time()
    results = search_by_image(image, top_k=top_k, score_threshold=min_score)
    elapsed = time.time() - start

    gallery_data = []
    for meta, score in results:
        img_path = meta.get("path", "")
        if os.path.exists(img_path):
            caption = f"Score: {score:.3f} | {Path(img_path).name}"
            gallery_data.append((img_path, caption))

    status = (
        f"✅ Found **{len(gallery_data)}** results · Query time: **{elapsed:.2f}s**"
    )
    preview_text = "🖼️ Query: [Uploaded Image]"
    return gallery_data, status, preview_text


def add_to_favorites(evt: gr.SelectData, gallery_data):
    """从 Gallery 点击图片加入收藏"""
    global favorites
    if evt.index < len(gallery_data):
        img_path = gallery_data[evt.index][0]
        if img_path not in favorites:
            favorites.append(img_path)
    fav_gallery = [(p, Path(p).name) for p in favorites]
    return fav_gallery, f"❤️ {len(favorites)} image(s) in favorites"


def export_favorites_zip():
    """将收藏图片打包为 ZIP 下载"""
    if not favorites:
        return None
    zip_path = "/tmp/favorites.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in favorites:
            zf.write(p, Path(p).name)
    return zip_path


def clear_favorites():
    """清空收藏"""
    global favorites
    favorites = []
    return [], "❤️ 0 image(s) in favorites"


# ─────────────────────────────────────────────
# Gradio 界面定义
# ─────────────────────────────────────────────

CSS = """
/* 整体背景 */
body { background-color: #0f1117 !important; }
.gradio-container { max-width: 1400px !important; font-family: 'Segoe UI', system-ui; }

/* 标题区 */
.app-header { text-align: center; padding: 24px 0 8px; }
.app-header h1 { font-size: 2.4rem; font-weight: 700; color: #e8eaf6; letter-spacing: -0.5px; }
.app-header p { color: #9e9eb3; font-size: 0.95rem; }

/* 卡片面板 */
.panel-card { background: #1a1d2e; border-radius: 16px; padding: 20px; border: 1px solid #2a2d3e; }

/* 状态栏 */
.status-bar { 
    background: #1e2235; border-radius: 10px; padding: 10px 16px;
    font-size: 0.9rem; color: #7c85b3; border: 1px solid #2a2d3e;
}

/* 搜索按钮 */
.search-btn { 
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important; color: white !important;
    font-size: 1.05rem !important; font-weight: 600 !important;
    border-radius: 12px !important; padding: 12px 0 !important;
}
.search-btn:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

/* Gallery 图片卡片 */
.gallery-item img { border-radius: 10px; transition: transform 0.2s; }
.gallery-item img:hover { transform: scale(1.03); }
"""

with gr.Blocks(title="Visual Search · CLIP") as demo:
    # ── 标题 ──────────────────────────────────────────
    gr.HTML("""
    <div class="app-header">
        <h1>🔍 Visual Search</h1>
        <p>Semantic Image Retrieval powered by CLIP + Upstash Vector</p>
    </div>
    """)

    # ── 主体：两列布局 ─────────────────────────────────
    with gr.Row():
        # ════════════════════════════
        # LEFT: 输入 Panel
        # ════════════════════════════
        with gr.Column(scale=1, elem_classes="panel-card"):
            gr.Markdown("### 🗂 Search Input")

            # Tabs：文字 / 图片
            with gr.Tabs() as input_tabs:
                # ── Tab 1: 文字搜索 (Formulation) ──
                with gr.TabItem("📝 Text Search"):
                    text_input = gr.Textbox(
                        placeholder="e.g. a dog playing in the park...",
                        label="Search Query",
                        lines=2,
                    )
                    # 查询预览 (Formulation: preview)
                    text_preview = gr.Textbox(
                        label="🔎 Query Preview",
                        interactive=False,
                        value="Type something above to preview your query",
                    )
                    # 实时更新预览
                    text_input.change(
                        fn=lambda t: (
                            f'🔍 Query: "{t}"' if t.strip() else "No query yet"
                        ),
                        inputs=text_input,
                        outputs=text_preview,
                    )
                    text_search_btn = gr.Button(
                        "🔍 Search by Text",
                        variant="primary",
                        elem_classes="search-btn",
                    )

                # ── Tab 2: 图片搜索 (Formulation) ──
                with gr.TabItem("🖼️ Image Search"):
                    image_input = gr.Image(
                        label="Upload Query Image", type="pil", height=220
                    )
                    # 查询预览 (Formulation: image preview)
                    img_preview_status = gr.Textbox(
                        label="🔎 Query Preview",
                        interactive=False,
                        value="Upload an image to preview",
                    )
                    image_input.change(
                        fn=lambda img: (
                            "✅ Image uploaded — ready to search"
                            if img is not None
                            else "No image yet"
                        ),
                        inputs=image_input,
                        outputs=img_preview_status,
                    )
                    image_search_btn = gr.Button(
                        "🔍 Search by Image",
                        variant="primary",
                        elem_classes="search-btn",
                    )

            gr.Markdown("---")
            gr.Markdown("### ⚙️ Search Parameters")

            # 参数调整 (Refinement)
            top_k_slider = gr.Slider(
                minimum=4, maximum=48, value=12, step=4, label="Top-K Results"
            )
            min_score_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.20,
                step=0.05,
                label="Minimum Similarity Score",
            )

        # ════════════════════════════
        # RIGHT: 结果 Panel
        # ════════════════════════════
        with gr.Column(scale=2, elem_classes="panel-card"):
            gr.Markdown("### 📊 Search Results")

            # 结果概览 (Review)
            status_bar = gr.Markdown(
                value="Results will appear here after search.",
                elem_classes="status-bar",
            )

            # 结果 Gallery (Review)
            result_gallery = gr.Gallery(
                label="Retrieved Images",
                columns=3,
                height=420,
                object_fit="cover",
                show_label=False,
                allow_preview=True,  # 点击放大
            )

            gr.Markdown("---")
            gr.Markdown("### ❤️ Favorites")

            # 收藏区 (Use)
            fav_status = gr.Markdown("❤️ 0 image(s) in favorites")
            fav_gallery = gr.Gallery(
                label="Saved Images",
                columns=4,
                height=200,
                object_fit="cover",
                show_label=False,
            )

            with gr.Row():
                export_btn = gr.Button(
                    "⬇️ Download Favorites (ZIP)", variant="secondary"
                )
                clear_fav_btn = gr.Button("🗑 Clear Favorites", variant="stop")

            export_file = gr.File(label="Download", visible=True)

    # ─────────────────────────────────────────────
    # 事件绑定
    # ─────────────────────────────────────────────

    # 文字搜索触发
    text_search_btn.click(
        fn=do_text_search,
        inputs=[text_input, top_k_slider, min_score_slider],
        outputs=[result_gallery, status_bar, text_preview],
    )

    # 图片搜索触发
    image_search_btn.click(
        fn=do_image_search,
        inputs=[image_input, top_k_slider, min_score_slider],
        outputs=[result_gallery, status_bar, img_preview_status],
    )

    # 点击图片 → 加入收藏 (Use)
    result_gallery.select(
        fn=add_to_favorites, inputs=[result_gallery], outputs=[fav_gallery, fav_status]
    )

    # 导出收藏 ZIP
    export_btn.click(fn=export_favorites_zip, outputs=export_file)

    # 清空收藏
    clear_fav_btn.click(fn=clear_favorites, outputs=[fav_gallery, fav_status])

    # 参数变化时自动重新搜索（Refinement：改参数即刷新）
    # 注意：需要 last_query state 来记住上次查询
    # 此处用简单版：改参数后手动点 Search 按钮即可
    # 进阶版：可加 gr.State 保存上次查询类型和内容

# ─────────────────────────────────────────────
# 启动
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os

    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

    demo.launch(
        server_port=7860,
        share=False,  # 改为 True 可生成公开链接
        show_error=True,
        css=CSS,
    )
