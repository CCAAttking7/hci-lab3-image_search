import gradio as gr
from PIL import Image
import time
import zipfile
import os
from pathlib import Path
from search import search_by_text, search_by_image

import urllib.request
import urllib.parse
import json

# ── suggestion corpus fallback ────────────────────────────────────────────────
FALLBACK_SUGGESTIONS = [
    "a dog running in the snow",
    "cats playing together",
    "a modern living room",
    "beautiful sunset over the ocean",
    "people walking on a busy street",
    "a table full of fresh fruits",
    "cars passing by at night",
    "a cup of coffee on a wooden desk",
]


def _get_name(src: str) -> str:
    return src.split("/")[-1].split("\\")[-1]


# ── global state ──────────────────────────────────────────────────────────────
favorites: list[str] = []
search_history: list[str] = []
_picking_suggestion = False  # guard flag

# ── translation & dynamic suggestion helpers ─────────────────────────────────


def translate_zh_to_en(text: str) -> str:
    """If text contains Chinese, translate to English for CLIP."""
    if not text.strip() or not any("\u4e00" <= char <= "\u9fff" for char in text):
        return text
    try:
        url = f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(text)}&langpair=zh-CN|en"
        # Dummy User-Agent to avoid blocks
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            return data["responseData"]["translatedText"]
    except Exception:
        return text


def fetch_dynamic_suggestions(query: str) -> list[str]:
    """Fetch live autocomplete suggestions from Bing API (Works better in China without proxy)."""
    if not query.strip():
        return []
    try:
        url = f"https://api.bing.com/osjson.aspx?query={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            res = json.loads(resp.read().decode())
            return res[1][:6]
    except Exception:
        return [s for s in FALLBACK_SUGGESTIONS if query.lower() in s.lower()][:6]


def get_suggestions(text: str) -> list[str]:
    text = text.strip()
    if not text:
        # History
        hist = [f"🕐  {h}" for h in reversed(search_history[-5:])]
        popular = [f"🌟 {s}" for s in FALLBACK_SUGGESTIONS[: (6 - len(hist))]]
        return hist + popular

    # Otherwise fetch live from Bing or fallback
    return fetch_dynamic_suggestions(text)


def update_suggestions(text):
    """Called on text_in.change — only show dropdown when user is actually typing."""
    global _picking_suggestion
    if _picking_suggestion:
        # a suggestion was just picked; suppress re-render and reset guard
        _picking_suggestion = False
        return gr.update(visible=False)
    suggestions = get_suggestions(text or "")
    if not suggestions:
        return gr.update(visible=False)
    # Always show if we got valid suggestions (either history/fallback or API)
    return gr.update(choices=suggestions, value=None, visible=True)


def pick_suggestion(choice):
    """Fill text box from suggestion, hide dropdown, set guard."""
    global _picking_suggestion
    if not choice:
        return gr.update(), gr.update(visible=False)
    _picking_suggestion = True
    if choice.startswith("🕐  "):
        choice = choice[5:].strip()
    return gr.update(value=choice), gr.update(visible=False)


# ── search ────────────────────────────────────────────────────────────────────


def _run(q_type, q_val, top_k, min_score):
    start = time.time()
    if q_type == "text":
        results = search_by_text(q_val, top_k=int(top_k), score_threshold=min_score)
    else:
        results = search_by_image(q_val, top_k=int(top_k), score_threshold=min_score)
    elapsed = time.time() - start
    gallery = []

    for meta, score in results:
        # 优先用 coco_url（Kaggle上传的），没有就用本地path（子集用的）
        src = meta.get("coco_url") or meta.get("path", "")

        # 兜底：如果元数据只有 image_id（当时批量上传漏传了完整URL），动态拼接COCO网络路径
        if not src and "image_id" in meta:
            img_id = int(meta["image_id"])
            src = f"http://images.cocodataset.org/val2017/{img_id:012d}.jpg"

        if src and (src.startswith("http") or os.path.exists(src)):
            gallery.append((src, f"{score:.3f}"))
    return gallery, f"{len(gallery)} results · {elapsed:.2f}s"


def go_text(query, top_k, min_score):
    if not query or not query.strip():
        return (
            [],
            "No query yet.",
            "—",
            gr.update(visible=False),
            {"type": None, "val": None},
            [],
        )
    q = query.strip()
    if q not in search_history:
        search_history.append(q)

    # Translate if there's Chinese
    translated_q = translate_zh_to_en(q)

    gal, stat = _run("text", translated_q, top_k, min_score)

    # Prepend translation status if it was translated
    if translated_q != q:
        stat = f"*(Translated to: {translated_q})*  |  {stat}"

    return (
        gal,
        stat,
        f'"{q}"',
        gr.update(visible=False),
        {"type": "text", "val": translated_q},
        gal,
    )


def go_image(img, top_k, min_score):
    if img is None:
        return (
            [],
            "No image uploaded.",
            "—",
            gr.update(visible=False),
            {"type": None, "val": None},
            [],
        )
    gal, stat = _run("image", img, top_k, min_score)
    return (
        gal,
        stat,
        "Image query",
        gr.update(value=img, visible=True),
        {"type": "image", "val": img},
        gal,
    )


def refine(state, top_k, min_score):
    if not state or not state.get("type"):
        return gr.update(), gr.update(), []
    gal, stat = _run(state["type"], state["val"], top_k, min_score)
    return gal, stat, gal


# ── preview & favorites ───────────────────────────────────────────────────────


def open_preview(evt: gr.SelectData, results_state):
    if results_state and evt.index < len(results_state):
        path = results_state[evt.index][0]
        return path
    return None


def reveal_preview(path):
    if path:
        is_fav = path in favorites
        btn_text = "❤️ Saved" if is_fav else "♡  Save to favorites"
        return (
            gr.update(visible=True),
            f'<img src="{path}" style="width:140px; height:140px; object-fit:cover; border-radius:8px;" />',
            gr.update(value=btn_text, visible=True),
            _get_name(path),
        )
    return (
        gr.update(visible=False),
        "",
        gr.update(visible=False),
        "",
    )


def _render_fav_html():
    html = '<div style="display:flex; gap:6px; overflow-x:auto; min-height:60px; padding:2px;">'
    for p in favorites:
        html += f'<img src="{p}" style="height:60px; width:60px; object-fit:cover; border-radius:4px; border:1px solid var(--border-color-primary); flex-shrink:0;" />'
    html += "</div>"
    return html


def toggle_fav(current_path):
    if not current_path:
        return (
            gr.update(),
            _render_fav_html(),
            f"{len(favorites)} saved",
        )

    if current_path in favorites:
        favorites.remove(current_path)
        btn_text = "♡  Save to favorites"
    else:
        favorites.append(current_path)
        btn_text = "❤️ Saved"

    return gr.update(value=btn_text), _render_fav_html(), f"{len(favorites)} saved"


def clear_fav():
    favorites.clear()
    return gr.update(value="♡  Save to favorites"), _render_fav_html(), "0 saved"


def export_fav():
    if not favorites:
        return None
    zip_path = "favorites.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in favorites:
            if p.startswith("http"):
                req = urllib.request.Request(p, headers={"User-Agent": "Mozilla/5.0"})
                try:
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        zf.writestr(_get_name(p), resp.read())
                except Exception:
                    pass
            elif os.path.exists(p):
                zf.write(p, _get_name(p))
    return zip_path


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
/* hide share/flag/download icons inside gallery toolbars */
.gallery-container .icon-button-wrapper,
.gallery-container .download-button,
.gallery-container .share-button,
.gallery-container [data-testid="share-btn"],
.gallery-container [data-testid="download-btn"],
.gallery-container [aria-label="Share"],
.gallery-container [aria-label="Download"],
.gallery-container [aria-label="Expand"],
.toolbar { display: none !important; }
footer { display: none !important; }

.gradio-container { max-width: 1100px !important; }

/* app shell */
#app-row { gap: 0 !important; }
#sidebar {
    border-right: 0.5px solid var(--border-color-primary);
    padding: 20px 16px !important;
    background: var(--background-fill-secondary) !important;
    min-width: 268px !important;
    max-width: 268px !important;
}

/* logo */
#logo { padding-bottom: 12px; border-bottom: 0.5px solid var(--border-color-primary); margin-bottom: 2px; }
#logo h1 { font-size: 15px !important; font-weight: 600; margin: 0 !important; }
#logo p  { font-size: 11px; color: var(--body-text-color-subdued); margin: 2px 0 0 !important; }

/* tabs compact */
#mode-tabs .tab-nav { padding: 3px; background: var(--background-fill-primary);
    border-radius: 8px; border: 0.5px solid var(--border-color-primary); }
#mode-tabs .tab-nav button { font-size: 12px !important; padding: 5px 10px !important; border-radius: 6px !important; }
#mode-tabs > div { padding: 0 !important; border: none !important; background: transparent !important; }

/* query textbox */
#text-query textarea { font-size: 13px !important; border-radius: 8px !important; }
#text-query label span { font-size: 11px !important; font-weight: 500; letter-spacing: .04em; }

/* suggestion dropdown — flush under textbox, no gap */
#suggestion-box {
    border: 0.5px solid var(--border-color-primary);
    border-radius: 8px;
    overflow: hidden;
    background: var(--background-fill-primary);
    margin-top: 2px;
    padding: 4px 0;
}
#suggestion-box .wrap { gap: 0 !important; flex-direction: column !important; }
#suggestion-box label { 
    padding: 7px 12px !important; 
    font-size: 13px !important; 
    cursor: pointer;
    border-radius: 0 !important;
    margin: 0 !important;
}
#suggestion-box label:hover { background: var(--background-fill-secondary) !important; }
#suggestion-box input[type=radio] { display: none !important; }
#suggestion-box .svelte-1gfkn6j { display: none !important; }

/* sliders */
.param-slider label span { font-size: 11px !important; }

/* main column */
#main-col { padding: 0 !important; }

/* top bar */
#topbar {
    padding: 9px 16px;
    border-bottom: 0.5px solid var(--border-color-primary);
    background: var(--background-fill-primary);
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 40px;
}
#query-pill {
    background: var(--background-fill-secondary);
    border: 0.5px solid var(--border-color-primary);
    border-radius: 99px;
    padding: 3px 12px;
    font-size: 12px;
    max-width: 360px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
#result-stat { font-size: 12px; color: var(--body-text-color-subdued); }

/* gallery */
#result-gallery { padding: 12px 16px !important; }
#result-gallery .grid-wrap { gap: 7px !important; }
#result-gallery img { border-radius: 7px !important; }

/* preview panel */
#preview-row {
    margin: 0 16px 10px;
    padding: 12px 16px;
    border: 0.5px solid var(--border-color-primary);
    border-radius: 10px;
    background: var(--background-fill-secondary);
}
#preview-row > div { gap: 14px !important; align-items: center !important; }
#preview-img img { border-radius: 8px !important; object-fit: cover !important; }
#selected-name { font-size: 12px; color: var(--body-text-color-subdued); margin: 0 !important; }
#save-btn { font-size: 12px !important; border-radius: 7px !important; }

/* bottom bar */
#bottom-bar {
    border-top: 0.5px solid var(--border-color-primary);
    padding: 8px 16px;
    align-items: center !important;
    gap: 10px !important;
}
#fav-count-md { font-size: 12px; color: var(--body-text-color-subdued); white-space: nowrap; margin: 0 !important; }
#fav-gallery { flex: 1; min-height: 40px !important; }
#fav-gallery .grid-wrap { gap: 4px !important; }
#fav-gallery img { border-radius: 5px !important; }
#dl-btn, #clear-btn { font-size: 12px !important; border-radius: 7px !important; white-space: nowrap; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="HCI_lab3 · Visual Search", css=CSS) as demo:
    last_query = gr.State({"type": None, "val": None})
    sel_path = gr.State(None)
    current_results = gr.State([])

    with gr.Row(elem_id="app-row", equal_height=True):
        # ══════════════ SIDEBAR ══════════════
        with gr.Column(elem_id="sidebar", scale=0):
            gr.HTML(
                '<div id="logo"><h1>HCI_lab3</h1><p>Semantic image search</p></div>'
            )

            with gr.Tabs(elem_id="mode-tabs"):
                with gr.Tab("Text"):
                    text_in = gr.Textbox(
                        placeholder="e.g. a dog running in snow…",
                        label="Query",
                        lines=2,
                        elem_id="text-query",
                    )
                    suggestion_box = gr.Radio(
                        choices=[],
                        label="",
                        visible=False,
                        elem_id="suggestion-box",
                    )
                    btn_text = gr.Button("Search", variant="primary")

                with gr.Tab("Image"):
                    img_in = gr.Image(
                        label="Upload query image",
                        type="pil",
                        height=160,
                    )
                    img_preview_thumb = gr.Image(
                        interactive=False,
                        show_label=False,
                        visible=False,
                        height=56,
                    )
                    btn_img = gr.Button("Search", variant="primary")

            gr.HTML(
                '<hr style="border:none;border-top:0.5px solid var(--border-color-primary);margin:10px 0">'
            )

            top_k = gr.Slider(
                4, 48, value=12, step=4, label="Results", elem_classes="param-slider"
            )
            min_score = gr.Slider(
                0.0,
                1.0,
                value=0.20,
                step=0.05,
                label="Min similarity",
                elem_classes="param-slider",
            )

        # ══════════════ MAIN ══════════════
        with gr.Column(elem_id="main-col", scale=1):
            # top bar
            with gr.Row(elem_id="topbar"):
                query_pill_md = gr.Markdown("—", elem_id="query-pill")
                result_stat = gr.Markdown("", elem_id="result-stat")

            # results
            result_gallery = gr.Gallery(
                label="",
                columns=4,
                height=390,
                allow_preview=False,
                show_label=False,
                elem_id="result-gallery",
            )

            # preview panel (hidden until image clicked)
            with gr.Row(elem_id="preview-row", visible=False) as preview_row:
                sel_img_display = gr.HTML(
                    elem_id="preview-img",
                )
                with gr.Column(scale=1):
                    sel_name_md = gr.Markdown("", elem_id="selected-name")
                    save_btn = gr.Button(
                        "♡  Save to favorites",
                        variant="secondary",
                        elem_id="save-btn",
                        visible=False,
                    )

            # bottom bar
            with gr.Row(elem_id="bottom-bar"):
                fav_count_md = gr.Markdown("0 saved", elem_id="fav-count-md")
                fav_gallery = gr.HTML(
                    value=_render_fav_html(),
                    elem_id="fav-gallery",
                )
                dl_btn = gr.Button("↓ ZIP", variant="secondary", elem_id="dl-btn")
                clear_btn = gr.Button("Clear", variant="stop", elem_id="clear-btn")

            out_file = gr.File(label="Download", visible=True)

    # ── events ────────────────────────────────────────────────────────────────

    # suggestions: only on non-empty input change
    text_in.change(
        fn=update_suggestions,
        inputs=[text_in],
        outputs=[suggestion_box],
    )

    # picking a suggestion fills the box and hides dropdown
    suggestion_box.change(
        fn=pick_suggestion,
        inputs=[suggestion_box],
        outputs=[text_in, suggestion_box],
    )

    # sidebar image thumb preview
    img_in.change(
        fn=lambda img: (
            gr.update(value=img, visible=True) if img else gr.update(visible=False)
        ),
        inputs=[img_in],
        outputs=[img_preview_thumb],
    )

    # text search
    btn_text.click(
        fn=go_text,
        inputs=[text_in, top_k, min_score],
        outputs=[
            result_gallery,
            result_stat,
            query_pill_md,
            preview_row,
            last_query,
            current_results,
        ],
    )

    text_in.submit(
        fn=go_text,
        inputs=[text_in, top_k, min_score],
        outputs=[
            result_gallery,
            result_stat,
            query_pill_md,
            preview_row,
            last_query,
            current_results,
        ],
    )

    btn_img.click(
        fn=go_image,
        inputs=[img_in, top_k, min_score],
        outputs=[
            result_gallery,
            result_stat,
            query_pill_md,
            preview_row,
            last_query,
            current_results,
        ],
    )

    # refinement
    top_k.change(
        fn=refine,
        inputs=[last_query, top_k, min_score],
        outputs=[result_gallery, result_stat, current_results],
    )
    min_score.change(
        fn=refine,
        inputs=[last_query, top_k, min_score],
        outputs=[result_gallery, result_stat, current_results],
    )

    # click image → preview panel
    result_gallery.select(
        fn=open_preview,
        inputs=[current_results],
        outputs=[sel_path],
    )
    sel_path.change(
        fn=reveal_preview,
        inputs=[sel_path],
        outputs=[preview_row, sel_img_display, save_btn, sel_name_md],
    )

    # favorites
    save_btn.click(
        fn=toggle_fav, inputs=[sel_path], outputs=[save_btn, fav_gallery, fav_count_md]
    )
    dl_btn.click(fn=export_fav, outputs=[out_file])
    clear_btn.click(fn=clear_fav, outputs=[save_btn, fav_gallery, fav_count_md])


if __name__ == "__main__":
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    demo.launch(server_port=7860, share=False, show_error=True)
