# 图像搜索系统 · 完整执行文档

> **作业目标**：构建一个满足 Five-Stage Search Framework 的图像搜索可视化页面，支持文字/图片两种检索方式，使用 CLIP + 向量数据库实现语义相似度匹配。  
> **技术栈**：Python · CLIP (OpenAI) · Upstash Vector · Gradio · MS-COCO / Flickr30k

---

## 一、项目总览

### 1.1 系统架构图

```
用户界面 (Gradio)
    │
    ├── 文字输入 ──→ CLIP Text Encoder ──→ 文本向量 (512-dim)
    │                                           │
    └── 图片上传 ──→ CLIP Image Encoder ──→ 图像向量 (512-dim)
                                                │
                                         Upstash Vector DB
                                         (cosine similarity top-K)
                                                │
                                         返回图片 URL + 得分
                                                │
                                    Gradio Gallery 展示 + 下载/收藏
```

### 1.2 分数对照表

| 要求 | 对应设计 | 分值 |
|------|---------|------|
| 文字搜索图像，按相似度排序 | 文字输入框 → CLIP → 向量检索 → Gallery | 2 |
| 图像搜索图像，按相似度排序 | 图片上传 → CLIP → 向量检索 → Gallery | 2 |
| 用户可使用检索结果（下载/收藏） | 下载按钮 + 收藏列表 + 导出功能 | 1 |
| 输入框（文字/图片上传）Formulation | gr.Textbox + gr.Image | 1 |
| 查询预览 Formulation | 实时预览 Panel（文字显示 / 图片缩略图） | 1 |
| 搜索按钮 Initiation | gr.Button("🔍 Search") | 1 |
| 结果概览 Review | "Found X results · Query time: Xs" 状态栏 | 1 |
| 修改搜索参数 Refinement | Top-K 滑块 + 相似度阈值滑块 | 1 |
| 收藏/下载动作 Use | ❤️ 收藏按钮 + ⬇️ 批量下载 ZIP | 1 |

**总计：11 分（全满）**

---

## 二、数据集选择

### 推荐：Flickr30k（更易获取，适合演示）

| 属性 | 详情 |
|------|------|
| 规模 | 31,783 张图片，每张 5 条文字描述 |
| 内容 | 日常生活场景（人物、活动、室外场景） |
| 获取 | HuggingFace `nlphuji/flickr30k` 或官网申请 |
| 授权 | 研究用途免费 |
| 预处理量 | ~3.2 万张，单卡约 2 小时完成 CLIP 向量化 |

**备选：MS-COCO 2017 Val（12 万张，更丰富但更慢）**

```bash
# Flickr30k via HuggingFace
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('nlphuji/flickr30k', split='test')"
```

---

## 三、环境搭建（Step by Step）

### Step 0：创建虚拟环境

```bash
conda create -n imgsearch python=3.10 -y
conda activate imgsearch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/openai/CLIP.git
pip install gradio upstash-vector pillow requests tqdm numpy
```

### Step 1：注册 Upstash Vector

1. 访问 https://upstash.com → 注册免费账号
2. 创建 Vector Index：
   - Dimensions: **512**（CLIP ViT-B/32 输出维度）
   - Distance Metric: **Cosine**
3. 复制 `UPSTASH_VECTOR_REST_URL` 和 `UPSTASH_VECTOR_REST_TOKEN`
4. 写入 `.env` 文件：

```env
UPSTASH_VECTOR_REST_URL=https://xxxx.upstash.io
UPSTASH_VECTOR_REST_TOKEN=xxxxxxxx
```

---

## 四、CLIP 向量化与入库（你自己完成这部分）

> **注意**：这部分是作业要求你自己完成的 CLIP 核心部分，下面是完整的流程说明与代码框架，你来填写并运行。

### 4.1 代码结构

```
image_search/
├── .env
├── encode_and_upload.py   ← 你负责：CLIP 向量化 + 上传
├── app.py                 ← AI 帮你：Gradio 界面（见第五章）
├── search.py              ← 你负责：向量检索封装
├── data/
│   └── flickr30k/         ← 数据集图片
└── cache/
    └── vectors.npy        ← 可选：本地向量缓存
```

---

## 五、Gradio 界面设计（详细规格）

> 这是最关键的部分，下面的设计规格对 AI 和你都可读，可直接用于自动生成 `app.py`。

### 5.1 界面整体布局

```
┌─────────────────────────────────────────────────────────┐
│  🔍 VISUAL SEARCH                          [主标题]      │
│  Semantic Image Retrieval · CLIP + Upstash              │
├────────────────────────┬────────────────────────────────┤
│   LEFT PANEL (输入区)   │   RIGHT PANEL (结果区)         │
│                        │                                │
│  [Tab: 📝 Text]        │  📊 Result Overview            │
│  [Tab: 🖼️ Image]       │  Found 12 results · 0.23s     │
│                        │                                │
│  ── Text Tab ──        │  ┌─────────────────────────┐  │
│  ┌──────────────────┐  │  │                         │  │
│  │ Enter query...   │  │  │   🖼️  Gallery Grid      │  │
│  └──────────────────┘  │  │   (3 × N images)        │  │
│                        │  │   Each card shows:      │  │
│  Query Preview:        │  │   - image thumbnail     │  │
│  ┌──────────────────┐  │  │   - similarity score    │  │
│  │ "red apple"      │  │  │   - ❤️ / ⬇️ buttons    │  │
│  └──────────────────┘  │  │                         │  │
│                        │  └─────────────────────────┘  │
│  ── Parameters ──      │                                │
│  Top-K: [====●] 12     │  ── Favorites List ──          │
│  Min Score: [=●] 0.20  │  [Saved image thumbnails]      │
│                        │  [⬇️ Export Favorites ZIP]     │
│  [🔍 Search]           │                                │
└────────────────────────┴────────────────────────────────┘
```

### 5.2 完整 app.py 代码


---


```

---

## 七、Five-Stage Framework 实现对照（报告素材）

| Stage | 界面元素 | 具体实现 |
|-------|---------|---------|
| **1. Formulation** | 输入框 + 预览区 | 文字 Textbox 实时更新 Preview；图片 Upload 后显示"Ready to search"确认 |
| **2. Initiation of Action** | Search 按钮 | `gr.Button("🔍 Search")` 触发检索，颜色醒目（紫色渐变） |
| **3. Review of Results** | Gallery + 状态栏 | 显示图片网格 + "Found X results · Xs" 时间统计 |
| **4. Refinement** | 参数滑块 | Top-K 滑块（4-48）+ Min Score 阈值，随时调整重新搜索 |
| **5. Use** | 收藏 + 下载 | 点击图片加入 Favorites；ZIP 打包下载 |

---

## 八、报告问题参考答案（英文）

### Q1: Dataset Description (1分)

> We use the **Flickr30k** dataset, which contains **31,783 images** sourced from the Flickr photo-sharing platform. Each image depicts everyday human activities and scenes, such as people playing sports, animals, urban environments, and social events. Each image is annotated with five independent natural-language captions written by human annotators, making it a rich resource for vision-language research. The dataset covers diverse subjects including people (the majority), animals, vehicles, and indoor/outdoor scenes. We use the test split (~1,000 images) for demonstration and the full dataset for indexing.

---

### Q2: How Does Your Interface Reflect the Five-Stage Search Framework? (1分)

> The interface is explicitly designed around the Five-Stage Search Framework proposed by Marchionini (1992):
>
> **Formulation**: Users can either type a natural-language description in the text input box or upload a query image. Both inputs include a real-time preview panel — text queries are echoed back immediately as the user types, and uploaded images trigger a status confirmation. This helps users verify their intent before searching.
>
> **Initiation of Action**: A prominent "🔍 Search" button with clear visual styling (gradient background) triggers the retrieval process. Users have explicit control over when the search begins.
>
> **Review of Results**: Retrieved images are displayed in a responsive Gallery grid sorted by cosine similarity score. Each thumbnail shows the similarity score as a caption. A status bar displays the total result count and query latency (e.g., "Found 12 results · 0.23s"), giving users an immediate overview.
>
> **Refinement**: Two sliders allow users to adjust search parameters without reformulating from scratch: (1) Top-K controls how many results to retrieve (4–48), and (2) Minimum Similarity Score filters out low-confidence matches. Changing these sliders and re-clicking Search refines results interactively.
>
> **Use**: Users can click any retrieved image to add it to a persistent Favorites list shown below the main gallery. The favorites can be downloaded as a ZIP archive, enabling direct use of retrieved assets.

---

### Q3: Impact of Different Input Modalities on User Workflow (1分)

> Text and image inputs create fundamentally different user workflows, each with distinct friction points.
>
> **Text search** requires users to verbalize their visual intent — a process that can be difficult when the target is abstract or when users struggle to find the right words (the "vocabulary mismatch" problem). To mitigate this, we provide a real-time query preview that echoes the user's text, helping them confirm their formulation before initiating the search. We also support descriptive, natural-language queries (not just keywords), leveraging CLIP's cross-modal understanding.
>
> **Image search** eliminates the language barrier but introduces a different friction: users must have a representative query image ready. We reduce this friction by supporting direct drag-and-drop or webcam capture through Gradio's Image component. The uploaded image is immediately previewed, so users can verify they uploaded the correct file.
>
> To make both modalities equally friendly, we apply three design principles: (1) **Symmetric layout** — both tabs share the same parameters, status bar, and result gallery, so switching modes feels seamless; (2) **Consistent feedback** — both modes provide a preview stage before searching and show identical result formats; (3) **Same refinement controls** — Top-K and score threshold sliders work identically for both modes, ensuring users don't need to relearn the interface when switching input types.

---

## 九、亮点总结（可在 Presentation 中强调）

1. **跨模态统一检索**：文字和图片共享同一 CLIP 向量空间，用户可以用文字找到视觉匹配，也可以用图片找到语义相似图片
2. **实时查询预览**：满足 Formulation 阶段的"用户确认意图"需求，减少误操作
3. **参数动态调整**：Top-K + 相似度阈值支持 Refinement，不需要重新输入查询
4. **完整的 Use 阶段**：收藏 + ZIP 下载，不只是"看图"，而是真正可以使用结果
5. **响应速度**：Upstash Vector 托管向量库，查询延迟通常 < 300ms
6. **数据集规模**：Flickr30k 31k 张图，覆盖多元日常场景，与 CLIP 训练分布高度契合

---

*文档版本：v1.0 · 2025*
