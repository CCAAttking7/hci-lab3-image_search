# 🔍 Five-Stage Visual Search System

An intelligent semantic image retrieval interface rigorously built on the **Five-Stage Search Framework**, powered by CLIP and Upstash Vector DB.

## 🚀 Execution Workflow & Architecture

### 1. Data Processing Pipeline (From CPU to GPU)
Our initial plan was to encode dataset images locally. However, running the **CLIP ViT-B/32** model across thousands of images on a local CPU was computationally overwhelming (CPU almost "smoked" 🔥). 

To ensure optimal performance and build a comprehensive database, **the vectorization pipeline was migrated to a GPU environment on Kaggle**.
- The script `src/full_on_GPU.py` was executed directly on a Kaggle GPU (T4 x2).
- We successfully downloaded, encoded, and upserted **over 5,000 images** from the `MS-COCO 2017 Validation` dataset into our cloud vector database (Upstash) in a fraction of the time.

### 2. Retrieval Architecture
- **Embedding:** Users input text or upload an image. The local app uses `OpenAI CLIP` to map the query to a 512-dimensional vector.
- **Search Engine:** We query the Remote `Upstash Vector DB` using *Cosine Similarity*.
- **User Interface:** A Gradio-based frontend customized for flawless human-computer interaction.

---

## 🎯 Satisfaction of HCI Rubrics

This project strictly adheres to the scoring rubric to satisfy the Five-Stage design practices.

### I. Formulation (2 Points)
* **Text & Image Formulation (1 pt):** Dedicated tabs allowing both Image uploads (`gr.Image()`) and Text descriptions (`gr.Textbox()`).
* **Query Preview (1 pt):** A real-time preview panel dynamically generates previews (translated text indicators for Chinese inputs, or thumbnails for image inputs) so the query is always visible.
* *Bonus (HCI Enhancement):* Integrated **Bing Autocomplete API** provides live, contextual search suggestions while typing. A fallback corpus is provided if offline. Intelligent Chinese-to-English translation enables cross-lingual semantics.

### II. Initiation & Review (2 Points)
* **Initiation (1 pt):** Clear **Search** buttons for both text and image modes. Implicit actions are also heavily utilized (Pressing Enter).
* **Review Overview (1 pt):** Accurate status bars summarize the retrieved volume and backend process time (e.g., `Found 12 results · Query time: 0.45s`).

### III. Refinement (1 Point)
* **Parameter Adjustments (1 pt):** We implemented non-blocking Refinement mechanisms. Changing the returned subset size (`Top-K`) or adjusting the Minimum Similarity Threshold dynamically **updates the results on-the-fly** without requiring the user to hit click again.

### IV. Use & Actions (1 Point)
* **Favorite & Download Actions (1 pt):** Selecting an image seamlessly toggles its "Saved" state (`❤️ Saved` / `♡ Save to favorites`). Saved items populate a localized gallery where they can be entirely cleared or packed into a `.ZIP` file for immediate local download.

### V. Core Retrival Capabilities (5 Points)
* **Text-to-Image (2 pts):** Retrieve images sorted by CLIP cosine similarity via text.
* **Image-to-Image (2 pts):** Retrieve visually/semantically similar images by uploading a query image.
* **Result Usability (1 pt):** Core architecture correctly bridges retrieval with the aforementioned HCI Download/Save mechanisms.

---

## 💻 How to Run Locally

1. **Clone the environment:**
   We use `uv` for lightning-fast package management. Ensure Python 3.12+ is installed.
   ```bash
   uv sync
   # or manually use the current virtual environment (.venv)
   ```

2. **Supply Environments:**
   You must have your Upstash credentials stored or exported:
   ```env
   UPSTASH_VECTOR_REST_URL=your_url
   UPSTASH_VECTOR_REST_TOKEN=your_token
   ```

3. **Launch the application:**
   ```bash
   uv run python app.py
   ```
   Open `http://localhost:7860` in any web browser to see the interface.