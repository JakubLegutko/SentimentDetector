
# Objectivity Gauge

A Chromium extension that analyzes the objectivity of the text currently visible on the active tab or selection. It determines if the content is **Objective** (fact-based) or **Subjective** (opinion-based).

## Features
- **Real-time Analysis**: Scrapes visible text and updates the objectivity gauge on demand.
- **Hybrid Architecture**: Uses a local Python server for heavy lifting (Transformers) and browser-side JS for lightweight fallback (Vader/Lexicon).
- **Auto-Translation**: Automatically detects non-English text and translates it to English using the NLLB model before analysis.
- **Models**:
  - **Fine-Tuned Subjectivity**: A DistilBERT model trained specifically for objectivity detection.
  - **DeBERTa (Zero-Shot)**: A heavy-duty NLI model for zero-shot classification.
  - **Vader**: A lightweight, lexicon-based sentiment analyzer (runs in-browser).
- **Selection Mode**: Highlight any text on the page and click "Sel" to analyze just that snippet.
- **"Why?" Panel**: Explains the analysis by showing subjective keywords found in the text. Only works for Vader right now.

## Server Setup (Required)
This extension relies on a local Python server to run the Transformer models.

### Option 1: Run Locally
1. Install dependencies:
   ```bash
   pip install fastapi uvicorn torch transformers datasets sentencepiece langdetect
   ```
2. Start the server:
   ```bash
   python scripts/server.py
   ```
   The server runs on `http://localhost:8000`.

### Restoring Models (First Run)
Since large model files are not stored in Git, you need to download them:

1. **Download Base Models** (DeBERTa, NLLB, BERT):
   ```bash
   python scripts/download_models.py
   ```
2. **Restore Fine-Tuned Model**:
   - *Option A (Train it)*: Run `python model_tuner/finetune.py --dataset_name <your-dataset>`
   - *Option B (Download it)*: If you have the `distilbert_subjectivity_v1` folder saved elsewhere, place it in the root directory.
   - *Fallback*: If missing, the server will default to `distilbert-base-uncased`.

### Option 2: Run with Docker
1. Build the image:
   ```bash
   docker build -t objectivity-server .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 objectivity-server
   ```

## Install Extension
1. Open Chrome/Edge and visit `chrome://extensions`.
2. Enable **Developer mode** (top-right).
3. Choose **Load unpacked**, then select the `ObjectivityGauge` folder.
4. Navigate to any page; the panel appears in the bottom-right corner.

## Project Structure
- `manifest.json`: Configuration for the Chrome Extension.
- `scripts/server.py`: FastAPI server handling Transformer models and translation.
- `scripts/contentScript.js`: Main UI and logic for the browser extension.
- `scripts/objectivityAnalyzer.js`: Client-side logic to communicate with the server.
- `vendor/vader.js`: Corrected VADER sentiment library (with custom normalization).
- `Dockerfile`: Configuration for containerizing the server.
