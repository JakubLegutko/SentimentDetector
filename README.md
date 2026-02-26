
# Objectivity Gauge

A Chromium extension that analyzes the objectivity of the text currently visible on the active tab or selection. It determines if the content is **Objective** (fact-based) or **Subjective** (opinion-based).

## Features
- **Real-time Analysis**: Scrapes visible text and updates the objectivity gauge on demand.
- **Minimize to Tray (Auto-Analysis)**: Minimize the extension panel to analyze text natively as you scroll, updating the toolbar icon with color-coded badges (Green=Objective, Yellow=Mixed, Red=Subjective).
- **Hybrid Architecture**: Uses a local Python server for heavy lifting (Transformers, CNNs, LLMs) and browser-side JS for lightweight fallback (Vader/Lexicon).
- **Auto-Translation**: Automatically detects non-English text and translates it to English using the NLLB model prior to analysis (specific to Vader).
- **Available Models**:
  - **1DCNN Objectivity**: A custom, lightweight Convolutional Neural Network trained specifically for fast objectivity detection.
  - **Fine-Tuned DeBERTa**: A powerful DeBERTa architecture fine-tuned for high-accuracy objectivity scoring.
  - **Local LLM Server**: Support for proxying requests to local Ollama/GGUF models (like Bielik, Llama3) via an integrated chat-completion endpoint.
  - **Gemini**: Proxy integration for Google's Gemini-flash for LLM reasoning.
  - **Vader**: A lightweight, lexicon-based sentiment analyzer (runs entirely in-browser).
- **Selection Mode**: Highlight any text on the page and click "Sel" to analyze just that snippet.
- **"Why?" Panel**: Explains the analysis by showing subjective keywords found in the text.

## Server Setup (Required)
This extension relies on a local Python server to run the Transformer models.

### Option 1: Run Locally
1. Install dependencies from the requirements file:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the combined API/LLM server:
   ```bash
   ./scripts/start.sh
   # Or run `python scripts/server.py` and `python scripts/LLM_server.py` individually.
   ```
   The main API runs on `http://localhost:8000`, and the Local LLM proxy runs on port `11434`.
   ```
   The server runs on `http://localhost:8000`.

### Restoring Models (First Run)
Since large model files are not stored in Git, you need to download and rebuild them. 
For detailed information on how to scrape datasets, fine-tune DeBERTa, and train the 1DCNN from scratch, please see the [**Developer Guide (DEV_README.md)**](./DEV_README.md).

For a quick restore:
1. Ensure your fine-tuned models (`deberta_objectivity` directory and `1dcnn_objectivity_model.pt`) exist inside the `models/` directory. 
2. Ensure you have the NLLB translation model downloaded.

### Option 2: Run with Docker
1. Build the image:
   ```bash
   docker build -t objectivity-server .
   ```
2. Run the container (exposing both API and LLM ports):
   ```bash
   docker run -p 8000:8000 -p 11434:11434 objectivity-server
   ```

## Install Extension
1. Open Chrome/Edge and visit `chrome://extensions`.
2. Enable **Developer mode** (top-right).
3. Choose **Load unpacked**, then select the `ObjectivityGauge` folder.
4. Navigate to any page; the panel appears in the bottom-right corner.

## Project Structure
- `manifest.json`: Configuration for the Chrome Extension.
- `scripts/server.py`: FastAPI server handling Transformer models, 1DCNN inference, and translation.
- `scripts/LLM_server.py`: FastAPI application proxying local `.gguf` inference by wrapping `llama-cpp-python`.
- `scripts/contentScript.js`: Main UI and logic for the browser extension, including scroll-tracking for tray mode.
- `scripts/background.js`: Chrome Service Worker managing `fetch` proxies and icon badge updates.
- `scripts/objectivityAnalyzer.js`: Client-side logic formatting API payloads.
- `scripts/start.sh`: Bash startup script to run dual servers in Docker.
- `vendor/vader.js`: Corrected VADER sentiment library (with custom normalization).
- `Dockerfile`: Configuration for containerizing the servers.
- `Dockerfile`: Configuration for containerizing the server.
