# Developer Guide: Recreating the Models

This guide details the pipeline used to create, fine-tune, and train the objectivity detection models used by the Objectivity Gauge extension.

## 1. Data Collection & Scraping
The project relies on real-world article data.
- Use `scripts/scraper/scraper.py` to scrape articles from various news sources.
- The scraper outputs a raw dataset (`.json`) containing the text of multiple articles.

## 2. Hosting the Local LLM proxy
We use local `.gguf` quantized models as another backend option. The same models are used in the auto_labeler and judge scripts.
- Script: `scripts/LLM_server.py`
- Setup: The server relies on `llama-cpp-python` to achieve high-performance inference on consumer hardware.
- It will automatically download specified `gguf` files (like Bielik or Mistral) on first run and host an OpenAI-compatible `/v1/chat/completions` endpoint for the extension to stream predictions from.

## 3. Dataset Management & Labeling
- **Cleaning & Trimming**: Use `clean_dataset.py` and `trim_dataset.py` to normalize the formatting and remove outliers or improperly formatted text.
Once you have the cleaned scraped text, you need to prepare it for training by judging its objectivity.
- **LLM based decision**: Run scripts/dataset_management/auto_labeler.py to evaluate the scraped text using an LLM (like `bielik` or `llama3`). The script uses the LLM as a judge to assign a continuous score between `-1.0` (Subjective) and `1.0` (Objective).
- **Consolidation**: Use `scripts/dataset_management/consolidate_datasets.py` to merge multiple runs or sources into a single, cohesive training dataset (e.g., `average_review_no_score_judged_bielik.json`). You can merge individual runs from the auto_labeler before feeding them into the LLM-as-a-judge script.
- **Judge Automation**: Run `scripts/dataset_management/judge.py` to evaluate the scraped text using an LLM-as-a-judge approach (like `bielik` or `llama3` evaluating other models). The script uses the LLM as a judge to choose which model gave the best answer.


## 4. Fine-Tuning the DeBERTa Model
DeBERTa is used as our heavy-duty analytical model.
- Script: `scripts/model_tuner/finetune.py`
- Setup:
  Ensure `average_review_no_score_judged_bielik.json` (or your chosen dataset) is in the root or `datasets/` directory.
- Run:
  Execute the tuning script. It will load the pre-trained DeBERTa weights, map the continuous `[-1, 1]` scores into a regression loss function (MSE), and train the model using Hugging Face's `Trainer`. 
- Output: The fine-tuned weights are saved to `models/deberta_objectivity/`.

## 5. Training the 1DCNN Model
The 1D Convolutional Neural Network (1DCNN) acts as our extremely fast, CPU-friendly alternative for real-time scrolling analysis.
- Script: `scripts/train_1dcnn.py`
- Concept: It builds a custom vocabulary from the training dataset, applies embeddings, and runs 1D convolutions over the text tokens to predict the regression score.
- Run:
  Execute `python scripts/train_1dcnn.py`.
- Output: The script saves the `1dcnn_objectivity_model.pt` PyTorch weights directly to the `models/` directory, alongside its vocabulary mapping.


