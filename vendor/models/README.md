Local model bundles (for offline / authorized loading)
======================================================

Place the model artifacts here so the extension can load them without hitting Hugging Face.

Directory layout used by the content scripts:
- `vendor/models/sentiment/` → Sentiment model (config.json entry point)
- `vendor/models/political/` → Political bias model (config.json entry point)

Expected files per model (all alongside config.json):
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `vocab.txt` (or merges.txt/vocab.json for BPE-based models)
- `model.onnx` (quantized is fine/preferred)

Suggested checkpoints to download (public):
- Sentiment: `Xenova/distilbert-base-uncased-finetuned-sst-2-english`
- Political bias (zero-shot labels left/center/right): `Xenova/bart-large-mnli` (smaller option) or `Xenova/deberta-v3-base-mnli`

How to populate:
1) Download the above repos from Hugging Face (or export to ONNX with `@xenova/transformers`/`optimum`) and copy the files into the corresponding folder.
2) Keep the filenames exactly as listed so the runtime can resolve relative paths.
3) After copying, reload the extension so the new assets are available.
