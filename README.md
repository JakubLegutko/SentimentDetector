# Sentiment & Bias Gauges

Chromium extension that scrapes the text currently visible on the active tab, performs dual analysis (positivity and political left/right leaning) and overlays two gauges on the page. Sentiment scoring leverages the open-source [AFINN-en-165](https://github.com/fnielsen/afinn) lexicon (MIT License).

## Features
- Lightweight content script that watches DOM mutations and keeps the overlay updated in real time.
- Positivity gauge (negative → positive) powered by the AFINN sentiment model with multi-word phrase support.
- Auto language detection: non-English pages fall back to a multilingual transformer (`Xenova/bert-base-multilingual-uncased-sentiment`) loaded entirely in-browser.
- Political spectrum gauge (left ↔ right) driven by a zero-shot transformer (`Xenova/nli-deberta-v3-base`) with a lexicon fallback when the model is unavailable.
- Context surface (match counts, word counts, top keywords) to understand why the gauges moved.
- “Sel” button lets you highlight any snippet on the page and run an on-demand report for just that text.
- “Why?” button expands a reasoning panel showing contributing sentences, keywords, and political references for the current page.
- Floating panel can be minimized/expanded so it stays out of the way on dense layouts.
- Fully client-side – no data leaves the page.

## Install & run
1. Open Chrome/Edge and visit `chrome://extensions`.
2. Enable **Developer mode** (top-right).
3. Choose **Load unpacked**, then select the `SentimentDetector` folder.
4. Navigate to any page with text; the floating panel appears in the bottom-right corner.

## Project structure
- `manifest.json` – MV3 definition, points to the content script/CSS bundle.
- `scripts/afinnLexicon.js` – generated JS map of the AFINN-en-165 lexicon (MIT).
- `scripts/contentScript.js` – DOM scraper, sentiment & bias analyzers, UI overlay.
- `scripts/multilingualSentiment.js` – async wrapper around `@xenova/transformers` for multilingual scoring.
- `styles/panel.css` – styling for the floating gauges.
- `icons/` – extension icons (PNG, generated locally).
- `vendor/transformers.min.js` – prebuilt runtime for the Xenova transformer pipelines.

## Notes & limitations
- AFINN is tuned for English informal text; accuracy declines on other languages or domain-specific jargon, but the transformer path improves coverage for many Romance/Germanic languages.
- Political leaning defaults to the zero-shot transformer and falls back to the lexicon only if the model fails or is still loading; satire, quotes, or mixed viewpoints may still yield middling scores.
- For very long pages the analyzer trims text to ~20k characters for performance.
- First multilingual inference loads the model (≈20 MB); subsequent pages reuse the in-memory pipeline.
- All computation happens in the isolated content script world; no access to page JS objects is required.

## Future ideas
- Ship a popup UI to show the top contributing sentences.
- Add multilingual sentiment models via `@xenova/transformers` (on-demand).
- Persist per-domain baselines to illustrate relative shifts over time.

