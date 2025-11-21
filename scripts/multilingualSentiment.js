(function () {
  const MODEL_ID = "Xenova/bert-base-multilingual-uncased-sentiment";
  const LABEL_TO_SCORE = {
    "1 star": 0.05,
    "2 stars": 0.25,
    "3 stars": 0.5,
    "4 stars": 0.75,
    "5 stars": 0.95
  };
  let pipelinePromise = null;

  function descriptorFromScore(score) {
    if (score > 0.66) return "Positive";
    if (score < 0.34) return "Negative";
    return "Neutral";
  }

  async function loadPipeline() {
    if (!window.transformers && typeof window.loadTransformersRuntime === "function") {
      await window.loadTransformersRuntime();
    }
    if (!window.transformers) {
      throw new Error("Transformers runtime (window.transformers) not found");
    }
    if (!pipelinePromise) {
      const { pipeline } = window.transformers;
      pipelinePromise = pipeline("sentiment-analysis", MODEL_ID, {
        quantized: true
      });
    }
    return pipelinePromise;
  }

  async function analyze(text, language = "auto") {
    const classifier = await loadPipeline();
    const truncated = text.slice(0, 4000);
    const output = await classifier(truncated, { topk: 1 });
    const first = Array.isArray(output) ? output[0] : output;
    const normalized =
      LABEL_TO_SCORE[first?.label] !== undefined ? LABEL_TO_SCORE[first.label] : first?.score ?? 0.5;
    return {
      normalized,
      aggregate: first?.score ?? normalized,
      matches: 0,
      descriptor: descriptorFromScore(normalized),
      sampleSentences: [],
      provider: "transformers",
      language
    };
  }

  window.MultilingualSentiment = {
    analyze,
    isLoaded: () => Boolean(pipelinePromise),
    MODEL_ID
  };
})();

