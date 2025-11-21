(function () {
  const MODEL_ID = "Xenova/nli-deberta-v3-base";
  const LABELS = [
    { label: "left-wing", descriptor: "Left leaning", value: 0 },
    { label: "centrist", descriptor: "Center", value: 0.5 },
    { label: "right-wing", descriptor: "Right leaning", value: 1 }
  ];
  const HYPOTHESIS_TEMPLATE = "The political perspective of this text is {}.";
  let classifierPromise = null;

  async function loadClassifier() {
    if (!window.transformers && typeof window.loadTransformersRuntime === "function") {
      await window.loadTransformersRuntime();
    }
    if (!window.transformers) {
      throw new Error("Transformers runtime unavailable");
    }
    if (!classifierPromise) {
      const { pipeline } = window.transformers;
      classifierPromise = pipeline("zero-shot-classification", MODEL_ID, {
        quantized: true
      });
    }
    return classifierPromise;
  }

  async function classify(text) {
    if (!text || !text.trim()) {
      throw new Error("Empty text");
    }
    const classifier = await loadClassifier();
    const truncated = text.slice(0, 2000);
    const output = await classifier(
      truncated,
      LABELS.map((label) => label.label),
      {
        hypothesis_template: HYPOTHESIS_TEMPLATE,
        multi_label: false
      }
    );

    const distribution = {};
    let expectation = 0;

    output.labels.forEach((labelText, index) => {
      const score = output.scores[index];
      const normalizedLabel = labelText.toLowerCase();
      const config = LABELS.find((entry) => entry.label === normalizedLabel);
      if (config) {
        distribution[config.descriptor] = score;
        expectation += config.value * score;
      }
    });

    const bestLabel = output.labels[0]?.toLowerCase();
    const bestConfig = LABELS.find((entry) => entry.label === bestLabel);

    return {
      descriptor: bestConfig?.descriptor ?? "Center",
      normalized: Math.min(1, Math.max(0, expectation)),
      confidence: output.scores[0] ?? 0,
      distribution,
      provider: "transformer-zero-shot"
    };
  }

  window.PoliticalBiasAnalyzer = {
    classify
  };
})();

