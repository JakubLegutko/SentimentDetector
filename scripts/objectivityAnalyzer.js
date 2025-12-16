
(function () {
  // Available models
  const MODELS = {
    "deberta-zero-shot": {
      id: "Xenova/nli-deberta-v3-base",
      type: "zero-shot-classification",
      labels: ["objective", "subjective"],
      name: "English (Deberta v3)"
    },
    "bert-base": {
      id: "Xenova/bert-base-uncased",
      type: "feature-extraction",
      name: "BERT Base (Uncased)"
    },
    "distilbert-subjectivity": {
      id: "distilbert-subjectivity", // Matches the folder name in models/
      type: "text-classification",   // It is now a classifier, not zero-shot
      labels: ["subjective", "objective"], // Label 0, Label 1 mapping
      name: "Fine-Tuned Subjectivity",
      quantized: false // We exported as standard ONNX (float32), not quantized
    },
    "gemini": {
      id: "gemini-2.5-flash",
      type: "llm",
      name: "LLM (Gemini)"
    }

  };

  const USE_LOCAL_SERVER = true;
  const SERVER_URL = "http://localhost:8000";

  let currentModelKey = "deberta-zero-shot";
  let pipelinePromise = null;
  let activePipeline = null;

  async function loadPipeline(modelKey) {

    if (!window.transformers && typeof window.loadTransformersRuntime === "function") {
      await window.loadTransformersRuntime();
    }
    if (!window.transformers) {
      throw new Error("Transformers runtime not found");
    }

    const { pipeline, env } = window.transformers;

    env.allowLocalModels = true;
    env.localModelPath = chrome.runtime.getURL('models/');
    env.useBrowserCache = false; // Disable cache to avoid 'chrome-extension' scheme errors

    const config = MODELS[modelKey];
    if (!config) throw new Error(`Unknown model key: ${modelKey}`);

    const pipe = await pipeline(config.type, config.id, {
      quantized: config.hasOwnProperty('quantized') ? config.quantized : true,
      local_files_only: !config.id.includes('/') // Force local if it's a simple folder name
    });

    return { pipe, config };
  }

  async function setModel(modelKey) {
    if (modelKey === currentModelKey && activePipeline) {
      return;
    }

    currentModelKey = modelKey;

    if (USE_LOCAL_SERVER) {
      console.log(`Model switched to ${MODELS[modelKey].name} (Server Mode)`);
      // We pretend the pipeline is loaded so analysis can proceed
      pipelinePromise = Promise.resolve({ config: MODELS[modelKey] });
      activePipeline = { config: MODELS[modelKey] };
      return;
    }

    pipelinePromise = loadPipeline(modelKey);

    try {
      const result = await pipelinePromise;
      activePipeline = result;
      console.log(`Model switched to ${result.config.name}`);
    } catch (e) {
      console.error("Failed to load model", e);
      pipelinePromise = null;
      activePipeline = null;
      throw e;
    }
  }



  async function analyze(text) {
    // 1. Local Server Mode
    if (USE_LOCAL_SERVER) {
      try {
        const response = await fetch(`${SERVER_URL}/analyze`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text, model: currentModelKey })
        });
        if (!response.ok) {
          let errorMsg = "Server returned " + response.status;
          try {
            const errData = await response.json();
            if (errData.detail) errorMsg += ": " + errData.detail;
          } catch (e) { /* ignore parse error */ }
          throw new Error(errorMsg);
        }
        const result = await response.json();
        console.log("SERVER RESPONSE:", result);

        // Handle Response based on Model Type
        if (currentModelKey === "deberta-zero-shot") {
          // Server returns { labels: [...], scores: [...] }
          const subjectiveIdx = result.labels.indexOf("subjective");
          const objectiveIdx = result.labels.indexOf("objective");
          const subjectiveScore = result.scores[subjectiveIdx];
          const objectiveScore = result.scores[objectiveIdx];
          const finalScore = (objectiveScore - subjectiveScore) * 100;
          return {
            score: finalScore,
            label: finalScore > 0 ? "Objective" : "Subjective",
            confidence: Math.abs(finalScore),
            model: "Local Server (DeBERTa)",
            provider: "server"
          };

        } else if (currentModelKey === "gemini") {
          // Gemini returns { score: -1 to 1, explanation: "..." }
          // UI expects score of -100 to 100
          const finalScore = result.score * 100;
          return {
            score: finalScore,
            label: result.label, // "Objective" or "Subjective"
            confidence: Math.abs(finalScore),
            model: "Gemini 2.5 Flash",
            provider: "server-llm",
            explanation: result.explanation
          }
        } else if (currentModelKey === "bert-base") {
          return {
            score: result.score,
            label: result.label,
            confidence: 0,
            model: result.model,
            provider: "server"
          };
        }

        // Default: Fine-Tuned Subjectivity (DistilBERT)
        // Map Python output to JS expected format
        // Python: { label: 'LABEL_0', score: 0.99 } or { label: 'subjective', score: ... }

        // Result is now an array of [{label, score}, {label, score}] due to top_k=None
        let subjectiveScore = 0;
        let objectiveScore = 0;

        if (Array.isArray(result)) {
          result.forEach(item => {
            if (item.label === "LABEL_0" || item.label === "subjective") {
              subjectivityLabelFound = true;
              subjectiveScore = item.score;
            } else if (item.label === "LABEL_1" || item.label === "objective") {
              objectiveScore = item.score;
            }
          });
        } else {
          // Fallback for single object (legacy)
          if (result.label === "LABEL_0" || result.label === "subjective") {
            subjectiveScore = result.score;
            objectiveScore = 1 - result.score;
          } else {
            objectiveScore = result.score;
            subjectiveScore = 1 - result.score;
          }
        }

        const finalScore = (objectiveScore - subjectiveScore) * 100;

        return {
          score: finalScore,
          label: finalScore > 0 ? "Objective" : "Subjective",
          confidence: Math.abs(finalScore),
          model: "Local Server (DistilBERT)",
          provider: "server"
        };
      } catch (e) {
        console.error("Server analysis failed, falling back to browser if possible", e);
        // Fallback or throw? Let's throw for now to see errors clearly.
        throw e;
      }
    }

    // 2. Browser WASM Mode
    // If no pipeline is loaded/loading, trigger load of current default
    if (!pipelinePromise) {
      setModel(currentModelKey);
    }

    // ... existing WASM logic ...
    try {
      const { pipe, config } = await pipelinePromise;

      const truncated = text.slice(0, 1500);

      if (config.type === "zero-shot-classification") {
        // ... existing zero shot ...
        const output = await pipe(truncated, config.labels);
        // ...
        // Copying barely needed since we are replacing the block wrapper
        // Actually, let's just delegate to a helper since we are inside 'analyze'
        return analyzeWASM(truncated, pipe, config);
      }
      else if (config.id.includes("bert-base")) {
        return {
          score: 0.5,
          label: "Needs Fine-tuning",
          confidence: 0,
          model: config.name,
          provider: "transformer-base"
        };
      } else if (config.type === "text-classification") {
        // Our fine-tuned WASM model logic
        const output = await pipe(truncated);
        // output is [{ label: 'subjective', score: 0.99 }] or LABEL_0
        const res = output[0];
        let sScore = 0;
        let oScore = 0;

        if (res.label === "subjective" || res.label === "LABEL_0") {
          sScore = res.score;
          oScore = 1 - res.score;
        } else {
          oScore = res.score;
          sScore = 1 - res.score;
        }
        const fScore = (oScore - sScore) * 100;
        return {
          score: fScore,
          label: fScore > 0 ? "Objective" : "Subjective",
          confidence: Math.abs(fScore),
          model: config.name,
          provider: "transformer-wasm"
        };
      }
    } catch (error) {
      console.warn("Transformer analysis failed", error);
      throw error;
    }
  }

  // Helper for original WASM logic to keep 'analyze' clean-ish
  async function analyzeWASM(text, pipe, config) {
    const output = await pipe(text, config.labels);
    const subjectiveIdx = output.labels.indexOf("subjective");
    const objectiveIdx = output.labels.indexOf("objective");
    const subjectiveScore = output.scores[subjectiveIdx];
    const objectiveScore = output.scores[objectiveIdx];
    const score = (objectiveScore - subjectiveScore) * 100;
    return {
      score: score,
      label: score > 0 ? "Objective" : "Subjective",
      confidence: Math.abs(score),
      model: config.name,
      provider: "transformer"
    };
  }

  let translatorPromise = null;
  let activeTranslator = null;

  async function loadTranslator() {
    if (activeTranslator) return activeTranslator;

    if (!window.transformers && typeof window.loadTransformersRuntime === "function") {
      await window.loadTransformersRuntime();
    }

    const { pipeline, env } = window.transformers;

    // Configure for local loading
    env.allowLocalModels = true;
    env.localModelPath = chrome.runtime.getURL('models/'); // Path to models directory root
    env.useBrowserCache = false; // Disable cache to avoid 'chrome-extension' scheme errors

    // Xenova/nllb-200-distilled-600M is the standard for multilingual -> English
    // It detects language automatically or accepts src_lang.
    try {
      console.log("Loading translation model locally...");
      const translator = await pipeline('translation', 'Xenova/nllb-200-distilled-600M', {
        quantized: true,
        local_files_only: true
      });
      activeTranslator = translator;
      console.log("Translation model loaded.");
      return translator;
    } catch (e) {
      console.error("Failed to load translator", e);
      throw e;
    }
  }

  async function translate(text) {
    // 1. Local Server Mode
    if (USE_LOCAL_SERVER) {
      try {
        const response = await fetch(`${SERVER_URL}/translate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text })
        });
        if (!response.ok) throw new Error("Server translation error " + response.status);
        const result = await response.json();
        return result.translated_text;
      } catch (e) {
        console.warn("Server translation failed", e);
        return text;
      }
    }

    // 2. Browser WASM Mode
    if (!translatorPromise) {
      translatorPromise = loadTranslator();
    }
    const translator = await translatorPromise;
    try {
      const output = await translator(text, {
        tgt_lang: 'eng_Latn'
      });
      return output[0].translation_text;
    } catch (e) {
      console.warn("Translation failed", e);
      return text;
    }
  }

  window.ObjectivityAnalyzer = {
    analyze,
    translate,
    setModel,
    getAvailableModels: () => MODELS,
    getCurrentModel: () => currentModelKey
  };
})();
