
(function () {
  // Available models
  const MODELS = {
    "deberta_objectivity": {
      id: "deberta_objectivity",
      type: "text-classification",
      name: "Fine-Tuned DeBERTa",
      quantized: false
    },
    "1dcnn_objectivity_model": {
      id: "1dcnn_objectivity_model",
      type: "custom",
      name: "1DCNN Objectivity"
    },
    "gemini": {
      id: "gemini-2.5-flash",
      type: "llm",
      name: "LLM (Gemini)"
    },
    "local_llm": {
      id: "local_llm",
      type: "llm",
      name: "Local LLM"
    }

  };

  const USE_LOCAL_SERVER = true;
  const SERVER_URL = "http://localhost:8000";

  let currentModelKey = "deberta_objectivity";
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



  let sessionApiKey = null;

  async function analyze(text) {
    // 1. Local Server Mode
    if (USE_LOCAL_SERVER) {
      try {
        const payload = { text: text, model: currentModelKey };
        if (currentModelKey === "gemini" && sessionApiKey) {
          payload.api_key = sessionApiKey;
        }

        let responseWrapper;
        try {
          responseWrapper = await chrome.runtime.sendMessage({
            type: "FETCH_API",
            endpoint: "/analyze",
            payload: payload
          });
          if (responseWrapper.error) {
            throw new Error(responseWrapper.error);
          }
        } catch (fetchErr) {
          console.error("Network or Extension error during background fetch:", fetchErr);
          throw new Error("Failed to connect to local server at 127.0.0.1:8000 via background.");
        }

        if (!responseWrapper.ok) {
          let errorMsg = "Server returned " + responseWrapper.status;
          let detail = "";

          if (responseWrapper.data && responseWrapper.data.detail) {
            errorMsg += ": " + responseWrapper.data.detail;
            detail = responseWrapper.data.detail;
          }

          // Check for missing API key error
          if (currentModelKey === "gemini" && (detail.includes("GEMINI_API_KEY") || response.status === 500)) {
            const userKey = prompt("Gemini API Key missing on server.\nPlease enter your Gemini API Key to proceed:");
            if (userKey) {
              sessionApiKey = userKey;
              return analyze(text); // Retry with new key
            }
          }

          throw new Error(errorMsg);
        }

        const result = responseWrapper.data;
        console.log("SERVER RESPONSE:", result);

        // Handle Response based on Model Type
        if (currentModelKey === "deberta_objectivity") {
          const finalScore = result.score * 100; // -1 to 1 mapped to -100 to 100
          return {
            score: finalScore,
            label: finalScore > 0 ? "Objective" : "Subjective",
            confidence: Math.abs(finalScore),
            model: "Local Server (DeBERTa)",
            provider: "server"
          };
        } else if (currentModelKey === "1dcnn_objectivity_model") {
          const finalScore = result.score * 100;
          return {
            score: finalScore,
            label: finalScore > 0 ? "Objective" : "Subjective",
            confidence: Math.abs(finalScore),
            model: "Local Server (1DCNN)",
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
        } else if (currentModelKey === "local_llm") {
          // Same format as Gemini
          const finalScore = result.score * 100;
          return {
            score: finalScore,
            label: result.label,
            confidence: Math.abs(finalScore),
            model: "Local LLM",
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

        // No fallback models here
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
        const responseWrapper = await chrome.runtime.sendMessage({
          type: "FETCH_API",
          endpoint: "/translate",
          payload: { text: text }
        });

        if (responseWrapper.error) throw new Error(responseWrapper.error);
        if (!responseWrapper.ok) throw new Error("Server translation error " + responseWrapper.status);

        return responseWrapper.data.translated_text;
      } catch (e) {
        console.warn("Server translation failed via background", e);
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
