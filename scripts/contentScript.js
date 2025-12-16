
(function () {
  const newLexiconScript = document.createElement('script');
  newLexiconScript.src = chrome.runtime.getURL('scripts/newLexicon.js');
  document.head.appendChild(newLexiconScript);

  if (window.__objectivityGaugeInjected || !document.body) {
    return;
  }
  window.__objectivityGaugeInjected = true;

  const MAX_TEXT_LENGTH = 20000;
  const TOKEN_REGEX = /\p{L}[\p{L}'-]+/gu;

  let lastAnalysis = null;

  const ui = createPanel();

  // --- Event Listeners ---

  if (ui.selectionButton) {
    ui.selectionButton.addEventListener("click", () => {
      handleSelectionAnalysis().catch((error) => {
        console.error("Selection analysis failed", error);
      });
    });
  }

  if (ui.pageButton) {
    ui.pageButton.addEventListener("click", () => {
      handlePageAnalysis().catch((error) => {
        console.error("Page analysis failed", error);
      });
    });
  }

  if (ui.detailsButton) {
    ui.detailsButton.addEventListener("click", () => {
      const expanded = ui.detailsButton.dataset.expanded === "true";
      const nextState = !expanded;
      ui.detailsButton.dataset.expanded = String(nextState);
      ui.detailsButton.textContent = nextState ? "Hide" : "Why?";
      ui.panel.classList.toggle("details-open", nextState);
      if (nextState) {
        updateAnalysisDetails(lastAnalysis);
      }
    });
  }

  if (ui.modelSelector) {
    ui.modelSelector.addEventListener("change", (e) => {
      const newModel = e.target.value;
      setPanelStatus("Switching model...");

      if (newModel === 'vader') {
        if (!window.vader) {
          setPanelStatus("Vader not found");
          console.error("Vader global missing despite static load.");
        } else {
          setPanelStatus("Model loaded");
        }
        return;
      }

      window.ObjectivityAnalyzer.setModel(newModel)
        .then(() => {
          setPanelStatus("Model loaded");
        })
        .catch((err) => {
          setPanelStatus("Model load failed");
          console.error(err);
        });
    });
  }

  // --- Analysis Logic ---

  async function handlePageAnalysis() {
    const rawText = extractVisibleText();

    if (!rawText.trim()) {
      setPanelStatus("No readable content found.");
      return;
    }

    setPanelStatus("Analyzing page...");

    const result = await analyzeObjectivity(rawText);

    updateGauge(ui.objectivityGauge, result);
    lastAnalysis = { result, text: rawText };

    if (ui.panel.classList.contains("details-open")) {
      updateAnalysisDetails(lastAnalysis);
    }
  }

  async function analyzeObjectivity(text) {
    // Auto-translation step
    // Auto-translation step (Skip for Gemini)
    if (ui.modelSelector.value !== 'gemini' && window.ObjectivityAnalyzer && window.ObjectivityAnalyzer.translate) {
      try {
        const translated = await window.ObjectivityAnalyzer.translate(text);
        if (translated !== text) {
          console.log("Text translated for analysis:", translated);
          text = translated; // Swap text for the English version
        }
      } catch (e) {
        console.warn("Auto-translation failed, proceeding with original text", e);
      }
    }

    const modelKey = ui.modelSelector.value;
    if (modelKey === 'vader') {
      // window.vader should be available.
      if (!window.vader) {
        throw new Error("Vader library not loaded");
      }
      const result = window.vader.SentimentIntensityAnalyzer.polarity_scores(text);

      // Objectivity points.
      // Compound is -1 to 1.
      // Emotional (Subjective) = High Abs(Compound).
      // Neutral (Objective) = 0.
      // Score: (0.5 - abs(compound)) * 200.
      // if 0 -> 100 pts (Objective).
      // if 1 -> -100 pts (Subjective).
      const score = (0.5 - Math.abs(result.compound)) * 200;

      // Identify contributing words
      const lexicon = window.vader.SentimentIntensityAnalyzer.LEXICON;
      const tokens = (text.match(TOKEN_REGEX) || []).map(t => t.toLowerCase());
      const subjectiveWords = [];
      let subjectiveCount = 0;

      tokens.forEach(token => {
        if (lexicon && lexicon[token] !== undefined) {
          subjectiveCount++;
          if (subjectiveWords.length < 15 && !subjectiveWords.includes(token)) {
            subjectiveWords.push(token);
          }
        }
      });

      return {
        score: score,
        label: score > 0 ? "Objective" : "Subjective",
        confidence: Math.abs(score),
        model: "Vader",
        provider: "lexicon",
        lexiconStats: {
          subjectiveWords: subjectiveWords,
          subjectiveCount: subjectiveCount,
          totalCount: tokens.length
        }
      }
    }
    // 1. Try Transformer Model
    if (window.ObjectivityAnalyzer) {
      try {
        const result = await window.ObjectivityAnalyzer.analyze(text);
        return {
          ...result,
          lexiconStats: scoreObjectivityLexicon(text)
        };
      } catch (e) {
        console.warn("Model unavailable, falling back to lexicon", e);
      }
    }

    // 2. Fallback to Lexicon
    const lexResult = scoreObjectivityLexicon(text);
    // lexResult.objectivityScore is 0..1+ (0.5 neutral)
    const fallBackScore = (lexResult.objectivityScore - 0.5) * 200;

    return {
      score: fallBackScore,
      label: fallBackScore > 0 ? "Objective" : "Subjective",
      confidence: Math.abs(fallBackScore),
      model: "Lexicon (Fallback)",
      provider: "lexicon",
      lexiconStats: lexResult
    };
  }



  function scoreObjectivityLexicon(text) {
    const tokens = (text.match(TOKEN_REGEX) || []).map(t => t.toLowerCase());
    if (tokens.length === 0) return { objectivityScore: 0.5, subjectiveCount: 0, totalCount: 0, subjectiveWords: [] };

    let score = 0;
    const subjectiveWords = [];
    const lexicon = window.subjectivityLexicon || new Map();

    tokens.forEach(token => {
      if (lexicon.has(token)) {
        score += lexicon.get(token);
        if (subjectiveWords.length < 5) subjectiveWords.push(token);
      }
    });

    const objectivityScore = Math.max(0, 0.5 + (score / tokens.length));

    return {
      objectivityScore,
      subjectiveCount: subjectiveWords.length,
      totalCount: tokens.length,
      subjectiveWords
    };
  }

  async function handleSelectionAnalysis() {
    const selectedText = getSelectionText();
    if (!selectedText) {
      renderSelectionReport({ error: "Select text first." });
      return;
    }

    renderSelectionReport({ loading: true });

    try {
      const result = await analyzeObjectivity(selectedText);
      renderSelectionReport({ result, text: selectedText });
      updateGauge(ui.objectivityGauge, result);

      lastAnalysis = { result, text: selectedText };
      if (ui.panel.classList.contains("details-open")) {
        updateAnalysisDetails(lastAnalysis);
      }

    } catch (error) {
      renderSelectionReport({ error: "Analysis failed." });
    }
  }

  // --- HTML Extraction & Filtering ---

  function extractVisibleText() {
    const primaryNode = findPrimaryContentNode();
    const cleaned = cleanNodeText(primaryNode);
    return cleaned.slice(0, MAX_TEXT_LENGTH);
  }

  function findPrimaryContentNode() {
    if (!document.body) return null;
    const selectors = [
      "main",
      "[role='main']",
      "article",
      ".article",
      ".post-content",
      ".story-body",
      ".content",
      "#content",
      "#main"
    ];

    for (const selector of selectors) {
      const node = document.querySelector(selector);
      if (node && node.innerText && node.innerText.trim().length > 400) {
        return node;
      }
    }

    const articles = Array.from(document.querySelectorAll("article"));
    if (articles.length) {
      return articles.sort((a, b) => b.innerText.length - a.innerText.length)[0];
    }

    return document.body;
  }

  function cleanNodeText(node) {
    if (!node) return "";
    const clone = node.cloneNode(true);
    const NOISE_SELECTORS = [
      "nav",
      "header",
      "footer",
      "aside",
      "[role='navigation']",
      "[role='banner']",
      "[role='contentinfo']",
      "[role='complementary']",
      ".menu",
      ".sidebar",
      ".breadcrumbs",
      ".advertisement",
      ".ad",
      ".ads",
      ".promo",
      ".modal",
      ".popup",
      "#comments",
      ".comments"
    ];

    ["script", "style", "noscript", "template", "iframe", "svg"].forEach((tag) => {
      clone.querySelectorAll(tag).forEach((el) => el.remove());
    });
    NOISE_SELECTORS.forEach((selector) => {
      clone.querySelectorAll(selector).forEach((el) => el.remove());
    });

    const text = clone.innerText || "";
    return text.replace(/\s+/g, " ").trim();
  }

  function getSelectionText() {
    return window.getSelection().toString().trim();
  }

  // --- UI Helpers ---

  function createPanel() {
    const panel = document.createElement("section");
    panel.id = "sentiment-detector-panel";

    const header = document.createElement("div");
    header.className = "panel-header";

    const titleBlock = document.createElement("div");
    titleBlock.className = "panel-title-block";
    titleBlock.innerHTML = "<h1>Objectivity</h1>";

    const actions = document.createElement("div");
    actions.className = "panel-actions";

    const modelSelector = document.createElement("select");
    modelSelector.className = "model-selector";
    if (window.ObjectivityAnalyzer) {
      const models = window.ObjectivityAnalyzer.getAvailableModels();
      Object.keys(models).forEach(key => {
        const opt = document.createElement("option");
        opt.value = key;
        opt.textContent = models[key].name;
        modelSelector.appendChild(opt);
      });
    }
    const vaderOption = document.createElement("option");
    vaderOption.value = "vader";
    vaderOption.textContent = "Vader";
    modelSelector.appendChild(vaderOption);

    const detailsButton = document.createElement("button");
    detailsButton.className = "panel-action";
    detailsButton.textContent = "Why?";

    const pageButton = document.createElement("button");
    pageButton.className = "panel-action";
    pageButton.textContent = "Page";
    pageButton.title = "Analyze main page content";

    const selectionButton = document.createElement("button");
    selectionButton.className = "panel-action";
    selectionButton.textContent = "Sel";
    selectionButton.title = "Analyze selected text";

    const toggleButton = document.createElement("button");
    toggleButton.className = "panel-toggle";
    toggleButton.textContent = "–";
    toggleButton.onclick = () => {
      panel.classList.toggle("collapsed");
      toggleButton.textContent = panel.classList.contains("collapsed") ? "+" : "–";
    };

    actions.append(modelSelector, detailsButton, pageButton, selectionButton, toggleButton);
    header.append(titleBlock, actions);

    const body = document.createElement("div");
    body.className = "panel-body";

    const gaugeRow = createGaugeRow({ title: "", legend: "Subj ↔ Obj" });
    body.appendChild(gaugeRow.row);

    const selectionReport = document.createElement("div");
    selectionReport.className = "selection-report";

    const analysisDetails = document.createElement("div");
    analysisDetails.className = "analysis-details";

    panel.append(header, body, selectionReport, analysisDetails);
    document.body.appendChild(panel);

    return {
      panel, modelSelector, detailsButton, pageButton, selectionButton,
      objectivityGauge: gaugeRow, selectionReport, analysisDetails
    };
  }

  function createGaugeRow({ title, legend }) {
    const row = document.createElement("div");
    row.className = "sentiment-row";
    row.innerHTML = `
      <label><span>${title}</span><span>${legend}</span></label>
      <div class="gauge"><div class="gauge-fill"></div><div class="gauge-indicator">•</div></div>
      <small class="context">Ready</small>
    `;
    return {
      row,
      fill: row.querySelector(".gauge-fill"),
      indicator: row.querySelector(".gauge-indicator"),
      status: row.querySelector(".context")
    };
  }

  function updateGauge(elements, result) {
    const score = result.score;
    // Map -100 (Subj) to 100 (Obj) points to 0% - 100% position
    let pct = (score + 100) / 2;
    // Clamp
    pct = Math.max(0, Math.min(100, pct));

    elements.fill.style.width = `100%`;
    elements.fill.style.background = `linear-gradient(90deg, #f87171, #34d399)`;
    elements.indicator.style.left = `${pct}%`;
    elements.indicator.textContent = `${Math.round(score)}`;
    elements.status.textContent = `${result.label} (${result.model})`;
  }

  function setPanelStatus(msg) {
    ui.objectivityGauge.status.textContent = msg;
  }

  function updateAnalysisDetails(analysis) {
    if (!analysis) return;
    const { result } = analysis;
    const stats = result.lexiconStats;

    let content = `<div class="details-section"><h3>Model Info</h3><p>${result.model}</p></div>`;

    if (result.explanation) {
      content += `
        <div class="details-section">
          <h3>Analysis Explanation</h3>
          <p class="explanation-text">${result.explanation}</p>
        </div>`;
    }

    if (stats && stats.subjectiveWords.length > 0) {
      content += `
        <div class="details-section">
          <h3>Subjective Triggers (Lexicon)</h3>
          <ul>${stats.subjectiveWords.map(w => `<li>${w}</li>`).join('')}</ul>
          <p class="subtitle">...and ${stats.subjectiveCount - stats.subjectiveWords.length} more.</p>
        </div>`;
    } else {
      // Fallback explanation when model sees subjectivity but lexicon does not
      if (result.label === "Subjective") {
        content += `
        <div class="details-section">
          <h3>Subjective Triggers</h3>
          <p>Detected by model semantics.</p>
          <p class="subtitle">The AI model identified this text as subjective based on context, even though no specific keywords were found in the current lexicon.</p>
        </div>`;
      } else {
        content += `<div class="details-section"><p>No specific subjective keywords detected.</p></div>`;
      }
    }
    ui.analysisDetails.innerHTML = content;
  }

  function renderSelectionReport(state) {
    const container = ui.selectionReport;
    if (state.loading) {
      container.innerHTML = "<div class='selection-message'>Analyzing selection...</div>";
    } else if (state.error) {
      container.innerHTML = `<div class='selection-message selection-error'>${state.error}</div>`;
    } else if (state.result) {
      const res = state.result;
      container.innerHTML = `
        <div class="selection-summary">
          <strong>${res.label}</strong> (${Math.round(res.score)} pts)
          <span class="selection-meta">via ${res.model}</span>
        </div>
        <div class="selection-snippet">${state.text.slice(0, 100)}...</div>
      `;
    }
  }
})();
