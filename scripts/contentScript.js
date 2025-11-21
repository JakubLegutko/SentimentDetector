(function () {
  if (window.__sentimentBiasGaugesInjected || !document.body) {
    return;
  }
  window.__sentimentBiasGaugesInjected = true;

  const MAX_TEXT_LENGTH = 20000;
  const SENTENCE_SPLIT_REGEX = /(?<=[.!?])\s+/;
  const TOKEN_REGEX = /\p{L}[\p{L}'-]+/gu;
  const MAX_PHRASE_LENGTH = 3;
  const ENGLISH_LANG_CODES = new Set(["en", "en-us", "en-gb"]);
  const SENTIMENT_SENSITIVITY = 1.35;
  const POLITICAL_SENSITIVITY = 1.4;

  const POLITICAL_LEXICON = {
    left: {
      weight: 1,
      phrases: [
        "climate justice",
        "wealth tax",
        "green new deal",
        "racial equity",
        "universal healthcare",
        "labor union",
        "collective bargaining",
        "social safety net",
        "living wage",
        "regulation",
        "redistribution",
        "public option"
      ],
      unigrams: [
        "equity",
        "union",
        "solidarity",
        "progressive",
        "sustainability",
        "inclusive",
        "redistribute",
        "taxation",
        "regulate",
        "diversity",
        "activism",
        "collective",
        "public",
        "welfare"
      ]
    },
    right: {
      weight: 1,
      phrases: [
        "free market",
        "gun rights",
        "border security",
        "law and order",
        "fiscal conservatism",
        "school choice",
        "limited government",
        "national sovereignty",
        "pro life",
        "tax cuts",
        "strong defense",
        "traditional values"
      ],
      unigrams: [
        "patriotism",
        "liberty",
        "conservative",
        "freedom",
        "heritage",
        "capitalism",
        "faith",
        "traditional",
        "constitution",
        "security",
        "entrepreneurial",
        "border",
        "defense",
        "deregulation"
      ]
    }
  };

  const STOPWORDS = new Set([
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "with",
    "to",
    "of",
    "for",
    "on",
    "in",
    "that",
    "this",
    "it",
    "is",
    "are",
    "be",
    "as",
    "at",
    "by",
    "from"
  ]);

  let lastDetectedLanguage = "en";

  const ui = createPanel();
  let lastSignature = "";
  let updateCounter = 0;
  let selectionState = null;
  let lastAnalysis = null;

  if (ui.selectionButton) {
    ui.selectionButton.addEventListener("click", () => {
      handleSelectionAnalysis().catch((error) => {
        console.error("Selection sentiment analysis failed", error);
      });
    });
  }

  if (ui.detailsButton) {
    ui.detailsButton.addEventListener("click", () => {
      const expanded = ui.detailsButton.dataset.expanded === "true";
      const nextState = !expanded;
      ui.detailsButton.dataset.expanded = String(nextState);
      ui.detailsButton.textContent = nextState ? "Hide" : "Why?";
      ui.detailsButton.setAttribute("aria-pressed", String(nextState));
      ui.panel.classList.toggle("details-open", nextState);
      if (nextState) {
        if (lastAnalysis) {
          updateAnalysisDetails(lastAnalysis);
        } else {
          updateAnalysisDetails(null);
        }
      }
    });
  }

  const debouncedUpdate = debounce(() => {
    const runId = ++updateCounter;
    runAutomaticAnalysis(runId).catch((error) => {
      console.error("Sentiment Gauges failed to update", error);
    });
  }, 900);

  debouncedUpdate();

  const observer = new MutationObserver(() => debouncedUpdate());
  observer.observe(document.body, { childList: true, subtree: true, characterData: true });
  window.addEventListener("scroll", debouncedUpdate, { passive: true });
  window.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      debouncedUpdate();
    }
  });

  async function runAutomaticAnalysis(runId) {
    const rawText = extractVisibleText();
    if (!rawText.trim()) {
      lastSignature = "";
      setPanelStatus("Waiting for readable text...");
      return;
    }
    const signature = `${rawText.length}:${rawText.slice(0, 120)}`;
    if (signature === lastSignature) {
      return;
    }
    lastSignature = signature;

    setPanelStatus("Analyzing page...");
    const { sentimentResult, politicalResult, contextResult } = await analyzeText(rawText, {
      updateBadge: true
    });

    if (runId !== updateCounter) {
      return;
    }

    updatePositivityGauge(ui.positivity, sentimentResult, contextResult);
    updatePoliticalGauge(ui.political, politicalResult, contextResult);
    lastAnalysis = { sentimentResult, politicalResult, contextResult };
    if (ui.panel.classList.contains("details-open")) {
      updateAnalysisDetails(lastAnalysis);
    }
  }

  async function analyzeText(rawText, { updateBadge = false, preferredLanguage } = {}) {
    const detectedLanguage =
      preferredLanguage || detectLanguage(rawText) || lastDetectedLanguage || "en";
    if (updateBadge) {
      updateLanguageBadge(detectedLanguage);
    }
    lastDetectedLanguage = detectedLanguage;

    const normalizedText = rawText.toLowerCase();
    const tokens = tokenize(normalizedText);

    let sentimentResult;
    let provider = "lexicon";

    if (!isEnglish(detectedLanguage) && isMultilingualAvailable()) {
      try {
        sentimentResult = await window.MultilingualSentiment.analyze(rawText, detectedLanguage);
        provider = "transformer";
      } catch (error) {
        console.warn("Multilingual sentiment failed; falling back to lexicon", error);
        sentimentResult = scoreSentiment(tokens, normalizedText);
        provider = "lexicon-fallback";
      }
    } else {
      sentimentResult = scoreSentiment(tokens, normalizedText);
    }

    const politicalResult = await analyzePoliticalOrientation({
      text: rawText,
      tokens,
      language: detectedLanguage
    });
    const contextResult = {
      ...buildContextSummary(rawText),
      language: detectedLanguage,
      provider
    };

    return { sentimentResult, politicalResult, contextResult };
  }

  async function handleSelectionAnalysis() {
    const selectedText = getSelectionText();
    if (!selectedText) {
      renderSelectionReport({ error: "Select some text on the page first." });
      return;
    }

    renderSelectionReport({ loading: true });
    const originalLabel = ui.selectionButton?.textContent;
    if (ui.selectionButton) {
      ui.selectionButton.disabled = true;
      ui.selectionButton.textContent = "…";
    }

    try {
      const { sentimentResult, politicalResult, contextResult } = await analyzeText(selectedText, {
        preferredLanguage: lastDetectedLanguage
      });
      renderSelectionReport({
        sentimentResult,
        politicalResult,
        contextResult,
        text: selectedText
      });
    } catch (error) {
      console.error("Selection analysis failed", error);
      renderSelectionReport({
        error: "Unable to analyze the selected text. Please try again."
      });
    } finally {
      if (ui.selectionButton) {
        ui.selectionButton.disabled = false;
        ui.selectionButton.textContent = originalLabel || "Sel";
      }
    }
  }

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
      ".popup"
    ];

    ["script", "style", "noscript", "template"].forEach((tag) => {
      clone.querySelectorAll(tag).forEach((el) => el.remove());
    });
    NOISE_SELECTORS.forEach((selector) => {
      clone.querySelectorAll(selector).forEach((el) => el.remove());
    });

    const text = clone.innerText || "";
    return text.replace(/\s+/g, " ").trim();
  }

  function tokenize(text) {
    const tokens = text.match(TOKEN_REGEX) || [];
    return tokens.map((token) => token.toLowerCase());
  }

  function scoreSentiment(tokens, text) {
    if (!tokens.length) {
      return {
        normalized: 0.5,
        aggregate: 0,
        matches: 0,
        sampleSentences: [],
        descriptor: "Neutral"
      };
    }
    const { score, matches } = computeLexiconScore(tokens);

    const baseNormalized = matches ? clamp((score / (matches * 5) + 1) / 2, 0, 1) : 0.5;
    const normalized = applySensitivity(baseNormalized, SENTIMENT_SENSITIVITY);
    const descriptor =
      normalized > 0.62 ? "Positive" : normalized < 0.38 ? "Negative" : "Neutral";

    const sampleSentences = extractChargedSentences(text, 3);

    return {
      normalized,
      aggregate: score,
      matches,
      descriptor,
      sampleSentences
    };
  }

  function scorePoliticalOrientationLex(tokens) {
    if (!tokens.length) {
      return {
        normalized: 0.5,
        descriptor: "Center",
        leftScore: 0,
        rightScore: 0,
        provider: "lexicon"
      };
    }
    let leftScore = 0;
    let rightScore = 0;

    const joinedTokens = tokens.join(" ");

    POLITICAL_LEXICON.left.phrases.forEach((phrase) => {
      const count = countOccurrences(joinedTokens, phrase);
      leftScore += count * 2;
    });

    POLITICAL_LEXICON.right.phrases.forEach((phrase) => {
      const count = countOccurrences(joinedTokens, phrase);
      rightScore += count * 2;
    });

    tokens.forEach((token) => {
      if (POLITICAL_LEXICON.left.unigrams.includes(token)) {
        leftScore += 1;
      } else if (POLITICAL_LEXICON.right.unigrams.includes(token)) {
        rightScore += 1;
      }
    });

    const total = leftScore + rightScore;
    const balance = total ? (rightScore - leftScore) / total : 0;
    const baseNormalized = clamp(balance / 2 + 0.5, 0, 1);
    const normalized = applySensitivity(baseNormalized, POLITICAL_SENSITIVITY);
    let descriptor = "Center";
    if (normalized > 0.58) descriptor = "Right leaning";
    else if (normalized < 0.42) descriptor = "Left leaning";

    return { normalized, descriptor, leftScore, rightScore, provider: "lexicon" };
  }

  function buildContextSummary(text) {
    const sentences = (text.match(SENTENCE_SPLIT_REGEX) ? text.split(SENTENCE_SPLIT_REGEX) : [text])
      .map((sentence) => sentence.trim())
      .filter(Boolean);

    const wordCount = text.split(/\s+/).filter(Boolean).length;
    const topKeywords = extractTopKeywords(text);

    return {
      sentences,
      wordCount,
      topKeywords
    };
  }

  function extractChargedSentences(text, limit = 2) {
    const sentences = text
      .split(SENTENCE_SPLIT_REGEX)
      .map((sentence) => sentence.trim())
      .filter((sentence) => sentence.length > 0);
    const scored = sentences
      .map((sentence) => {
        const lower = sentence.toLowerCase();
        const tokens = tokenize(lower);
        const { score } = computeLexiconScore(tokens);
        return { sentence, weight: Math.abs(score) };
      })
      .sort((a, b) => b.weight - a.weight)
      .slice(0, limit);
    return scored.map((entry) => entry.sentence);
  }

  function computeLexiconScore(tokens) {
    let score = 0;
    let matches = 0;
    for (let i = 0; i < tokens.length; i += 1) {
      let matched = false;
      for (let window = Math.min(MAX_PHRASE_LENGTH, tokens.length - i); window > 1; window -= 1) {
        const phrase = tokens.slice(i, i + window).join(" ");
        const lexScore = AFINN_LEXICON[phrase];
        if (lexScore !== undefined) {
          score += lexScore;
          matches += 1;
          i += window - 1;
          matched = true;
          break;
        }
      }
      if (!matched) {
        const lexScore = AFINN_LEXICON[tokens[i]];
        if (lexScore !== undefined) {
          score += lexScore;
          matches += 1;
        }
      }
    }
    return { score, matches };
  }

  function extractTopKeywords(text) {
    const counts = new Map();
    const tokens = tokenize(text);
    tokens.forEach((token) => {
      if (STOPWORDS.has(token) || token.length < 4) return;
      counts.set(token, (counts.get(token) || 0) + 1);
    });
    return Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4)
      .map(([token, count]) => `${token} (${count})`);
  }

  async function analyzePoliticalOrientation({ text, tokens }) {
    if (typeof window.PoliticalBiasAnalyzer?.classify === "function") {
      try {
        const result = await window.PoliticalBiasAnalyzer.classify(text);
        return {
          normalized: applySensitivity(result.normalized, POLITICAL_SENSITIVITY),
          descriptor: result.descriptor,
          provider: result.provider,
          confidence: result.confidence,
          distribution: result.distribution
        };
      } catch (error) {
        console.warn("Political bias transformer failed; falling back to lexicon", error);
      }
    }
    return scorePoliticalOrientationLex(tokens);
  }

  function createPanel() {
    const panel = document.createElement("section");
    panel.id = "sentiment-detector-panel";
    panel.classList.add("sentiment-detector-fade-enter", "sentiment-detector-fade-enter-active");

    const header = document.createElement("div");
    header.className = "panel-header";

    const titleBlock = document.createElement("div");
    titleBlock.className = "panel-title-block";
    const title = document.createElement("h1");
    title.textContent = "Sentiment Gauges";
    const subtitle = document.createElement("p");
    subtitle.className = "subtitle";
    subtitle.textContent = "Live analysis of visible text";
    titleBlock.appendChild(title);
    titleBlock.appendChild(subtitle);

    const actions = document.createElement("div");
    actions.className = "panel-actions";
    const languageBadge = document.createElement("span");
    languageBadge.className = "language-badge";
    languageBadge.textContent = "EN";
    languageBadge.title = "Detected language: EN";

    const detailsButton = document.createElement("button");
    detailsButton.type = "button";
    detailsButton.className = "panel-action";
    detailsButton.textContent = "Why?";
    detailsButton.title = "Show reasoning behind the gauges";
    detailsButton.dataset.expanded = "false";

    const selectionButton = document.createElement("button");
    selectionButton.type = "button";
    selectionButton.className = "panel-action";
    selectionButton.textContent = "Sel";
    selectionButton.title = "Analyze the currently highlighted text";

    const toggleButton = document.createElement("button");
    toggleButton.type = "button";
    toggleButton.className = "panel-toggle";
    toggleButton.setAttribute("aria-label", "Minimize sentiment gauges");
    toggleButton.setAttribute("aria-expanded", "true");
    toggleButton.textContent = "–";
    toggleButton.addEventListener("click", () => {
      const collapsed = panel.classList.toggle("collapsed");
      toggleButton.textContent = collapsed ? "+" : "–";
      toggleButton.setAttribute(
        "aria-label",
        collapsed ? "Expand sentiment gauges" : "Minimize sentiment gauges"
      );
      toggleButton.setAttribute("aria-expanded", String(!collapsed));
    });
    actions.appendChild(languageBadge);
    actions.appendChild(detailsButton);
    actions.appendChild(selectionButton);
    actions.appendChild(toggleButton);

    header.appendChild(titleBlock);
    header.appendChild(actions);

    const body = document.createElement("div");
    body.className = "panel-body";

    const positivityRow = createGaugeRow({
      title: "Positivity",
      legend: "Negative ↔ Positive"
    });
    const politicalRow = createGaugeRow({
      title: "Political spectrum",
      legend: "Left ↔ Right"
    });

    body.appendChild(positivityRow.row);
    body.appendChild(politicalRow.row);

    const selectionReport = document.createElement("div");
    selectionReport.className = "selection-report";
    selectionReport.innerHTML =
      '<div class="selection-message">Highlight text and click "Sel" to analyze a snippet.</div>';

    const analysisDetails = document.createElement("div");
    analysisDetails.className = "analysis-details";
    analysisDetails.innerHTML =
      '<div class="details-message">Run an analysis to see contributing sentences, keywords, and political cues.</div>';

    panel.appendChild(header);
    panel.appendChild(body);
    panel.appendChild(selectionReport);
    panel.appendChild(analysisDetails);
    document.body.appendChild(panel);

    return {
      panel,
      body,
      toggleButton,
      languageBadge,
      detailsButton,
      selectionButton,
      selectionReport,
      analysisDetails,
      positivity: positivityRow,
      political: politicalRow
    };
  }

  function setPanelStatus(message) {
    ui.positivity.status.textContent = message;
    ui.political.status.textContent = message;
  }

  function updateLanguageBadge(language) {
    if (!ui.languageBadge) return;
    const label = (language || "en").toUpperCase();
    ui.languageBadge.textContent = label;
    ui.languageBadge.title = `Detected language: ${label}`;
  }

  function createGaugeRow({ title, legend }) {
    const row = document.createElement("div");
    row.className = "sentiment-row";

    const label = document.createElement("label");
    const titleSpan = document.createElement("span");
    titleSpan.textContent = title;
    const legendSpan = document.createElement("span");
    legendSpan.textContent = legend;
    label.appendChild(titleSpan);
    label.appendChild(legendSpan);

    const gauge = document.createElement("div");
    gauge.className = "gauge";
    const fill = document.createElement("div");
    fill.className = "gauge-fill";
    const indicator = document.createElement("div");
    indicator.className = "gauge-indicator";
    indicator.textContent = "•";

    gauge.appendChild(fill);
    gauge.appendChild(indicator);

    const status = document.createElement("small");
    status.className = "context";
    status.textContent = "Collecting text...";

    row.appendChild(label);
    row.appendChild(gauge);
    row.appendChild(status);

    return { row, fill, indicator, status };
  }

  function updatePositivityGauge(elements, result, context) {
    const width = Math.round(result.normalized * 100);
    elements.fill.style.width = `${width}%`;
    elements.fill.style.background =
      result.normalized >= 0.5
        ? "linear-gradient(90deg, #34d399, #10b981)"
        : "linear-gradient(90deg, #f87171, #ef4444)";
    elements.indicator.style.left = `${width}%`;
    elements.indicator.textContent = `${Math.round((result.normalized - 0.5) * 200)}%`;
    const confidenceLabel =
      context.provider === "transformer" && typeof result.aggregate === "number"
        ? `conf: ${(result.aggregate * 100).toFixed(0)}%`
        : `matches: ${result.matches}`;
    elements.status.textContent = [
      `${result.descriptor} • ${confidenceLabel}`,
      context.wordCount ? `${context.wordCount} words processed` : "",
      context.language ? `lang: ${context.language.toUpperCase()}` : "",
      context.provider ? `model: ${context.provider}` : "",
      context.topKeywords.length ? `focus: ${context.topKeywords.join(", ")}` : ""
    ]
      .filter(Boolean)
      .join(" · ");
  }

  function updatePoliticalGauge(elements, result, context) {
    const width = Math.round(result.normalized * 100);
    elements.fill.style.width = `${width}%`;
    elements.fill.style.background = "linear-gradient(90deg, #60a5fa, #f87171)";
    elements.indicator.style.left = `${width}%`;
    elements.indicator.textContent = result.descriptor;
    const distributionText = result.distribution
      ? `L:${formatPercent(result.distribution["Left leaning"])} C:${formatPercent(
          result.distribution.Center
        )} R:${formatPercent(result.distribution["Right leaning"])}`
      : `Left refs: ${result.leftScore} · Right refs: ${result.rightScore}`;
    elements.status.textContent = [
      distributionText,
      result.provider ? `model: ${result.provider}` : "",
      result.confidence ? `conf: ${Math.round(result.confidence * 100)}%` : "",
      context.language ? `lang: ${context.language.toUpperCase()}` : "",
      context.topKeywords.length ? `focus: ${context.topKeywords.join(", ")}` : ""
    ]
      .filter(Boolean)
      .join(" · ");
  }

  function renderSelectionReport(state) {
    selectionState = state;
    if (!ui.selectionReport) return;

    if (!state) {
      ui.selectionReport.innerHTML =
        '<div class="selection-message">Highlight text and click "Sel" to analyze a snippet.</div>';
      return;
    }

    if (state.loading) {
      ui.selectionReport.innerHTML =
        '<div class="selection-message">Analyzing highlighted text…</div>';
      return;
    }

    if (state.error) {
      ui.selectionReport.innerHTML = `<div class="selection-message selection-error">${state.error}</div>`;
      return;
    }

    const { sentimentResult, politicalResult, contextResult, text } = state;
    const positivityPercent = Math.round(sentimentResult.normalized * 100);
    const metaParts = [];
    if (contextResult.language) metaParts.push(`Lang: ${contextResult.language.toUpperCase()}`);
    if (contextResult.provider) metaParts.push(`Model: ${contextResult.provider}`);
    if (contextResult.wordCount) metaParts.push(`Words: ${contextResult.wordCount}`);

    const keywords =
      contextResult.topKeywords && contextResult.topKeywords.length
        ? `<div class="selection-insights"><strong>Focus:</strong> ${contextResult.topKeywords.join(
            ", "
          )}</div>`
        : "";

    const sentences =
      sentimentResult.sampleSentences && sentimentResult.sampleSentences.length
        ? `<div class="selection-insights"><strong>Notable lines:</strong><ul>${sentimentResult.sampleSentences
            .map((sentence) => `<li>${sentence}</li>`)
            .join("")}</ul></div>`
        : "";

    ui.selectionReport.innerHTML = `
      <div class="selection-summary">
        <div>
          <strong>${sentimentResult.descriptor}</strong> (${positivityPercent}%)
          <span class="selection-meta">${metaParts.join(" · ")}</span>
        </div>
        <div class="selection-meta">Political: ${politicalResult.descriptor}${
          politicalResult.provider ? ` (${politicalResult.provider})` : ""
        }</div>
      </div>
      <div class="selection-meta">${buildPoliticalMetaSummary(politicalResult)}</div>
      ${keywords}
      ${sentences}
      <div class="selection-snippet">${text.slice(0, 240)}${text.length > 240 ? "…" : ""}</div>
    `;
  }

  function updateAnalysisDetails(analysis) {
    if (!ui.analysisDetails) return;
    if (!analysis) {
      ui.analysisDetails.innerHTML =
        '<div class="details-message">No analysis yet. Scroll or select text to trigger the gauges.</div>';
      return;
    }

    const { sentimentResult, politicalResult, contextResult } = analysis;
    const sentimentLines =
      (sentimentResult.sampleSentences && sentimentResult.sampleSentences.length
        ? sentimentResult.sampleSentences
        : contextResult.sentences.slice(0, 2)) || [];
    const keywords =
      contextResult.topKeywords && contextResult.topKeywords.length
        ? contextResult.topKeywords
        : [];

    const providerLabel =
      contextResult.provider === "transformer"
        ? "Multilingual transformer"
        : contextResult.provider === "lexicon-fallback"
          ? "AFINN lexicon (fallback)"
          : "AFINN lexicon";

    const keywordMarkup = keywords.length
      ? `<ul>${keywords.map((keyword) => `<li>${keyword}</li>`).join("")}</ul>`
      : "<p>No standout keywords detected.</p>";

    const sentenceMarkup = sentimentLines.length
      ? `<ol>${sentimentLines.map((line) => `<li>${line}</li>`).join("")}</ol>`
      : "<p>No specific sentences highlighted.</p>";

    ui.analysisDetails.innerHTML = `
      <div class="details-section">
        <h3>Sentiment drivers</h3>
        <p><strong>${sentimentResult.descriptor}</strong> (${Math.round(sentimentResult.normalized * 100)}%) · ${providerLabel}</p>
        ${sentenceMarkup}
      </div>
      <div class="details-section">
        <h3>Top keywords</h3>
        ${keywordMarkup}
      </div>
      <div class="details-section">
        <h3>Political cues</h3>
        <p>${buildPoliticalMetaSummary(politicalResult)}</p>
        <p>Detected language: ${contextResult.language?.toUpperCase() ?? "EN"}</p>
      </div>
    `;
  }

  function buildPoliticalMetaSummary(result) {
    if (!result) {
      return "No political data";
    }
    if (result.distribution) {
      return `L:${formatPercent(result.distribution["Left leaning"])} C:${formatPercent(
        result.distribution.Center
      )} R:${formatPercent(result.distribution["Right leaning"])}`;
    }
    return `Left refs: ${result.leftScore ?? 0} · Right refs: ${result.rightScore ?? 0}`;
  }

  function isMultilingualAvailable() {
    return (
      typeof window.MultilingualSentiment !== "undefined" &&
      typeof window.MultilingualSentiment.analyze === "function"
    );
  }

  function isEnglish(languageCode) {
    if (!languageCode) return true;
    const normalized = languageCode.toLowerCase();
    return normalized === "en" || ENGLISH_LANG_CODES.has(normalized);
  }

  function detectLanguage(text) {
    const attrLang = normalizeLang(document.documentElement.getAttribute("lang"));
    if (attrLang) return attrLang;
    const meta = document.querySelector("meta[http-equiv='content-language']");
    if (meta?.content) {
      const metaLang = normalizeLang(meta.content.split(",")[0]);
      if (metaLang) return metaLang;
    }
    const asciiRatio = text ? text.replace(/[^\x00-\x7F]/g, "").length / text.length : 1;
    if (asciiRatio < 0.85) {
      const navigatorLang = normalizeLang(
        (navigator.language || (Array.isArray(navigator.languages) ? navigator.languages[0] : "")) ??
          ""
      );
      if (navigatorLang) return navigatorLang;
      return "auto";
    }
    return "en";
  }

  function normalizeLang(value = "") {
    return value ? value.toLowerCase().split("-")[0].trim() : "";
  }

  function getSelectionText() {
    const selection = window.getSelection && window.getSelection();
    return selection ? selection.toString().trim() : "";
  }

  function formatPercent(value) {
    if (typeof value !== "number" || Number.isNaN(value)) {
      return "0%";
    }
    return `${Math.round(value * 100)}%`;
  }

  function debounce(fn, delay) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      timer = window.setTimeout(() => fn(...args), delay);
    };
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function applySensitivity(normalized, factor) {
    const delta = normalized - 0.5;
    return clamp(0.5 + delta * factor, 0, 1);
  }

  function countOccurrences(haystack, needle) {
    if (!haystack || !needle) return 0;
    const pattern = new RegExp(`\\b${escapeRegex(needle)}\\b`, "g");
    const matches = haystack.match(pattern);
    return matches ? matches.length : 0;
  }

  function escapeRegex(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }
})();

