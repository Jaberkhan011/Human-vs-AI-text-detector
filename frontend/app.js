(function () {
  "use strict";

  var API_BASE_URL = window.TEXTGUARD_API_BASE_URL || "http://127.0.0.1:2200";
  var MIN_RELIABLE_WORDS = 25;
  var REQUEST_TIMEOUT_MS = 60000;

  // Elements
  var textInput = document.getElementById("text-input");
  var wordCountEl = document.getElementById("word-count");
  var charCountEl = document.getElementById("char-count");
  var shortWarningEl = document.getElementById("short-warning");
  var analyzeBtn = document.getElementById("analyze-btn");
  var clearBtn = document.getElementById("clear-btn");

  var emptyState = document.getElementById("empty-state");
  var loadingState = document.getElementById("loading-state");
  var errorState = document.getElementById("error-state");
  var errorMessageEl = document.getElementById("error-message");
  var resultState = document.getElementById("result-state");

  var verdictDot = document.getElementById("verdict-dot");
  var verdictHeadline = document.getElementById("verdict-headline");
  var probHuman = document.getElementById("prob-human");
  var probAi = document.getElementById("prob-ai");
  var humanPctEl = document.getElementById("human-pct");
  var aiPctEl = document.getElementById("ai-pct");
  var confidenceTicks = document.getElementById("confidence-ticks");
  var confidenceValueEl = document.getElementById("confidence-value");

  var featureToggle = document.getElementById("feature-toggle");
  var featureToggleLabel = document.getElementById("feature-toggle-label");
  var featureGrid = document.getElementById("feature-grid");

  var CONFIDENCE_LEVELS = ["Low", "Medium", "High"];

  var FEATURE_DEFS = [
    { key: "Word_Count", label: "Word count" },
    { key: "Character_Count_Total", label: "Character count" },
    { key: "Sentence_Count", label: "Sentence count" },
    { key: "Average_Word_Length", label: "Avg. word length", format: fixed(2) },
    { key: "Average_Sentence_Length_Words", label: "Avg. sentence length", format: fixed(2) },
    { key: "Lexical_Diversity_Score", label: "Lexical diversity", format: fixed(2) },
    { key: "Punctuation_Density", label: "Punctuation density", format: fixed(3) },
    { key: "Stopword_Density", label: "Stopword density", format: fixed(2) },
    { key: "Flesch_Kincaid_Grade_Level", label: "Flesch-Kincaid grade", format: fixed(1) },
    { key: "Smog_Index", label: "SMOG index", format: fixed(1) },
    { key: "Average_Syllables_Per_Word", label: "Avg. syllables/word", format: fixed(2) },
    { key: "Capital_Letter_Count", label: "Capital letters" },
    { key: "Burstiness_Score", label: "Burstiness score", format: fixed(1) },
    { key: "Contains_Code_Snippet", label: "Code detected", format: yesNo }
  ];

  function fixed(digits) {
    return function (value) {
      return Number(value).toFixed(digits);
    };
  }

  function yesNo(value) {
    return value ? "Yes" : "No";
  }

  // ---- Live stats ----

  function countWords(text) {
    var trimmed = text.trim();
    return trimmed ? trimmed.split(/\s+/).length : 0;
  }

  function updateStats() {
    var text = textInput.value;
    var words = countWords(text);
    var chars = text.length;

    wordCountEl.textContent = words + " words";
    charCountEl.textContent = chars + " characters";

    var isShort = words > 0 && words < MIN_RELIABLE_WORDS;
    shortWarningEl.hidden = !isShort;

    analyzeBtn.disabled = text.trim().length === 0;
    clearBtn.disabled = text.length === 0;
  }

  textInput.addEventListener("input", updateStats);

  // ---- Report state machine ----

  function showState(name) {
    emptyState.hidden = name !== "empty";
    loadingState.hidden = name !== "loading";
    errorState.hidden = name !== "error";
    resultState.hidden = name !== "result";
  }

  function setLoading(isLoading) {
    analyzeBtn.disabled = isLoading || textInput.value.trim().length === 0;
    clearBtn.disabled = isLoading || textInput.value.length === 0;
    textInput.disabled = isLoading;
    analyzeBtn.textContent = isLoading ? "Analyzing…" : "Analyze Text";
  }

  // ---- API ----

  function analyzeText(text) {
    var controller = new AbortController();
    var timeoutId = setTimeout(function () {
      controller.abort();
    }, REQUEST_TIMEOUT_MS);

    return fetch(API_BASE_URL + "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
      signal: controller.signal
    })
      .then(function (response) {
        if (!response.ok) {
          return response
            .json()
            .catch(function () {
              return null;
            })
            .then(function (body) {
              var detail = body && typeof body.detail === "string" ? body.detail : null;
              throw new Error(detail || "The server couldn't process that text. Please try again.");
            });
        }
        return response.json();
      })
      .then(function (data) {
        if (!isPredictionResponse(data)) {
          throw new Error("The server returned an unexpected response.");
        }
        return data;
      })
      .catch(function (err) {
        if (err && err.name === "AbortError") {
          throw new Error("The request timed out. Please check your connection and try again.");
        }
        if (err instanceof Error && err.message) {
          throw err;
        }
        throw new Error(
          "Unable to analyze the text right now. Please check that the server is running and try again."
        );
      })
      .finally(function () {
        clearTimeout(timeoutId);
      });
  }

  function isPredictionResponse(data) {
    return (
      data &&
      typeof data === "object" &&
      typeof data.label === "string" &&
      typeof data.ai_probability === "number" &&
      typeof data.human_probability === "number" &&
      typeof data.confidence === "string" &&
      data.features &&
      typeof data.features === "object"
    );
  }

  // ---- Rendering ----

  function renderResult(result) {
    var isAi = result.label === "AI";

    verdictDot.className = "verdict-dot " + (isAi ? "is-ai" : "is-human");
    verdictHeadline.textContent = isAi ? "Likely AI-generated" : "Likely Human-written";

    var aiPct = Math.round(result.ai_probability * 100);
    var humanPct = Math.round(result.human_probability * 100);

    // Reset widths first so the fill transition replays on each analysis
    probHuman.style.width = "0%";
    probAi.style.width = "0%";
    requestAnimationFrame(function () {
      probHuman.style.width = humanPct + "%";
      probAi.style.width = aiPct + "%";
    });

    document
      .getElementById("prob-bar")
      .setAttribute("aria-label", humanPct + "% human probability, " + aiPct + "% AI probability");

    humanPctEl.textContent = humanPct + "%";
    aiPctEl.textContent = aiPct + "%";

    var activeIndex = CONFIDENCE_LEVELS.indexOf(result.confidence);
    confidenceTicks.innerHTML = "";
    CONFIDENCE_LEVELS.forEach(function (level, i) {
      var tick = document.createElement("span");
      tick.className = "confidence-tick" + (i <= activeIndex ? " is-active" : "");
      confidenceTicks.appendChild(tick);
    });
    confidenceValueEl.textContent = result.confidence;

    renderFeatures(result.features);

    // Collapse the feature grid on each new result
    featureGrid.hidden = true;
    featureToggle.setAttribute("aria-expanded", "false");
    featureToggleLabel.textContent = "show";
  }

  function renderFeatures(features) {
    featureGrid.innerHTML = "";
    FEATURE_DEFS.forEach(function (def) {
      var raw = features[def.key];
      var value = def.format ? def.format(raw) : String(raw);

      var dt = document.createElement("dt");
      dt.textContent = def.label;

      var dd = document.createElement("dd");
      dd.textContent = value;

      featureGrid.appendChild(dt);
      featureGrid.appendChild(dd);
      featureGrid.appendChild(document.createElement("br")); // separator
    });
  }

  featureToggle.addEventListener("click", function () {
    var isHidden = featureGrid.hidden;
    featureGrid.hidden = !isHidden;
    featureToggle.setAttribute("aria-expanded", String(isHidden));
    featureToggleLabel.textContent = isHidden ? "hide" : "show";
  });

  // ---- Actions ----

  function handleAnalyze() {
    var text = textInput.value.trim();
    if (!text) return;

    showState("loading");
    setLoading(true);

    analyzeText(text)
      .then(function (result) {
        renderResult(result);
        showState("result");
      })
      .catch(function (err) {
        errorMessageEl.textContent = err.message;
        showState("error");
      })
      .finally(function () {
        setLoading(false);
      });
  }

  function handleClear() {
    textInput.value = "";
    updateStats();
    showState("empty");
  }

  analyzeBtn.addEventListener("click", handleAnalyze);
  clearBtn.addEventListener("click", handleClear);

  // Initial state
  updateStats();
  showState("empty");
})();
