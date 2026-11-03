/**
 * Frontend Application Controller for TeluguPeople Bilingual Voice Assistant
 * Team Vaani - VIT-AP University
 */

document.addEventListener("DOMContentLoaded", () => {
  // State
  let activeModel = "bge_m3"; // Default BGE-M3 as shown in screenshot
  let currentArticle = null;
  let currentAnswer = "";
  let currentAudioUrl = null;

  // DOM Elements
  const articleUrlInput = document.getElementById("articleUrlInput");
  const loadArticleBtn = document.getElementById("loadArticleBtn");
  const tryTeluguBtn = document.getElementById("tryTeluguBtn");
  const tryEnglishBtn = document.getElementById("tryEnglishBtn");
  
  const articleMetaBox = document.getElementById("articleMetaBox");
  const metaLangBadge = document.getElementById("metaLangBadge");
  const metaCategory = document.getElementById("metaCategory");
  const metaTitle = document.getElementById("metaTitle");
  const metaWords = document.getElementById("metaWords");
  const metaChunks = document.getElementById("metaChunks");
  const metaNewsId = document.getElementById("metaNewsId");
  const toggleArticleBodyBtn = document.getElementById("toggleArticleBodyBtn");
  const articleBodyPreview = document.getElementById("articleBodyPreview");
  const articleBodyText = document.getElementById("articleBodyText");

  const cardE5 = document.getElementById("cardE5");
  const cardBge = document.getElementById("cardBge");

  const questionInput = document.getElementById("questionInput");
  const micBtn = document.getElementById("micBtn");
  const askBtn = document.getElementById("askBtn");
  const summarizeBtn = document.getElementById("summarizeBtn");
  const compareBtn = document.getElementById("compareBtn");
  const promptChips = document.querySelectorAll(".chip");

  const loadingIndicator = document.getElementById("loadingIndicator");
  const loadingText = document.getElementById("loadingText");
  const resultsSection = document.getElementById("resultsSection");
  const answerText = document.getElementById("answerText");
  const playAudioBtn = document.getElementById("playAudioBtn");
  const ttsAudio = document.getElementById("ttsAudio");
  const retrievalModelBadge = document.getElementById("retrievalModelBadge");
  const retrievalDimBadge = document.getElementById("retrievalDimBadge");
  const retrievalLatencyBadge = document.getElementById("retrievalLatencyBadge");
  const chunksList = document.getElementById("chunksList");

  const comparisonBox = document.getElementById("comparisonBox");
  const e5CompMeta = document.getElementById("e5CompMeta");
  const e5CompChunks = document.getElementById("e5CompChunks");
  const bgeCompMeta = document.getElementById("bgeCompMeta");
  const bgeCompChunks = document.getElementById("bgeCompChunks");

  // ============================================================
  // Model Selection Switching
  // ============================================================
  function selectModel(modelId) {
    activeModel = modelId;
    if (modelId === "e5") {
      cardE5.classList.add("selected");
      cardBge.classList.remove("selected");
    } else {
      cardBge.classList.add("selected");
      cardE5.classList.remove("selected");
    }
    console.log(`[Model Switch] Active model is now: ${modelId.toUpperCase()}`);
    fetch("/api/switch-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId })
    }).catch(console.error);
  }

  cardE5.addEventListener("click", () => selectModel("e5"));
  cardBge.addEventListener("click", () => selectModel("bge_m3"));

  // ============================================================
  // Article Loading
  // ============================================================
  async function loadArticle(url) {
    if (!url) return;
    articleUrlInput.value = url;
    loadArticleBtn.disabled = true;
    loadArticleBtn.textContent = "Loading...";

    try {
      const resp = await fetch("/api/load-article", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url })
      });

      if (!resp.ok) {
        let errMsg = `Status ${resp.status}`;
        try {
          const errData = await resp.json();
          if (errData.error) errMsg = errData.error;
        } catch(e) {}
        throw new Error(errMsg);
      }

      const data = await resp.json();
      currentArticle = data.article;

      // Populate UI
      metaTitle.textContent = currentArticle.title;
      metaCategory.textContent = currentArticle.category || "General News";
      metaLangBadge.textContent = currentArticle.language === "te" ? "TELUGU" : "ENGLISH";
      metaWords.textContent = currentArticle.text.split(/\s+/).length;
      metaChunks.textContent = data.chunk_count;
      metaNewsId.textContent = currentArticle.news_id || "N/A";
      articleBodyText.textContent = currentArticle.text;

      articleMetaBox.classList.remove("hidden");
      resultsSection.classList.add("hidden");
      comparisonBox.classList.add("hidden");

      console.log(`✅ Loaded article with ${data.chunk_count} chunks.`);
    } catch (err) {
      alert(`Error loading article: ${err.message}`);
    } finally {
      loadArticleBtn.disabled = false;
      loadArticleBtn.textContent = "Load article";
    }
  }

  loadArticleBtn.addEventListener("click", () => {
    loadArticle(articleUrlInput.value.trim());
  });

  tryTeluguBtn.addEventListener("click", (e) => {
    e.preventDefault();
    loadArticle("https://telugupeople.com/news/news.asp?newsID=108267");
  });

  tryEnglishBtn.addEventListener("click", (e) => {
    e.preventDefault();
    loadArticle("https://telugupeople.com/news/news.asp?newsId=108330");
  });

  toggleArticleBodyBtn.addEventListener("click", () => {
    const isHidden = articleBodyPreview.classList.toggle("hidden");
    toggleArticleBodyBtn.textContent = isHidden ? "View Text ▼" : "Hide Text ▲";
  });

  // Prompt Chips
  promptChips.forEach(chip => {
    chip.addEventListener("click", () => {
      questionInput.value = chip.getAttribute("data-q");
      executeAskQuestion();
    });
  });

  // ============================================================
  // Question & Query Execution
  // ============================================================
  async function executeAskQuestion(isSummary = false) {
    const query = questionInput.value.trim();
    if (!query && !isSummary) {
      alert("Please enter a question or select a prompt.");
      return;
    }

    loadingText.textContent = `Retrieving with ${activeModel === "e5" ? "multilingual-E5-base" : "BGE-M3"}...`;
    loadingIndicator.classList.remove("hidden");
    resultsSection.classList.add("hidden");
    comparisonBox.classList.add("hidden");

    try {
      const resp = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: isSummary ? "Summarize this article" : query,
          model_id: activeModel,
          is_summary: isSummary
        })
      });

      const data = await resp.json();
      currentAnswer = data.answer;
      currentAudioUrl = data.audio_url || null;

      // Populate Answer
      answerText.textContent = data.answer;
      retrievalModelBadge.textContent = `Model: ${data.model_name}`;
      retrievalDimBadge.textContent = `Dim: ${data.dim}d`;
      retrievalLatencyBadge.textContent = `Retrieval: ${data.latency_ms} ms`;

      // Populate Chunks
      renderChunks(data.chunks || []);

      resultsSection.classList.remove("hidden");
    } catch (err) {
      alert(`Query failed: ${err.message}`);
    } finally {
      loadingIndicator.classList.add("hidden");
    }
  }

  askBtn.addEventListener("click", () => executeAskQuestion(false));
  summarizeBtn.addEventListener("click", () => executeAskQuestion(true));
  questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") executeAskQuestion(false);
  });

  // ============================================================
  // Render Retrieved Chunks
  // ============================================================
  function renderChunks(chunks) {
    chunksList.innerHTML = "";
    if (!chunks.length) {
      chunksList.innerHTML = "<p class='text-muted'>No chunks retrieved.</p>";
      return;
    }

    chunks.forEach((chunk, idx) => {
      const card = document.createElement("div");
      card.className = "chunk-card";
      card.innerHTML = `
        <div class="chunk-header">
          <span>Rank #${chunk.rank || idx + 1} (${chunk.chunk_id})</span>
          <span>Similarity Score: <span class="score-badge">${chunk.score}</span></span>
        </div>
        <div class="chunk-text">${chunk.text}</div>
      `;
      chunksList.appendChild(card);
    });
  }

  // ============================================================
  // Side-by-Side Dual Model Comparison Shootout
  // ============================================================
  compareBtn.addEventListener("click", async () => {
    const query = questionInput.value.trim() || "What is this article about?";
    loadingText.textContent = "Benchmarking Multilingual-E5 vs BGE-M3 side-by-side...";
    loadingIndicator.classList.remove("hidden");

    try {
      const resp = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query })
      });

      const data = await resp.json();

      // Render E5 side
      e5CompMeta.textContent = `Retrieval Latency: ${data.e5.latency_ms} ms | Dim: 768d`;
      e5CompChunks.innerHTML = data.e5.results.map(r => `
        <div class="chunk-card" style="margin-top: 8px;">
          <div class="chunk-header">
            <span>Rank #${r.rank} (${r.chunk_id})</span>
            <span class="score-badge">${r.score}</span>
          </div>
          <div class="chunk-text" style="font-size: 13px;">${r.text.substring(0, 160)}...</div>
        </div>
      `).join("");

      // Render BGE side
      bgeCompMeta.textContent = `Retrieval Latency: ${data.bge_m3.latency_ms} ms | Dim: 1024d`;
      bgeCompChunks.innerHTML = data.bge_m3.results.map(r => `
        <div class="chunk-card" style="margin-top: 8px;">
          <div class="chunk-header">
            <span>Rank #${r.rank} (${r.chunk_id})</span>
            <span class="score-badge">${r.score}</span>
          </div>
          <div class="chunk-text" style="font-size: 13px;">${r.text.substring(0, 160)}...</div>
        </div>
      `).join("");

      comparisonBox.classList.remove("hidden");
      resultsSection.classList.remove("hidden");
    } catch (err) {
      alert(`Comparison failed: ${err.message}`);
    } finally {
      loadingIndicator.classList.add("hidden");
    }
  });

  // ============================================================
  // Audio Playback (Web Speech API + Backend fallback)
  // ============================================================
  playAudioBtn.addEventListener("click", () => {
    if (!currentAnswer) return;

    // Check if browser Web Speech API is available
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(currentAnswer);
      
      // Check if Telugu characters exist
      const hasTelugu = /[\u0C00-\u0C7F]/.test(currentAnswer);
      utterance.lang = hasTelugu ? "te-IN" : "en-US";
      utterance.rate = 1.0;

      playAudioBtn.textContent = "🔊 Playing...";
      utterance.onend = () => { playAudioBtn.textContent = "🔊 Listen (Voice)"; };
      utterance.onerror = () => { playAudioBtn.textContent = "🔊 Listen (Voice)"; };
      window.speechSynthesis.speak(utterance);
    } else if (currentAudioUrl) {
      ttsAudio.src = currentAudioUrl;
      ttsAudio.play();
    } else {
      alert("Audio synthesis is not supported on this browser.");
    }
  });

  // ============================================================
  // Microphone / Speech-to-Text Input
  // ============================================================
  micBtn.addEventListener("click", () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Speech recognition is not supported in this browser. Please type your question.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "te-IN"; // Supports Telugu or mixed English
    recognition.continuous = false;

    micBtn.textContent = "🔴 Listening...";
    recognition.onresult = (event) => {
      const speechText = event.results[0][0].transcript;
      questionInput.value = speechText;
      micBtn.textContent = "🎤";
      executeAskQuestion();
    };
    recognition.onerror = () => { micBtn.textContent = "🎤"; };
    recognition.onend = () => { micBtn.textContent = "🎤"; };
    recognition.start();
  });

  // Initialize: Load default Telugu article on startup
  loadArticle("https://telugupeople.com/news/news.asp?newsID=108267");
});
