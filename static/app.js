/**
 * Frontend Application Controller – Bilingual Telugu–English News Assistant
 * Team Vaani · VIT-AP University
 * v9: Dark Mode, Article Feed, Named Entity Highlighting, Improved TTS, Share/Export
 */

document.addEventListener("DOMContentLoaded", () => {

  // ============================================================
  // State
  // ============================================================
  let activeModel   = "bge_m3";
  let currentArticle = null;
  let currentAnswer  = "";
  let currentQuery   = "";
  let currentAudioUrl = null;
  let ttsUtterance   = null;
  let isSpeaking     = false;
  let qaHistory      = []; // [{q, a, model, latency}]

  // All cached articles (fed from backend + hardcoded fallback)
  const FEED_ARTICLES = [
    { news_id: "108267", lang: "te", title: "హోంవర్క్‌ విషయంలో ఉపాధ్యాయురాలి మందలింపు.. 6 ఏళ్ల బాలుడు మృతి",
      url: "https://telugupeople.com/news/news.asp?newsID=108267" },
    { news_id: "108330", lang: "en", title: "Keerthy Suresh Joins Nani in 'The Paradise'",
      url: "https://telugupeople.com/news/news.asp?newsId=108330" },
    { news_id: "109199", lang: "en", title: "Ram Charan and Sukumar Reunite for RC17",
      url: "https://www.telugupeople.com/news/news.asp?newsId=109199" },
    { news_id: "109200", lang: "en", title: "Cyclone Murasu Hits Andhra Coast, Causes Widespread Damage",
      url: "https://www.telugupeople.com/news/news.asp?newsID=109200" },
    { news_id: "109197", lang: "en", title: "Telangana High Court Seeks Report on Damagundam Forest",
      url: "https://www.telugupeople.com/news/news.asp?newsID=109197" },
    { news_id: "109196", lang: "en", title: "Pune AI Summit Showcases 54 Student Projects",
      url: "https://www.telugupeople.com/news/news.asp?newsID=109196" },
    { news_id: "109194", lang: "en", title: "Telangana Groundwater Report Flags Contamination Concerns",
      url: "https://www.telugupeople.com/news/news.asp?newsID=109194" },
    { news_id: "109193", lang: "en", title: "Tamil Nadu Deploys 500 Electricity Personnel to Andhra Pradesh",
      url: "https://www.telugupeople.com/news/news.asp?newsID=109193" },
    { news_id: "109192", lang: "en", title: "Severe Cyclone Murasu Approaches Andhra Coast",
      url: "https://www.telugupeople.com/news/news.asp?newsID=109192" },
    { news_id: "109191", lang: "en", title: "South Korean Firms Plan ₹1,614-Crore Investment in Andhra Pradesh",
      url: "https://www.telugupeople.com/news/news.asp?newsID=109191" },
    { news_id: "109188", lang: "en", title: "TCS HyperVault Plans ₹70,000-Crore AI Data Centre",
      url: "https://www.telugupeople.com/news/news.asp?newsID=109188" },
    { news_id: "108822", lang: "te", title: "నోబెల్ శాంతి పురస్కారానికి నవీ పిళ్లై ఎంపిక",
      url: "https://telugupeople.com/news/news.asp?newsID=108822" },
    { news_id: "108256", lang: "te", title: "అనకాపల్లిలో ₹1.36 లక్షల కోట్ల భారీ ఉక్కు కర్మాగారం",
      url: "https://telugupeople.com/news/news.asp?newsID=108256" },
    { news_id: "108224", lang: "te", title: "ఆంధ్రప్రదేశ్‌లో పాఠశాల ప్రహరీ గోడ కూలి 10వ తరగతి విద్యార్థి మృతి",
      url: "https://telugupeople.com/news/news.asp?newsID=108224" },
    { news_id: "108187", lang: "te", title: "కోటి మంది యువతకు ఏఐ శిక్షణ: ప్రధాని మోదీ కీలక ప్రకటన",
      url: "https://telugupeople.com/news/news.asp?newsID=108187" },
    { news_id: "108008", lang: "te", title: "జగన్ 2.0 సూపర్ యాప్‌ను ప్రారంభించిన వైఎస్సార్‌సీపీ",
      url: "https://telugupeople.com/news/news.asp?newsID=108008" },
  ];

  // ============================================================
  // DOM refs
  // ============================================================
  const articleUrlInput   = document.getElementById("articleUrlInput");
  const loadArticleBtn    = document.getElementById("loadArticleBtn");
  const articleMetaBox    = document.getElementById("articleMetaBox");
  const metaLangBadge     = document.getElementById("metaLangBadge");
  const metaCategory      = document.getElementById("metaCategory");
  const metaTitle         = document.getElementById("metaTitle");
  const metaWords         = document.getElementById("metaWords");
  const metaChunks        = document.getElementById("metaChunks");
  const metaNewsId        = document.getElementById("metaNewsId");
  const toggleArticleBodyBtn = document.getElementById("toggleArticleBodyBtn");
  const articleBodyPreview   = document.getElementById("articleBodyPreview");
  const articleBodyText      = document.getElementById("articleBodyText");
  const entityTagsRow        = document.getElementById("entityTagsRow");

  const cardE5  = document.getElementById("cardE5");
  const cardBge = document.getElementById("cardBge");

  const questionInput    = document.getElementById("questionInput");
  const micBtn           = document.getElementById("micBtn");
  const askBtn           = document.getElementById("askBtn");
  const summarizeBtn     = document.getElementById("summarizeBtn");
  const compareBtn       = document.getElementById("compareBtn");
  const promptChips      = document.querySelectorAll(".chip");

  const loadingIndicator = document.getElementById("loadingIndicator");
  const loadingText      = document.getElementById("loadingText");
  const resultsSection   = document.getElementById("resultsSection");
  const answerText       = document.getElementById("answerText");

  const playAudioBtn   = document.getElementById("playAudioBtn");
  const stopAudioBtn   = document.getElementById("stopAudioBtn");
  const ttsVoiceSelect = document.getElementById("ttsVoiceSelect");
  const ttsAudio       = document.getElementById("ttsAudio");

  const copyAnswerBtn  = document.getElementById("copyAnswerBtn");
  const exportPdfBtn   = document.getElementById("exportPdfBtn");

  const retrievalModelBadge  = document.getElementById("retrievalModelBadge");
  const retrievalDimBadge    = document.getElementById("retrievalDimBadge");
  const retrievalLatencyBadge = document.getElementById("retrievalLatencyBadge");
  const chunksList            = document.getElementById("chunksList");
  const comparisonBox         = document.getElementById("comparisonBox");
  const e5CompMeta            = document.getElementById("e5CompMeta");
  const e5CompChunks          = document.getElementById("e5CompChunks");
  const bgeCompMeta           = document.getElementById("bgeCompMeta");
  const bgeCompChunks         = document.getElementById("bgeCompChunks");

  // Feed sidebar
  const openFeedBtn       = document.getElementById("openFeedBtn");
  const closeFeedBtn      = document.getElementById("closeFeedBtn");
  const articleFeedSidebar = document.getElementById("articleFeedSidebar");
  const articleFeedOverlay = document.getElementById("articleFeedOverlay");
  const feedArticleList    = document.getElementById("feedArticleList");
  const feedSearchInput    = document.getElementById("feedSearchInput");
  const feedTabs           = document.querySelectorAll(".feed-tab");

  // Dark mode
  const darkModeBtn = document.getElementById("darkModeBtn");

  // ============================================================
  // ① DARK MODE
  // ============================================================
  const savedTheme = localStorage.getItem("vaani-theme") || "light";
  document.documentElement.setAttribute("data-theme", savedTheme);
  darkModeBtn.textContent = savedTheme === "dark" ? "☀️" : "🌙";

  darkModeBtn.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("vaani-theme", next);
    darkModeBtn.textContent = next === "dark" ? "☀️" : "🌙";
  });

  // ============================================================
  // ② MODEL SELECTION
  // ============================================================
  function selectModel(modelId) {
    activeModel = modelId;
    cardE5.classList.toggle("selected", modelId === "e5");
    cardBge.classList.toggle("selected", modelId === "bge_m3");
    fetch("/api/switch-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId })
    }).catch(console.error);
  }

  cardE5.addEventListener("click",  () => selectModel("e5"));
  cardBge.addEventListener("click", () => selectModel("bge_m3"));

  // ============================================================
  // ③ INPUT MODE TABS
  // ============================================================
  const tabUrlBtn          = document.getElementById("tabUrlBtn");
  const tabTextBtn         = document.getElementById("tabTextBtn");
  const urlInputContainer  = document.getElementById("urlInputContainer");
  const textInputContainer = document.getElementById("textInputContainer");
  const customArticleText  = document.getElementById("customArticleText");
  const loadCustomTextBtn  = document.getElementById("loadCustomTextBtn");

  if (tabUrlBtn && tabTextBtn) {
    tabUrlBtn.addEventListener("click", () => {
      tabUrlBtn.classList.add("active");
      tabTextBtn.classList.remove("active");
      urlInputContainer.classList.remove("hidden");
      textInputContainer.classList.add("hidden");
    });
    tabTextBtn.addEventListener("click", () => {
      tabTextBtn.classList.add("active");
      tabUrlBtn.classList.remove("active");
      textInputContainer.classList.remove("hidden");
      urlInputContainer.classList.add("hidden");
    });
  }

  // ============================================================
  // ④ NAMED ENTITY RECOGNITION (client-side heuristic)
  // ============================================================

  // Patterns for common English / Telugu entities
  const ENTITY_PATTERNS = [
    // Money / amounts
    { re: /₹[\d,]+(?:\s*(?:crore|lakh|thousand|కోట్లు|లక్షలు))?(?:\s*crore)?/gi,           type: "money" },
    { re: /\$[\d,]+(?:\s*(?:billion|million|thousand))?/gi,                                 type: "money" },
    // Dates (English)
    { re: /\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}/g, type: "date" },
    { re: /\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}/g, type: "date" },
    { re: /\b\d{4}-\d{2}-\d{2}\b/g, type: "date" },
    // Andhra / Telangana places
    { re: /\b(?:Visakhapatnam|Vijayawada|Hyderabad|Guntur|Tirupati|Kurnool|Kakinada|Nellore|Anantapur|Ongole|Eluru|Rajahmundry|Bapatla|Prakasam|Machilipatnam|Amaravati|Andhra\s+Pradesh|Telangana|Tamil\s+Nadu|Chennai|Pune|Mumbai|Delhi|New\s+Delhi|Bengaluru|Bangalore)\b/g, type: "place" },
    // Telugu place names
    { re: /\b(?:విశాఖపట్నం|హైదరాబాద్|విజయవాడ|గుంటూరు|తిరుపతి|కర్నూలు|కాకినాడ|నెల్లూరు|అమరావతి|ఒంగోలు|ఏలూరు|రాజమహేంద్రవరం|ఆంధ్రప్రదేశ్|తెలంగాణ)\b/g, type: "place" },
    // Known orgs / films / brands
    { re: /\b(?:BGE-M3|TCS|VIT-AP|HyperVault|ISRO|NASA|BJP|TDP|YSRCP|SLV\s+Cinemas|Google|Microsoft|Amazon|OpenAI|KGH|King\s+George\s+Hospital)\b/g, type: "org" },
    // Telugu person names and titles
    { re: /\b(?:Chiranjeevi|Ram\s+Charan|Sukumar|Nani|Keerthy\s+Suresh|Srikanth\s+Odela|Santhosh\s+Narayanan|Navi\s+Pillay|Narendra\s+Modi|Modi|Lokesh|Nara\s+Lokesh|Prem\s+Rakshith)\b/g, type: "person" },
    { re: /\b(?:ప్రణయ్|ప్రణయ్\s+తేజ|చిరంజీవి|రామ్\s+చరణ్|సుకుమార్|కీర్తి\s+సురేష్|నవీ\s+పిళ్లై|మోదీ|లోకేశ్)\b/g, type: "person" },
  ];

  function extractEntities(text) {
    const found = {}; // type -> Set of strings
    for (const { re, type } of ENTITY_PATTERNS) {
      const matches = text.match(re) || [];
      if (!found[type]) found[type] = new Set();
      matches.forEach(m => found[type].add(m.trim()));
    }
    return found;
  }

  function highlightEntities(text, entities) {
    // Build a replacement map: entity string -> html span
    let result = text;
    // Process in order: money, date, place, org, person (to avoid double-wrap)
    const typeOrder = ["money", "date", "org", "place", "person"];
    const allPairs = [];
    for (const type of typeOrder) {
      if (entities[type]) {
        entities[type].forEach(str => allPairs.push({ str, type }));
      }
    }
    // Sort longest first to avoid partial replacements
    allPairs.sort((a, b) => b.str.length - a.str.length);

    // Use a placeholder approach to avoid re-replacing inside tags
    const placeholders = {};
    let idx = 0;
    for (const { str, type } of allPairs) {
      if (!str) continue;
      const escaped = str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const key = `\x00${idx++}\x00`;
      const span = `<mark class="ent-${type}" title="${type}">${str}</mark>`;
      placeholders[key] = span;
      result = result.replace(new RegExp(escaped, 'g'), key);
    }
    // Restore
    for (const [key, span] of Object.entries(placeholders)) {
      result = result.split(key).join(span);
    }
    return result;
  }

  function renderEntityTags(entities) {
    const typeLabels = { person: "👤 People", place: "📍 Places", org: "🏛 Orgs", date: "📅 Dates", money: "💰 Amounts" };
    const hasAny = Object.values(entities).some(s => s.size > 0);
    if (!hasAny) { entityTagsRow.classList.add("hidden"); return; }

    entityTagsRow.innerHTML = '<span class="entity-tag-label">Entities:</span>';
    for (const [type, nameSet] of Object.entries(entities)) {
      if (!nameSet.size) continue;
      // Show up to 5 per type
      const items = [...nameSet].slice(0, 5);
      items.forEach(name => {
        const tag = document.createElement("span");
        tag.className = `entity-tag ${type}`;
        tag.title = typeLabels[type] || type;
        tag.textContent = name;
        entityTagsRow.appendChild(tag);
      });
    }
    entityTagsRow.classList.remove("hidden");
  }

  // ============================================================
  // ⑤ ARTICLE LOADING
  // ============================================================
  async function loadArticle(url, customText = null) {
    if (!url && !customText) return;
    if (url) articleUrlInput.value = url;
    loadArticleBtn.disabled = true;
    loadArticleBtn.textContent = "Loading...";
    if (loadCustomTextBtn) {
      loadCustomTextBtn.disabled = true;
      loadCustomTextBtn.textContent = "Indexing...";
    }

    try {
      const resp = await fetch("/api/load-article", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url || "", text: customText || "" })
      });

      if (!resp.ok) {
        let errMsg = `Status ${resp.status}`;
        try { const d = await resp.json(); if (d.error) errMsg = d.error; } catch(e) {}
        throw new Error(errMsg);
      }

      const data = await resp.json();
      currentArticle = data.article;

      // Populate metadata
      metaTitle.textContent    = currentArticle.title;
      metaCategory.textContent = currentArticle.category || "General News";
      metaLangBadge.textContent = currentArticle.language === "te" ? "TELUGU" : "ENGLISH";
      metaWords.textContent    = currentArticle.text.split(/\s+/).length;
      metaChunks.textContent   = data.chunk_count;
      metaNewsId.textContent   = currentArticle.news_id || "N/A";

      // Named entity extraction + rendering
      const entities = extractEntities(currentArticle.text);
      renderEntityTags(entities);

      // Article body with entity highlights
      const highlighted = highlightEntities(
        currentArticle.text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'),
        entities
      );
      articleBodyText.innerHTML = highlighted;

      articleMetaBox.classList.remove("hidden");
      resultsSection.classList.add("hidden");
      comparisonBox.classList.add("hidden");

      // Context-aware prompt chips
      if (data.suggested_prompts && data.suggested_prompts.length) {
        renderPromptChips(data.suggested_prompts);
      }

      // Mark active in feed sidebar
      document.querySelectorAll(".feed-article-item").forEach(el => {
        el.classList.toggle("active", el.dataset.newsId === (currentArticle.news_id || ""));
      });

      console.log(`✅ Article loaded: ${currentArticle.title} (${data.chunk_count} chunks)`);
    } catch (err) {
      alert(`Error loading article: ${err.message}`);
    } finally {
      loadArticleBtn.disabled = false;
      loadArticleBtn.textContent = "Load article";
      if (loadCustomTextBtn) {
        loadCustomTextBtn.disabled = false;
        loadCustomTextBtn.textContent = "Load & Index Article";
      }
    }
  }

  loadArticleBtn.addEventListener("click", () => loadArticle(articleUrlInput.value.trim()));

  if (loadCustomTextBtn) {
    loadCustomTextBtn.addEventListener("click", () => {
      const txt = customArticleText.value.trim();
      if (!txt) { alert("Please paste the article text first."); return; }
      loadArticle("", txt);
    });
  }

  document.querySelectorAll(".try-link").forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const targetUrl = link.getAttribute("data-url");
      if (targetUrl) { articleUrlInput.value = targetUrl; loadArticle(targetUrl); }
    });
  });

  toggleArticleBodyBtn.addEventListener("click", () => {
    const isHidden = articleBodyPreview.classList.toggle("hidden");
    toggleArticleBodyBtn.textContent = isHidden ? "View Text ▼" : "Hide Text ▲";
  });

  // ============================================================
  // ⑥ ARTICLE FEED SIDEBAR
  // ============================================================
  let feedLangFilter = "all";
  let feedSearchQuery = "";

  function renderFeed() {
    feedArticleList.innerHTML = "";
    const filtered = FEED_ARTICLES.filter(a => {
      if (feedLangFilter !== "all" && a.lang !== feedLangFilter) return false;
      if (feedSearchQuery) {
        const q = feedSearchQuery.toLowerCase();
        return a.title.toLowerCase().includes(q) || a.news_id.includes(q);
      }
      return true;
    });
    if (!filtered.length) {
      feedArticleList.innerHTML = `<p style="padding:20px;color:var(--text-muted);font-size:13px;">No articles found.</p>`;
      return;
    }
    filtered.forEach(a => {
      const item = document.createElement("div");
      item.className = "feed-article-item";
      item.dataset.newsId = a.news_id;
      item.dataset.url = a.url;
      if (currentArticle && currentArticle.news_id === a.news_id) item.classList.add("active");
      item.innerHTML = `
        <div class="feed-item-title">${a.title}</div>
        <div class="feed-item-meta">
          <span class="feed-lang-badge ${a.lang}">${a.lang === "te" ? "తెలుగు" : "English"}</span>
          <span>ID: ${a.news_id}</span>
        </div>
      `;
      item.addEventListener("click", () => {
        loadArticle(a.url);
        closeFeed();
      });
      feedArticleList.appendChild(item);
    });
  }

  function openFeed() {
    articleFeedSidebar.classList.add("open");
    articleFeedOverlay.classList.remove("hidden");
    articleFeedOverlay.setAttribute("aria-hidden", "false");
    renderFeed();
  }
  function closeFeed() {
    articleFeedSidebar.classList.remove("open");
    articleFeedOverlay.classList.add("hidden");
    articleFeedOverlay.setAttribute("aria-hidden", "true");
  }

  openFeedBtn.addEventListener("click", openFeed);
  closeFeedBtn.addEventListener("click", closeFeed);
  articleFeedOverlay.addEventListener("click", closeFeed);

  feedTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      feedTabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      feedLangFilter = tab.dataset.lang;
      renderFeed();
    });
  });

  feedSearchInput.addEventListener("input", () => {
    feedSearchQuery = feedSearchInput.value.trim();
    renderFeed();
  });

  // ============================================================
  // ⑦ DYNAMIC PROMPT CHIPS
  // ============================================================
  const promptChipsContainer = document.getElementById("promptChipsContainer") ||
                               document.querySelector(".prompt-chips");

  function renderPromptChips(prompts) {
    if (!promptChipsContainer || !prompts || !prompts.length) return;
    promptChipsContainer.innerHTML = '<span class="chip-label">Quick Prompts:</span>';
    prompts.forEach(p => {
      const chip = document.createElement("button");
      chip.className = "chip";
      chip.setAttribute("data-q", p.query);
      chip.textContent = p.label || p.query;
      chip.addEventListener("click", () => {
        questionInput.value = p.query;
        const isSum = p.query.toLowerCase().includes("summar") || p.query.includes("సారాంశం");
        executeAskQuestion(isSum);
      });
      promptChipsContainer.appendChild(chip);
    });
  }

  // Bind initial static chips
  promptChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-q");
      if (q) {
        questionInput.value = q;
        const isSum = q.toLowerCase().includes("summar") || q.includes("సారాంశం");
        executeAskQuestion(isSum);
      }
    });
  });

  // ============================================================
  // ⑧ ASK / QUERY EXECUTION
  // ============================================================
  async function executeAskQuestion(isSummary = false) {
    const query = questionInput.value.trim();
    if (!query && !isSummary) { alert("Please enter a question or select a prompt."); return; }

    const effectiveQuery = isSummary ? "Summarize this article" : query;
    currentQuery = effectiveQuery;

    loadingText.textContent = `Retrieving with ${activeModel === "e5" ? "multilingual-E5-base" : "BGE-M3"}...`;
    loadingIndicator.classList.remove("hidden");
    resultsSection.classList.add("hidden");
    comparisonBox.classList.add("hidden");

    try {
      const resp = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: effectiveQuery,
          model_id: activeModel,
          url: articleUrlInput.value.trim(),
          is_summary: isSummary
        })
      });

      if (!resp.ok) {
        let errMsg = `Status ${resp.status}`;
        try { const d = await resp.json(); if (d.error) errMsg = d.error; } catch(e) {}
        throw new Error(errMsg);
      }

      const data = await resp.json();
      currentAnswer = data.answer;
      currentAudioUrl = data.audio_url || null;

      answerText.textContent = data.answer;
      retrievalModelBadge.textContent  = `Model: ${data.model_name}`;
      retrievalDimBadge.textContent    = `Dim: ${data.dim}d`;
      retrievalLatencyBadge.textContent = `Retrieval: ${data.latency_ms} ms`;

      // Save to history
      qaHistory.push({ q: effectiveQuery, a: data.answer, model: data.model_name, latency: data.latency_ms });

      // Chunks
      chunksList.innerHTML = "";
      (data.chunks || []).forEach(chunk => {
        const card = document.createElement("div");
        card.className = "chunk-card";
        card.innerHTML = `
          <div class="chunk-header">
            <span>Rank #${chunk.rank} · ${chunk.chunk_id}</span>
            <span class="score-badge">${chunk.score}</span>
          </div>
          <div class="chunk-text">${chunk.text}</div>
        `;
        chunksList.appendChild(card);
      });

      resultsSection.classList.remove("hidden");
      comparisonBox.classList.add("hidden");
      resultsSection.scrollIntoView({ behavior: "smooth", block: "nearest" });

      // Auto-speak if TTS was already on
      if (isSpeaking) { stopTTS(); }

    } catch (err) {
      alert(`Query failed: ${err.message}`);
    } finally {
      loadingIndicator.classList.add("hidden");
    }
  }

  askBtn.addEventListener("click", () => executeAskQuestion(false));
  summarizeBtn.addEventListener("click", () => executeAskQuestion(true));
  questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); executeAskQuestion(false); }
  });

  // ============================================================
  // ⑨ IMPROVED TTS WITH LANGUAGE AUTO-DETECTION
  // ============================================================
  let availableVoices = [];

  function loadVoices() {
    availableVoices = window.speechSynthesis.getVoices();
    ttsVoiceSelect.innerHTML = '<option value="auto">🌐 Auto-detect language</option>';

    // Group voices by language
    const teVoices = availableVoices.filter(v => v.lang.startsWith("te"));
    const enVoices = availableVoices.filter(v => v.lang.startsWith("en"));
    const otherVoices = availableVoices.filter(v => !v.lang.startsWith("te") && !v.lang.startsWith("en"));

    if (teVoices.length) {
      const og = document.createElement("optgroup"); og.label = "తెలుగు Voices";
      teVoices.forEach(v => {
        const o = document.createElement("option");
        o.value = v.name; o.textContent = `${v.name} (${v.lang})`;
        og.appendChild(o);
      });
      ttsVoiceSelect.appendChild(og);
    }
    if (enVoices.length) {
      const og = document.createElement("optgroup"); og.label = "English Voices";
      enVoices.slice(0, 8).forEach(v => {
        const o = document.createElement("option");
        o.value = v.name; o.textContent = `${v.name} (${v.lang})`;
        og.appendChild(o);
      });
      ttsVoiceSelect.appendChild(og);
    }
    if (otherVoices.length) {
      const og = document.createElement("optgroup"); og.label = "Other";
      otherVoices.slice(0, 4).forEach(v => {
        const o = document.createElement("option");
        o.value = v.name; o.textContent = `${v.name} (${v.lang})`;
        og.appendChild(o);
      });
      ttsVoiceSelect.appendChild(og);
    }
  }

  if ("speechSynthesis" in window) {
    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;
  }

  function detectAnswerLang(text) {
    const teluguChars = (text.match(/[\u0C00-\u0C7F]/g) || []).length;
    return teluguChars > 5 ? "te" : "en";
  }

  function getBestVoice(lang) {
    const selectedName = ttsVoiceSelect.value;
    if (selectedName !== "auto") {
      return availableVoices.find(v => v.name === selectedName) || null;
    }
    // Auto: prefer native Telugu voice, then any Telugu, then English
    if (lang === "te") {
      return availableVoices.find(v => v.lang === "te-IN") ||
             availableVoices.find(v => v.lang.startsWith("te")) ||
             availableVoices.find(v => v.lang === "en-IN") ||
             availableVoices.find(v => v.lang.startsWith("en")) || null;
    }
    return availableVoices.find(v => v.lang === "en-IN") ||
           availableVoices.find(v => v.lang === "en-US") ||
           availableVoices.find(v => v.lang.startsWith("en")) || null;
  }

  function stopTTS() {
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    isSpeaking = false;
    playAudioBtn.classList.remove("hidden");
    stopAudioBtn.classList.add("hidden");
    playAudioBtn.textContent = "🔊 Listen";
  }

  playAudioBtn.addEventListener("click", () => {
    if (!currentAnswer) return;

    if (!("speechSynthesis" in window)) {
      if (currentAudioUrl) { ttsAudio.src = currentAudioUrl; ttsAudio.play(); }
      else alert("Speech synthesis is not supported in this browser.");
      return;
    }

    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(currentAnswer);
    const lang = detectAnswerLang(currentAnswer);
    const voice = getBestVoice(lang);

    utt.lang = lang === "te" ? "te-IN" : "en-IN";
    utt.rate = 0.95;
    utt.pitch = 1.0;
    if (voice) utt.voice = voice;

    isSpeaking = true;
    playAudioBtn.classList.add("hidden");
    stopAudioBtn.classList.remove("hidden");

    utt.onend  = () => stopTTS();
    utt.onerror = () => stopTTS();

    window.speechSynthesis.speak(utt);
    ttsUtterance = utt;
  });

  stopAudioBtn.addEventListener("click", stopTTS);

  // ============================================================
  // ⑩ SHARE / EXPORT
  // ============================================================

  // Copy answer to clipboard
  copyAnswerBtn.addEventListener("click", async () => {
    if (!currentAnswer) { alert("No answer to copy yet."); return; }
    try {
      await navigator.clipboard.writeText(currentAnswer);
      copyAnswerBtn.textContent = "✅ Copied!";
      setTimeout(() => { copyAnswerBtn.textContent = "📋 Copy"; }, 2000);
    } catch {
      // Fallback
      const ta = document.createElement("textarea");
      ta.value = currentAnswer;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      copyAnswerBtn.textContent = "✅ Copied!";
      setTimeout(() => { copyAnswerBtn.textContent = "📋 Copy"; }, 2000);
    }
  });

  // Export Q&A session as formatted PDF (using browser print)
  exportPdfBtn.addEventListener("click", () => {
    if (!currentAnswer) { alert("No answer to export yet."); return; }

    const articleTitle = currentArticle ? currentArticle.title : "Unknown Article";
    const articleId    = currentArticle ? (currentArticle.news_id || "custom") : "N/A";
    const lang         = currentArticle ? (currentArticle.language === "te" ? "Telugu" : "English") : "Unknown";
    const now          = new Date().toLocaleString("en-IN", { timeZone: "Asia/Kolkata" });

    // Build full history HTML
    const historyHTML = qaHistory.map((entry, i) => `
      <div style="margin-bottom:20px;padding:14px;border:1px solid #e2e8f0;border-radius:8px;border-left:4px solid #1c4b75;">
        <div style="font-size:12px;color:#64748b;margin-bottom:6px;">Q${i+1} · ${entry.model} · ${entry.latency} ms</div>
        <div style="font-weight:600;margin-bottom:8px;color:#1e293b;">${entry.q}</div>
        <div style="font-size:14px;line-height:1.65;color:#334155;white-space:pre-wrap;">${entry.a}</div>
      </div>
    `).join("");

    const printWin = window.open("", "_blank", "width=800,height=700");
    printWin.document.write(`
      <!DOCTYPE html><html><head><meta charset="UTF-8">
      <title>Vaani Export – ${articleTitle}</title>
      <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; padding: 36px; color: #1e293b; max-width: 760px; margin: 0 auto; }
        h1 { font-size: 20px; color: #1c4b75; margin-bottom: 4px; }
        .meta { font-size: 12px; color: #64748b; margin-bottom: 24px; padding-bottom: 14px; border-bottom: 1px solid #e2e8f0; }
        .footer { margin-top: 32px; font-size: 11px; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 12px; }
        @media print { button { display: none !important; } }
      </style></head><body>
      <h1>📰 Bilingual News Assistant – Session Export</h1>
      <div class="meta">
        <strong>Article:</strong> ${articleTitle} &nbsp;|&nbsp;
        <strong>ID:</strong> ${articleId} &nbsp;|&nbsp;
        <strong>Language:</strong> ${lang} &nbsp;|&nbsp;
        <strong>Exported:</strong> ${now}
      </div>
      <h2 style="font-size:15px;margin-bottom:14px;">Q&amp;A History (${qaHistory.length} exchange${qaHistory.length !== 1 ? "s" : ""})</h2>
      ${historyHTML || `<div style="padding:14px;border:1px solid #e2e8f0;border-radius:8px;border-left:4px solid #1c4b75;white-space:pre-wrap;">${currentAnswer}</div>`}
      <div class="footer">Team Vaani · VIT-AP University · Bilingual News RAG Benchmark · bilingualeassistance.vercel.app</div>
      <br><button onclick="window.print()" style="padding:10px 20px;background:#1c4b75;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:14px;">🖨️ Print / Save as PDF</button>
      </body></html>
    `);
    printWin.document.close();
    printWin.focus();
  });

  // ============================================================
  // ⑪ COMPARE E5 vs BGE-M3
  // ============================================================
  compareBtn.addEventListener("click", async () => {
    const query = questionInput.value.trim() || "What is this article about?";
    loadingText.textContent = "Benchmarking Multilingual-E5 vs BGE-M3 side-by-side...";
    loadingIndicator.classList.remove("hidden");

    try {
      const resp = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, url: articleUrlInput.value.trim() })
      });

      if (!resp.ok) {
        let errMsg = `Status ${resp.status}`;
        try { const d = await resp.json(); if (d.error) errMsg = d.error; } catch(e) {}
        throw new Error(errMsg);
      }

      const data = await resp.json();

      e5CompMeta.textContent  = `Retrieval Latency: ${data.e5.latency_ms} ms | Dim: 768d`;
      e5CompChunks.innerHTML  = data.e5.results.map(r => `
        <div class="chunk-card" style="margin-top:8px;">
          <div class="chunk-header"><span>Rank #${r.rank} (${r.chunk_id})</span><span class="score-badge">${r.score}</span></div>
          <div class="chunk-text" style="font-size:13px;">${r.text.substring(0, 180)}...</div>
        </div>`).join("");

      bgeCompMeta.textContent  = `Retrieval Latency: ${data.bge_m3.latency_ms} ms | Dim: 1024d`;
      bgeCompChunks.innerHTML  = data.bge_m3.results.map(r => `
        <div class="chunk-card" style="margin-top:8px;">
          <div class="chunk-header"><span>Rank #${r.rank} (${r.chunk_id})</span><span class="score-badge">${r.score}</span></div>
          <div class="chunk-text" style="font-size:13px;">${r.text.substring(0, 180)}...</div>
        </div>`).join("");

      comparisonBox.classList.remove("hidden");
      resultsSection.classList.remove("hidden");
      comparisonBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) {
      alert(`Comparison failed: ${err.message}`);
    } finally {
      loadingIndicator.classList.add("hidden");
    }
  });

  // ============================================================
  // ⑫ MICROPHONE / SPEECH-TO-TEXT
  // ============================================================
  micBtn.addEventListener("click", () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Speech recognition is not supported in this browser. Please type your question.");
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = "te-IN";
    recognition.continuous = false;
    micBtn.textContent = "🔴 Listening...";
    recognition.onresult = (event) => {
      questionInput.value = event.results[0][0].transcript;
      micBtn.textContent = "🎤";
      executeAskQuestion(false);
    };
    recognition.onerror = () => { micBtn.textContent = "🎤"; };
    recognition.onend   = () => { micBtn.textContent = "🎤"; };
    recognition.start();
  });

  // ============================================================
  // ⑬ GENERATION TOGGLE
  // ============================================================
  const genToggleBtn = document.getElementById("genToggleBtn");
  if (genToggleBtn) {
    genToggleBtn.addEventListener("click", () => {
      const dot = genToggleBtn.querySelector(".status-dot");
      const isOn = dot.style.backgroundColor !== "rgb(248, 113, 113)";
      dot.style.backgroundColor = isOn ? "#f87171" : "#4ade80";
      genToggleBtn.childNodes[1].textContent = isOn ? " Generation disabled" : " Generation enabled";
    });
  }

  // ============================================================
  // INIT: Load default article on startup
  // ============================================================
  loadArticle("https://telugupeople.com/news/news.asp?newsID=108267");
});
