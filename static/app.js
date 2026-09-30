const API = window.location.protocol === "file:" ? "http://localhost:8080" : "";

function randomSessionId() {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("");
}
let evaluationSession;
try {
  evaluationSession = sessionStorage.getItem("aic.evaluation-session") || randomSessionId();
  sessionStorage.setItem("aic.evaluation-session", evaluationSession);
} catch (_) { evaluationSession = randomSessionId(); }
let activeSearchReceipt = null;
const fileSearchReceipts = new Map();
function newEvaluationSession() {
  evaluationSession = randomSessionId();
  try { sessionStorage.setItem("aic.evaluation-session", evaluationSession); } catch (_) {}
  activeSearchReceipt = null;
  fileSearchReceipts.clear();
  clearShortlist();
  document.getElementById("evaluation-status").textContent = "New competition session started.";
}
async function trackedSearch(url, options = {}) {
  const session = evaluationSession;
  const response = await fetch(url, {...options, headers: {...options.headers, "X-Search-Session": session}});
  const receipt = response.headers.get("X-Search-Id");
  if (receipt && session === evaluationSession) {
    activeSearchReceipt = receipt;
    const data = await response.clone().json().catch(() => ({}));
    const frames = [...(data.results || []), ...(data.videos || [])];
    frames.forEach(item => {
      if (item.file) fileSearchReceipts.set(item.file, receipt);
      (item.frames || []).forEach(frame => fileSearchReceipts.set(frame.file, receipt));
    });
    while (fileSearchReceipts.size > 20000) fileSearchReceipts.delete(fileSearchReceipts.keys().next().value);
  }
  return response;
}
function recordEvaluation(event, answer = "", question = "", changedFiles = []) {
  const ids = new Set(shortlist.map(f => fileSearchReceipts.get(f.file)).filter(Boolean));
  (Array.isArray(changedFiles) ? changedFiles : [changedFiles]).forEach(file => {
    if (fileSearchReceipts.has(file)) ids.add(fileSearchReceipts.get(file));
  });
  if (!ids.size && activeSearchReceipt) ids.add(activeSearchReceipt);
  ids.forEach(search_id => {
    fetch(`${API}/evaluation/event`, {method: "POST", keepalive: true,
      headers: {"Content-Type": "application/json", "X-Search-Session": evaluationSession},
      body: JSON.stringify({event, search_id, task_mode: taskMode, frames: shortlist,
                            answer, question})}).catch(() => {});
  });
}
fetch(`${API}/search/features`).then(r => r.json()).then(features => {
  document.getElementById("inp-rerank").disabled = !features.vlm_available;
  document.getElementById("inp-rerank").title = features.vlm_available
    ? "Refine the first matches with the local vision model" : "A local vision model has not been configured";
}).catch(() => {});
fetch(`${API}/evaluation/status`).then(r => r.json()).then(status => {
  if (!status.enabled) document.getElementById("evaluation-status").textContent = "Session logging is disabled.";
}).catch(() => {});


// ── state ─────────────────────────────────────────────────────────────────────
let activeCategories = new Set();
let activeObjects    = new Set();
let lastResults      = [];
let lastResultContext = {
  total: 0,
  autoFiltered: [],
  operator: "semantic",
  terms: [],
  deduplicated: 0,
};
let groupMode        = false;
let searchReady      = false;
let renderGeneration = 0;
let activeResultsScrollHandler = null;
const QUERY_OPERATOR_STORAGE_KEY = "frame-search.query-operator.v1";
const OCR_OPERATOR_STORAGE_KEY = "frame-search.ocr-operator.v1";
const VIDEO_FILTER_MODE_STORAGE_KEY = "frame-search.video-filter-mode.v1";
let activeQueryOperator = "and";
let activeOcrOperator = "and";
let activeAsrOperator = "and";
const ASR_OPERATOR_STORAGE_KEY = "frame-search.asr-operator.v1";
let activeVideoMode = "include";
try {
  activeQueryOperator =
    localStorage.getItem(QUERY_OPERATOR_STORAGE_KEY) === "or" ? "or" : "and";
  activeOcrOperator =
    localStorage.getItem(OCR_OPERATOR_STORAGE_KEY) === "or" ? "or" : "and";
  activeAsrOperator =
    localStorage.getItem(ASR_OPERATOR_STORAGE_KEY) === "or" ? "or" : "and";
  activeVideoMode =
    localStorage.getItem(VIDEO_FILTER_MODE_STORAGE_KEY) === "exclude" ? "exclude" : "include";
} catch {}

// ── box colors ────────────────────────────────────────────────────────────────
const CLASS_COLORS = {};
const PALETTE = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c","#e67e22","#e91e63","#00bcd4","#8bc34a","#ff5722","#607d8b","#795548","#ffc107","#673ab7"];
function colorFor(cls) {
  if (!CLASS_COLORS[cls]) CLASS_COLORS[cls] = PALETTE[Object.keys(CLASS_COLORS).length % PALETTE.length];
  return CLASS_COLORS[cls];
}

// ── boot ──────────────────────────────────────────────────────────────────────
let statusRetryTimer = null;
async function fetchStatus() {
  try {
    const response = await fetch(`${API}/status`, { cache: "no-store" });
    if (!response.ok) throw new Error(`status ${response.status}`);
    const d = await response.json();
    document.getElementById("status-pill").textContent = `${d.indexed} frames indexed`;
    if (d.indexed > 0) buildSidebar(d.categories || {}, d.objects || {}, d.obj_to_cat || {}, d.all_categories || [], d.colors || {});
    if (statusRetryTimer) {
      clearInterval(statusRetryTimer);
      statusRetryTimer = null;
    }
  } catch {
    document.getElementById("status-pill").textContent = "⚠ backend offline";
    if (!statusRetryTimer) {
      statusRetryTimer = setInterval(fetchStatus, 2000);
    }
  }
}
async function waitForSearchReady() {
  const button = document.getElementById("btn-search");
  while (!searchReady) {
    try {
      const response = await fetch(`${API}/ready`, { cache: "no-store" });
      const state = await response.json();
      if (response.ok && state.ready) {
        searchReady = true;
        button.disabled = false;
        button.textContent = "Search";
        document.getElementById("status-pill").textContent =
          `${state.frames} frames ready`;
        await fetchStatus();
        return;
      }
      button.textContent =
        state.status === "failed" ? "Search unavailable" : "Loading search…";
    } catch {
      button.textContent = "Waiting for backend…";
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}
waitForSearchReady();

// ── color config ──────────────────────────────────────────────────────────────
const COLOR_HEX = {
  Red:"#e74c3c", Orange:"#e67e22", Yellow:"#f1c40f", Green:"#27ae60",
  Cyan:"#00bcd4", Blue:"#2980b9", Purple:"#8e44ad", Pink:"#e91e63",
  White:"#ecf0f1", Gray:"#95a5a6", Black:"#2c3e50"
};
let activeColors = new Set();

// ── sidebar ───────────────────────────────────────────────────────────────────
function buildSidebar(catCounts, objCounts, objToCat, allCategories, colorCounts) {
  const catToObjs = {};
  Object.entries(objToCat).forEach(([obj, cat]) => {
    if (!catToObjs[cat]) catToObjs[cat] = [];
    catToObjs[cat].push(obj);
  });

  const list = document.getElementById("cat-list");

  // ── Categories (all shown, even if count 0) ──
  const catsHtml = (allCategories || Object.keys(catCounts)).map(cat => {
    const objs  = (catToObjs[cat] || []).sort();
    const count = catCounts[cat] || 0;
    return `
      <div class="category-group">
        <div class="category-header" onclick="toggleCatExpand('${cat}')">
          <input type="checkbox" id="cat-${cat}" ${activeCategories.has(cat)?"checked":""}
            onclick="event.stopPropagation();toggleCategory('${cat}',this.checked)">
          <label for="cat-${cat}" onclick="event.stopPropagation()">${cat}</label>
          <span class="cat-count ${count?'':'zero'}" id="catcount-${cat}">${count}</span>
          <span class="category-arrow" id="arrow-${cat}">▶</span>
        </div>
        <div class="category-children" id="children-${cat}">
          ${objs.map(obj => `
            <div class="obj-chip">
              <input type="checkbox" id="obj-${obj}" ${activeObjects.has(obj)?"checked":""}
                onchange="toggleObject('${obj}',this.checked)">
              <label for="obj-${obj}">${obj}</label>
              <span class="obj-count" id="objcount-${obj}">${objCounts[obj]||0}</span>
            </div>`).join("")}
        </div>
      </div>`;
  }).join("");

  // ── Colors ──
  const knownColors = ["Red","Orange","Yellow","Green","Cyan","Blue","Purple","Pink","White","Gray","Black"];
  const colorsHtml = `
    <div class="sidebar-section" style="margin-top:12px;">
      <h2>Colors</h2>
      <div style="display:flex;flex-wrap:wrap;gap:5px;padding:4px 0;">
        ${knownColors.map(color => {
          const hex   = COLOR_HEX[color] || "#999";
          const count = colorCounts[color] || 0;
          const active = activeColors.has(color);
          return `
            <button type="button" onclick="toggleColor('${color}')" id="colorbtn-${color}"
              aria-label="${color}" aria-pressed="${active}"
              title="${color} (${count})"
              style="width:28px;height:28px;min-height:28px;padding:0;border-radius:50%;background:${hex};
                     cursor:pointer;border:3px solid ${active?'#333':'transparent'};
                     opacity:${count>0?1:0.3};position:relative;transition:border .15s;"
            ></button>`;
        }).join("")}
      </div>
      <div id="active-colors" style="font-size:10px;color:#888;margin-top:2px;min-height:14px;"></div>
    </div>`;

  list.innerHTML = catsHtml + colorsHtml;
}

function updateSidebarCounts(results) {
  const catCount = {}, objCount = {};
  results.forEach(r => {
    r.categories.forEach(c => { catCount[c] = (catCount[c]||0)+1; });
    r.detections.forEach(d => { objCount[d.class] = (objCount[d.class]||0)+1; });
  });
  document.querySelectorAll(".cat-count").forEach(el => {
    const cat = el.id.replace("catcount-", "");
    const n = catCount[cat] ?? 0;
    el.textContent = n;
    el.classList.toggle("zero", n === 0);
  });
  document.querySelectorAll(".obj-count").forEach(el => {
    const obj = el.id.replace("objcount-", "");
    el.textContent = objCount[obj] ?? 0;
  });
}

function toggleCatExpand(cat) {
  const open = document.getElementById(`children-${cat}`).classList.toggle("open");
  document.getElementById(`arrow-${cat}`).textContent = open ? "▼" : "▶";
}
function toggleCategory(cat, checked) {
  checked ? activeCategories.add(cat) : activeCategories.delete(cat);
  doSearch();
}
function toggleObject(obj, checked) {
  checked ? activeObjects.add(obj) : activeObjects.delete(obj);
  doSearch();
}
function toggleColor(color) {
  if (activeColors.has(color)) activeColors.delete(color);
  else activeColors.add(color);
  // update border on color dot
  const btn = document.getElementById(`colorbtn-${color}`);
  if (btn) {
    btn.style.border = `3px solid ${activeColors.has(color) ? '#333' : 'transparent'}`;
    btn.setAttribute('aria-pressed', String(activeColors.has(color)));
  }
  // update label
  const label = document.getElementById("active-colors");
  if (label) label.textContent = activeColors.size ? [...activeColors].join(", ") : "";
  doSearch();
}

// ── conf slider ───────────────────────────────────────────────────────────────
const confSlider = document.getElementById("conf-slider");
const confVal    = document.getElementById("conf-val");
confSlider.oninput = () => { confVal.textContent = confSlider.value + "%"; };

// ── search ────────────────────────────────────────────────────────────────────
function setQueryOperator(operator, runSearch = true) {
  const next = operator === "or" ? "or" : "and";
  const changed = activeQueryOperator !== next;
  activeQueryOperator = next;
  const input = updateOperatorControl(next, QUERY_OPERATOR_STORAGE_KEY, ".query-op", "inp-query", "person, red shirt, motorcycle");
  if (runSearch && changed && searchReady && input.value.includes(",")) doSearch();
}

setQueryOperator(activeQueryOperator, false);

function setOcrOperator(operator, runSearch = true) {
  const next = operator === "or" ? "or" : "and";
  const changed = activeOcrOperator !== next;
  activeOcrOperator = next;
  const input = updateOperatorControl(next, OCR_OPERATOR_STORAGE_KEY, ".ocr-op", "inp-ocr", "exit, airport");
  if (runSearch && changed && searchReady && input.value.includes(",")) doSearch();
}

setOcrOperator(activeOcrOperator, false);

function setAsrOperator(operator, runSearch = true) {
  const next = operator === "or" ? "or" : "and";
  const changed = activeAsrOperator !== next;
  activeAsrOperator = next;
  const input = updateOperatorControl(next, ASR_OPERATOR_STORAGE_KEY, ".asr-op", "inp-asr-phrase", "hello, thank you");
  if (runSearch && changed &&  input.value.includes(",")) doAsrSearch();
}

setAsrOperator(activeAsrOperator, false);

function setVideoMode(mode, runSearch = true) {
  const nextMode = mode === "exclude" ? "exclude" : "include";
  const changed = activeVideoMode !== nextMode;
  activeVideoMode = nextMode;

  try {
    localStorage.setItem(VIDEO_FILTER_MODE_STORAGE_KEY, activeVideoMode);
  } catch {}

  const button = document.getElementById("btn-video-mode");
  const excluding = activeVideoMode === "exclude";
  button.dataset.mode = activeVideoMode;
  button.textContent = excluding ? "⊘ Exclude" : "✓ Include";
  button.setAttribute("aria-pressed", String(excluding));
  button.setAttribute("aria-label", `Video filter: ${activeVideoMode} matching video`);
  button.title = excluding
    ? "Return frames from every video except this one"
    : "Only return frames from this video";

  const input = document.getElementById("inp-video");
  input.placeholder = excluding ? "exclude e.g. L20_V001" : "e.g. L20_V001";

  if (runSearch && changed && searchReady && input.value.trim()) {
    doSearch();
  }
}

function toggleVideoMode() {
  setVideoMode(activeVideoMode === "include" ? "exclude" : "include");
}

setVideoMode(activeVideoMode, false);

let frameRequestSerial = 0;
let frameRequestController = null;
async function doSearch(targetPage = 1, reuseRequest = false) {
  targetPage = Number.isInteger(targetPage) ? targetPage : 1;
  if (!searchReady) return;
  const requestSerial = ++frameRequestSerial;
  frameRequestController?.abort();
  frameRequestController = new AbortController();
  clearSearchError();
  const query    = document.getElementById("inp-query").value.trim();
  const ocrQuery = document.getElementById("inp-ocr").value.trim();
  const videoFilter = document.getElementById("inp-video").value.trim();

  const conf     = parseInt(confSlider.value) / 100;
  const cats     = [...activeCategories].join(",");
  const objs     = [...activeObjects].join(",");
  const cols     = [...activeColors].join(",");

  setLoading(true);
  const params = new URLSearchParams({
    page: targetPage,
    page_size: FRAME_PAGE_SIZE,
    min_conf: conf,
    query_operator: activeQueryOperator,
    ocr_operator: activeOcrOperator,
  });
  params.set("domain", document.getElementById("summary-category").value);
  if (query)       params.set("query", query);
  if (ocrQuery)    params.set("ocr_query", ocrQuery);
  if (cats)        params.set("categories", cats);
  if (objs)        params.set("objects", objs);
  if (cols)        params.set("colors", cols);
  if (videoFilter) {
    params.set("video", videoFilter);
    params.set("video_mode", activeVideoMode);
  }

  try {
    params.set("rerank", document.getElementById("inp-rerank").checked);
    params.set("timeline", document.getElementById("inp-timeline").checked);
    const source = document.getElementById('summary-mode').value;
    if (source !== 'visual') {
      if (!query && !reuseRequest) throw new Error('Enter a summary query');

      params.set('category', document.getElementById('summary-category').value);
      params.set('mode', source);
    }
    let endpoint = source !== 'visual' ? '/summary/search' : '/search';
    let requestParams = params;
    if (reuseRequest && serverFramePagination) {
      endpoint = serverFramePagination.endpoint;
      requestParams = new URLSearchParams(serverFramePagination.params);
      requestParams.set('page', targetPage);
    }
    const response = await trackedSearch(`${API}${endpoint}?${requestParams}`, {signal: frameRequestController.signal});
    const d = await readJsonResponse(response);
    if (requestSerial !== frameRequestSerial) return;

    const rerankMessages = {applied: "Relationships refined", timeout: "Refinement timed out; original ranking shown",
      busy: "Refinement busy; original ranking shown", error: "Refinement unavailable; original ranking shown",
      cooldown: "Refinement unavailable; original ranking shown", unconfigured: "Vision model unavailable"};
    document.getElementById("rerank-status").textContent = rerankMessages[d.reranking?.status] || "";
    lastResults = d.results;
    lastResultContext = {
      total: d.total,
      autoFiltered: d.auto_filtered || [],
      operator: d.query_operator,
      terms: d.query_terms || [],
      deduplicated: d.deduplicated || 0,
    };
    renderResults(
      d.results,
      lastResultContext.total,
      lastResultContext.autoFiltered,
      lastResultContext.operator,
      lastResultContext.terms,
      lastResultContext.deduplicated,
      {page: d.page || 1, total: d.total, endpoint, params: requestParams.toString(),
       partial: !!d.partial, warnings: d.warnings || []},
    );
    updateSidebarCounts(d.results);
  } catch(e) {
    if (requestSerial === frameRequestSerial && e.name !== "AbortError") showSearchError(e.message);
  }
  if (requestSerial === frameRequestSerial) setLoading(false);
}

document.getElementById("btn-search").onclick = doSearch;
["inp-query","inp-ocr","inp-video"].forEach(id =>
  document.getElementById(id).addEventListener("keydown", e => { if (e.key==="Enter") doSearch(); })
);

// ── group by video ────────────────────────────────────────────────────────────
function toggleGroupMode() {
  groupMode = !groupMode;
  document.getElementById("btn-group-toggle").classList.toggle("active", groupMode);
  document.getElementById("btn-group-toggle").setAttribute("aria-pressed", String(groupMode));
  if (lastResults.length) renderCurrentFramePage();
}

function renderGrouped(results) {
  const grid = document.getElementById("grid");
  resultImages.prune(grid);
  grid.innerHTML = "";
  const groups = {};
  results.forEach(item => {
    const name = fileToAnswer(item.file).video;
    if (!groups[name]) groups[name] = [];
    groups[name].push(item);
  });
  const sorted = Object.entries(groups)
    .map(([name, frames]) => ({ name, frames, bestScore: Math.max(...frames.map(f=>f.score)) }));
  const colors = ["#c0392b","#e67e22","#f1c40f","#bdc3c7"];
  sorted.forEach((group, i) => {
    const col  = colors[Math.min(i, colors.length-1)];
    const card = document.createElement("div");
    card.className = "video-group-card";
    card.innerHTML = `
      <div class="video-group-header" style="border-left:4px solid ${col}"
           onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display==='none'?'grid':'none'">
        <div class="video-group-rank" style="color:${col}">#${i+1}</div>
        <div class="video-group-name">📁 ${escapeHtml(group.name)}</div>
        <div class="video-group-score">${(group.bestScore*100).toFixed(1)}%</div>
        <div class="video-group-count">${group.frames.length} frame${group.frames.length>1?"s":""}</div>
      </div>
      <div class="video-group-frames" style="${i===0?"":"display:none"}">
        ${group.frames.map(item => `
          <div style="position:relative;cursor:pointer" data-group-frame="${escapeHtml(item.file)}">
            <img src="${API}/image/${encodeURIComponent(item.file)}"
              style="width:100%;aspect-ratio:16/9;object-fit:cover;display:block;" loading="lazy" alt="${escapeHtml(item.file)}">
            <div style="position:absolute;bottom:3px;right:3px;background:rgba(0,0,0,.6);
                        color:#fff;font-size:10px;padding:1px 5px;border-radius:2px;">
              ${(item.score*100).toFixed(1)}%
            </div>
          </div>`).join("")}
      </div>`;
    card.querySelectorAll("[data-group-frame]").forEach((element, index) => {
      const item = group.frames[index];
      element.onclick = () => openLightbox(item);
      makeKeyboardAction(element, () => openLightbox(item), `Open ${item.file}`);
    });
    card.querySelectorAll("img").forEach(setupResultImage);
    grid.appendChild(card);
  });
}

// ── render results ────────────────────────────────────────────────────────────
// ── answer shortlist (for competition export) ──────────────────────────────
let shortlist = [];   // array of { file, frame, video, frameIdx, mapped, label }
let taskMode  = "kis"; // "kis" | "vqa" | "temporal"
const MAX_SHORTLIST = 5;
const MEMORY_STORAGE_KEY = "frame-search.saved-queries.v1";
let savedQueries = loadSavedQueries();

function setTaskMode(mode) {
  taskMode = mode;
  // VQA only ever holds 1 frame — trim if switching in with more selected
  if (taskMode === "vqa" && shortlist.length > 1) {
    shortlist = shortlist.slice(0, 1);
  }
  renderShortlistTray();
  refreshCardSelection();
}

function fileToAnswer(file) {
  // "L20/L20_V001/frame_0101.webp" → { frame: "101", video: "L20_V001" }
  const parts = file.replace(/\\/g, "/").split("/");
  const name  = parts[parts.length - 1];
  // video id is the deepest folder matching pattern like L20_V001
  let video = "UNKNOWN";
  for (let i = parts.length - 2; i >= 0; i--) {
    if (/^L\d+_V\d+$/i.test(parts[i])) { video = parts[i]; break; }
  }
  if (video === "UNKNOWN" && parts.length > 1) video = parts[parts.length - 2]; // fallback: immediate parent folder
  const m     = name.match(/(\d+)(?=\.\w+$)/);   // last number before extension
  const frame = m ? String(parseInt(m[1], 10)) : name;  // strip leading zeros
  return { frame, video, label: `${video},${frame}` };
}

async function toggleShortlist(file) {
  const idx = shortlist.findIndex(s => s.file === file);
  if (idx >= 0) {
    shortlist.splice(idx, 1);
    recordEvaluation("shortlist", "", "", file);
    renderShortlistTray();
    refreshCardSelection();
    return;
  }

  const base = fileToAnswer(file);

  // Temporal mode: all frames must come from the same video
  if (taskMode === "temporal" && shortlist.length > 0 && shortlist[0].video !== base.video) {
    alert(`Temporal Alignment requires all frames from the same video.\nCurrent video: ${shortlist[0].video}`);
    return;
  }

  // VQA mode: only one frame allowed — replace instead of accumulate
  if (taskMode === "vqa" && shortlist.length >= 1) {
    shortlist = [];
  } else if (taskMode !== "vqa" && taskMode !== "temporal" && shortlist.length >= MAX_SHORTLIST) {
    alert(`Max ${MAX_SHORTLIST} answers — remove one first.`);
    return;
  }

  const entry = { file, ...base, frameIdx: null, mapped: null, label: `${base.video},resolving…` };
  shortlist.push(entry);
  renderShortlistTray();
  refreshCardSelection();

  try {
    const d = await fetch(`${API}/frame-idx?video=${encodeURIComponent(base.video)}&n=${base.frame}`).then(r => r.json());
    entry.frameIdx = d.frame_idx;
    entry.mapped   = d.mapped;
    entry.label    = `${base.video},${d.frame_idx}`;
  } catch (e) {
    entry.frameIdx = base.frame;
    entry.mapped   = false;
    entry.label    = `${base.video},${base.frame}`;
  }
  renderShortlistTray();
  recordEvaluation("shortlist");
}

function refreshCardSelection() {
  document.querySelectorAll(`[data-file]`).forEach(el => {
    el.classList.toggle("selected", shortlist.some(s => s.file === el.dataset.file));
  });
}

function isShortlisted(file) {
  return shortlist.some(s => s.file === file);
}

function renderShortlistTray() {
  const previousName = document.getElementById("inp-question-num")?.value || "";
  const previousVqaAnswer = document.getElementById("inp-vqa-answer")?.value || "";
  let tray = document.getElementById("shortlist-tray");
  if (!tray) {
    tray = document.createElement("div");
    tray.id = "shortlist-tray";
    document.body.appendChild(tray);
  }
  if (!shortlist.length) {
    tray.innerHTML = "";
    tray.style.display = "none";
    return;
  }
  tray.style.display = "flex";

  const modeLabel = taskMode === "kis" ? "Textual KIS" : taskMode === "vqa" ? "VQA" : "Temporal Alignment";
  const capLabel  = taskMode === "vqa" ? "1 max" : taskMode === "temporal" ? "same video" : `${MAX_SHORTLIST} max`;

  const chips = shortlist.map((s,i) => {
    const bg     = s.frameIdx===null ? "#f0f0f0" : s.mapped ? "#fff3cd" : "#fdecea";
    const border = s.frameIdx===null ? "#ddd"    : s.mapped ? "#ffe69c" : "#f5b7b1";
    const numCol = s.frameIdx===null ? "#999"    : s.mapped ? "#856404" : "#c0392b";
    return `
      <div style="display:flex;align-items:center;gap:4px;background:${bg};
                  border:1px solid ${border};border-radius:4px;padding:3px 8px;font-size:11px;">
        <span style="font-weight:700;color:${numCol}">${i+1}.</span>
        <span style="font-family:monospace">${escapeHtml(s.label)}</span>
        ${s.frameIdx!==null && !s.mapped ? `<span title="No mapping found in CSV — using raw frame number as fallback" style="color:#c0392b;font-size:10px">⚠</span>` : ""}
        <button type="button" data-remove-file="${escapeHtml(s.file)}" aria-label="Remove ${escapeHtml(s.label)}"
                style="color:#c0392b;padding:2px 5px;background:transparent">✕</button>
      </div>`;
  }).join("");

  const vqaInput = taskMode === "vqa" ? `
    <input id="inp-vqa-answer" type="text" placeholder="answer e.g. 5"
      style="width:90px;padding:4px 7px;border:1px solid #4a90e2;border-radius:4px;font-size:11px;margin-left:6px;">` : "";

  tray.innerHTML = `
    <div style="font-weight:600;font-size:12px;color:#333;margin-right:4px;">
      ${modeLabel} (${shortlist.length}, ${capLabel}):
    </div>
    ${chips}
    ${vqaInput}
    <input id="inp-question-num" type="text" placeholder="e.g. query-1-kis"
      style="width:110px;padding:4px 7px;border:1px solid #ccc;border-radius:4px;font-size:11px;margin-left:6px;">
    <button onclick="exportAnswer()" style="background:#27ae60;color:#fff;padding:5px 12px;font-size:12px;border-radius:4px;border:none;cursor:pointer;">
      ⬇ Export .csv
    </button>
    <button onclick="switchTab('dres')" style="background:#245a91;color:#fff;padding:5px 12px;font-size:12px;border-radius:4px;border:none;cursor:pointer;">Nộp DRES</button>
    <button onclick="clearShortlist()" style="background:#eee;color:#444;padding:5px 10px;font-size:12px;border-radius:4px;border:none;cursor:pointer;">
      Clear
    </button>`;

  document.getElementById("inp-question-num").value = previousName;
  tray.querySelectorAll('[data-remove-file]').forEach(button => {
    button.onclick = () => toggleShortlist(button.dataset.removeFile);
  });
  const vqaAnswer = document.getElementById("inp-vqa-answer");
  if (vqaAnswer) vqaAnswer.value = previousVqaAnswer;
}

function clearShortlist() {
  const previousFiles = shortlist.map(f => f.file);
  shortlist = [];
  recordEvaluation("shortlist", "", "", previousFiles);
  renderShortlistTray();
  refreshCardSelection();
}

let queryCounter = 1;

function loadSavedQueries() {
  try {
    const parsed = JSON.parse(localStorage.getItem(MEMORY_STORAGE_KEY) || "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed.map((entry, index) => ({
      id: String(entry.id || `saved-${index}-${entry.savedAt || Date.now()}`),
      name: String(entry.name || `Saved query ${index + 1}`),
      taskMode: ["kis", "vqa", "temporal"].includes(entry.taskMode) ? entry.taskMode : "kis",
      sourceQuery: String(entry.sourceQuery || ""),
      savedAt: entry.savedAt || new Date().toISOString(),
      frames: Array.isArray(entry.frames)
        ? entry.frames.filter(frame => frame && frame.file && frame.frameIdx !== null && frame.frameIdx !== undefined)
        : [],
    })).filter(entry => entry.frames.length);
  } catch {
    return [];
  }
}

function storeSavedQueries(nextQueries) {
  try {
    localStorage.setItem(MEMORY_STORAGE_KEY, JSON.stringify(nextQueries));
    savedQueries = nextQueries;
    updateMemoryButton();
    renderMemoryPanel();
    return true;
  } catch (error) {
    alert("Could not save queries in this browser: " + error.message);
    return false;
  }
}

function currentSourceQuery() {
  const scene = document.getElementById("inp-query")?.value.trim();
  const sceneWithOperator = scene && scene.includes(",")
    ? `[${activeQueryOperator.toUpperCase()}] ${scene}` : scene;
  const ocr = document.getElementById("inp-ocr")?.value.trim();
  const ocrWithOperator = ocr && ocr.includes(",")
    ? `[${activeOcrOperator.toUpperCase()}] ${ocr}` : ocr;
  if (sceneWithOperator && ocrWithOperator) return `${sceneWithOperator} · OCR: ${ocrWithOperator}`;
  return sceneWithOperator || ocrWithOperator || document.getElementById("inp-match-desc")?.value.trim() ||
    document.getElementById("inp-asr-phrase")?.value.trim() || "";
}

function rememberCurrentQuery({ openPanel = true } = {}) {
  if (!shortlist.length) { alert("Select at least one frame first."); return; }
  if (shortlist.some(frame => frame.frameIdx === null)) {
    alert("Still resolving frame index — wait a second and try again.");
    return;
  }

  const nameInput = document.getElementById("inp-question-num");
  let name = nameInput?.value.trim().replace(/\.csv$/i, "") || "";
  if (!name) {
    const suffix = taskMode === "kis" ? "kis" : taskMode === "vqa" ? "qa" : "trake";
    do {
      name = `query-${queryCounter}-${suffix}`;
      queryCounter++;
    } while (savedQueries.some(entry => entry.name === name));
    if (nameInput) nameInput.value = name;
  }

  const oldEntry = savedQueries.find(entry => entry.name === name);
  const savedEntry = {
    id: oldEntry?.id || `saved-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    name,
    taskMode,
    sourceQuery: currentSourceQuery(),
    savedAt: new Date().toISOString(),
    frames: shortlist.map(frame => ({
      file: frame.file,
      video: frame.video,
      frame: frame.frame,
      frameIdx: frame.frameIdx,
      mapped: frame.mapped,
    })),
  };

  const nextQueries = oldEntry
    ? savedQueries.map(entry => entry.id === oldEntry.id ? savedEntry : entry)
    : [savedEntry, ...savedQueries];
  if (storeSavedQueries(nextQueries) && openPanel) openMemoryPanel();
}

function updateMemoryButton() {
  const count = document.getElementById("memory-count");
  const button = document.getElementById("btn-memory-toggle");
  if (count) count.textContent = savedQueries.length;
  if (button) button.title = `${savedQueries.length} saved quer${savedQueries.length === 1 ? "y" : "ies"}`;
}

function renderMemoryPanel() {
  const list = document.getElementById("memory-list");
  if (!list) return;
  if (!savedQueries.length) {
    list.innerHTML = `
      <div class="memory-empty">
        <div style="font-size:28px;margin-bottom:8px;">🔖</div>
        No saved queries yet.<br>Select frames and export a CSV to save the query automatically.
      </div>`;
    return;
  }

  const modeNames = { kis: "Textual KIS", vqa: "VQA", temporal: "Temporal Alignment" };
  list.innerHTML = savedQueries.map(entry => {
    const savedDate = new Date(entry.savedAt);
    const savedAt = Number.isNaN(savedDate.getTime()) ? "" : savedDate.toLocaleString();
    return `
      <article class="memory-query">
        <div class="memory-query-head">
          <div style="flex:1;min-width:0;">
            <div class="memory-query-title">${escapeHtml(entry.name)}</div>
            <span class="memory-mode">${modeNames[entry.taskMode]}</span>
          </div>
          <button class="memory-delete" data-memory-id="${escapeHtml(entry.id)}"
                  title="Delete saved query" aria-label="Delete ${escapeHtml(entry.name)}">✕</button>
        </div>
        ${entry.sourceQuery ? `<div class="memory-source-query" title="${escapeHtml(entry.sourceQuery)}">🔍 ${escapeHtml(entry.sourceQuery)}</div>` : ""}
        <div class="memory-frames">
          ${entry.frames.map(frame => `
            <button class="memory-frame" data-file="${escapeHtml(frame.file)}"
                    data-video="${escapeHtml(frame.video)}" data-frame-idx="${escapeHtml(frame.frameIdx)}"
                    title="Open ${escapeHtml(frame.video)}, frame ${escapeHtml(frame.frameIdx)}">
              <span>${escapeHtml(frame.video)}</span><strong>#${escapeHtml(frame.frameIdx)}</strong>
            </button>`).join("")}
        </div>
        ${savedAt ? `<div class="memory-saved-at">Saved ${escapeHtml(savedAt)}</div>` : ""}
      </article>`;
  }).join("");

  list.querySelectorAll(".memory-frame").forEach(button => {
    button.addEventListener("click", () => {
      openSavedFrame(button.dataset.file, button.dataset.video, button.dataset.frameIdx);
    });
  });
  list.querySelectorAll(".memory-delete").forEach(button => {
    button.addEventListener("click", () => deleteSavedQuery(button.dataset.memoryId));
  });
}

function toggleMemoryPanel() {
  const panel = document.getElementById("memory-panel");
  panel.classList.contains("open") ? closeMemoryPanel() : openMemoryPanel();
}

function openMemoryPanel() {
  renderMemoryPanel();
  document.getElementById("memory-panel").classList.add("open");
  document.getElementById("memory-panel").setAttribute("aria-hidden", "false");
  dialogFocus.open(document.getElementById("memory-panel"), document.getElementById("memory-close"));
  document.getElementById("memory-backdrop").classList.add("open");
  document.getElementById("btn-memory-toggle").classList.add("active");
  document.getElementById("btn-memory-toggle").setAttribute("aria-expanded", "true");
}

function closeMemoryPanel(instant = false) {
  const panel = document.getElementById("memory-panel");
  const backdrop = document.getElementById("memory-backdrop");
  if (instant) {
    panel.classList.add("instant");
    backdrop.classList.add("instant");
  }
  panel.classList.remove("open");
  panel.setAttribute("aria-hidden", "true");
  dialogFocus.close(panel);
  backdrop.classList.remove("open");
  document.getElementById("btn-memory-toggle").classList.remove("active");
  document.getElementById("btn-memory-toggle").setAttribute("aria-expanded", "false");
  if (instant) requestAnimationFrame(() => requestAnimationFrame(() => {
    panel.classList.remove("instant");
    backdrop.classList.remove("instant");
  }));
}

function deleteSavedQuery(id) {
  const entry = savedQueries.find(item => item.id === id);
  if (!entry || !confirm(`Delete saved query "${entry.name}"?`)) return;
  storeSavedQueries(savedQueries.filter(item => item.id !== id));
}

async function openSavedFrame(file, video, frameIdx) {
  closeMemoryPanel();
  try {
    const meta = await fetch(`${API}/frame/meta/${encodeURIComponent(file)}`).then(response => response.json());
    openLightbox({
      file,
      detections: meta.detections || [],
      ocr_texts: meta.ocr_texts || [],
      answerLabel: `${video},${frameIdx}`,
    });
  } catch {
    openLightbox({ file, detections: [], ocr_texts: [], answerLabel: `${video},${frameIdx}` });
  }
}

updateMemoryButton();
renderMemoryPanel();
function escapeCsvField(value) {
  const text = String(value);
  return `"${text.replace(/"/g, '""')}"`;
}

function exportAnswer() {
  if (!shortlist.length) { alert("Select at least one frame first."); return; }
  if (shortlist.some(s => s.frameIdx === null)) { alert("Still resolving frame index — wait a second and try again."); return; }

  let content;
  if (taskMode === "kis") {
    // one line per frame: video,frame_idx
    content = shortlist.map(s => `${s.video},${s.frameIdx}`).join("\n");

  } else if (taskMode === "vqa") {
    const answerInput = document.getElementById("inp-vqa-answer");
    const answer = answerInput ? answerInput.value : "";
    if (!answer.trim()) { alert("Type the VQA answer before exporting."); return; }
    if (answer.length > 100) { alert("The Q&A answer must be at most 100 characters."); return; }
    // Quoting every answer is valid CSV and safely handles commas and double quotes.
    content = `${shortlist[0].video},${shortlist[0].frameIdx},${escapeCsvField(answer)}`;

  } else { // temporal — exports resolved frame_idx values
    // single line: video,frame_idx1,frame_idx2,...
    const video = shortlist[0].video;
    const ids   = shortlist.map(s => s.frameIdx);
    content = `${video},${ids.join(",")}`;
  }

  const nameInput = document.getElementById("inp-question-num");
  let qname = nameInput.value.trim();
  if (!qname) {
    const suffix = taskMode === "kis" ? "kis" : taskMode === "vqa" ? "qa" : "trake";
    do {
      qname = `query-${queryCounter}-${suffix}`;
      queryCounter++;
    } while (savedQueries.some(entry => entry.name === qname));
  }
  qname = qname.replace(/\.csv$/i, "");
  const blob = new Blob([content], { type: "text/csv;charset=utf-8" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url;
  a.download = `${qname}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  nameInput.value = qname;
  rememberCurrentQuery({ openPanel: false });
  recordEvaluation("answer", content, qname);
  nameInput.value = "";
}

function renderSearchPager(id, page, total, navigate) {
  const host = document.getElementById(id);
  const pages = Math.max(1, Math.ceil(total / 1000));
  host.hidden = !total;
  host.replaceChildren();
  const previous = document.createElement('button');
  previous.textContent = 'Previous'; previous.disabled = page <= 1;
  previous.onclick = () => navigate(page - 1);
  const input = document.createElement('input');
  input.type = 'number'; input.min = 1; input.max = pages; input.value = page;
  input.setAttribute('aria-label', 'Trang');
  input.onchange = () => navigate(Math.max(1, Math.min(pages, parseInt(input.value) || 1)));
  const next = document.createElement('button');
  next.textContent = 'Next'; next.disabled = page >= pages;
  next.onclick = () => navigate(page + 1);
  host.append(previous, input, document.createTextNode(' / ' + pages), next);
}
async function kiscLoadPage(page) {
  if (kiscBusy || !kiscSessionId) return;
  kiscBusy = true;
  document.getElementById('kisc-send').disabled = true;
  try {
    const params = new URLSearchParams({session_id: kiscSessionId, page});
    const response = await trackedSearch(`${API}/kis/results?${params}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Page unavailable');
    kiscLastResults = data.results;
    kiscRenderGrid(data.results, data.total, false);
    renderSearchPager('kisc-pagination', data.page, data.total, kiscLoadPage);
    document.getElementById('kisc-results-wrap').scrollTop = 0;
  } catch (error) { kiscAddMessage('bot', error.message, 'error'); }
  finally { kiscBusy = false; document.getElementById('kisc-send').disabled = false; }
}
// Keep the first response and DOM update comfortably inside the interactive
// latency budget. Additional pages are fetched from the bounded candidate set.
const FRAME_PAGE_SIZE = 200;
const RENDER_BATCH_SIZE = 40;
let serverFramePagination = null;
let framePage = 1;
function setFramePage(page) {
  const count = serverFramePagination ? serverFramePagination.total : lastResults.length;
  const totalPages = Math.max(1, Math.ceil(count / FRAME_PAGE_SIZE));
  const nextPage = Math.max(1, Math.min(totalPages, parseInt(page) || 1));
  if (nextPage === framePage) return;
  if (serverFramePagination) { doSearch(nextPage, true); return; }
  framePage = nextPage;
  renderCurrentFramePage();
}
function renderResults(results, total, autoFiltered = [], operator = "semantic", terms = [], deduplicated = 0, pagination = null) {
  serverFramePagination = pagination;
  lastResults = results;
  lastResultContext = {total, autoFiltered, operator, terms, deduplicated};
  framePage = pagination ? pagination.page : 1;
  renderCurrentFramePage();
}
function renderCurrentFramePage() {
  resultImages.prune(document.getElementById("grid"));
  const start = (framePage - 1) * FRAME_PAGE_SIZE;
  const pageResults = serverFramePagination ? lastResults : lastResults.slice(start, start + FRAME_PAGE_SIZE);
  const ctx = lastResultContext;
  renderFrameCards(pageResults, ctx.total, ctx.autoFiltered, ctx.operator, ctx.terms, ctx.deduplicated);
  if (groupMode) renderGrouped(pageResults);
  const totalPages = Math.max(1, Math.ceil((serverFramePagination ? serverFramePagination.total : lastResults.length) / FRAME_PAGE_SIZE));
  document.getElementById("frame-pagination").hidden = !lastResults.length;
  document.getElementById("inp-frame-page").value = framePage;
  document.getElementById("inp-frame-page").max = totalPages;
  document.getElementById("frame-total-pages").textContent = ` / ${totalPages}`;
  document.getElementById("btn-frame-prev").disabled = framePage <= 1;
  document.getElementById("btn-frame-next").disabled = framePage >= totalPages;
  document.getElementById("results-wrap").scrollTop = 0;
}

function renderFrameCards(
  results,
  total,
  autoFiltered = [],
  queryOperator = "semantic",
  queryTerms = [],
  deduplicated = 0,
) {
  const grid  = document.getElementById("grid");
  const meta  = document.getElementById("results-meta");
  const empty = document.getElementById("empty");
  const generation = ++renderGeneration;
  const resultsWrap = document.getElementById("results-wrap");
  if (activeResultsScrollHandler) {
    resultsWrap.removeEventListener("scroll", activeResultsScrollHandler);
    activeResultsScrollHandler = null;
  }
  const isAndQuery = queryOperator === "and" && queryTerms.length > 1;
  const isOrQuery = queryOperator === "or" && queryTerms.length > 1;
  const safeQueryTerms = queryTerms.map(term => escapeHtml(term));
  resultImages.prune(grid);
  grid.innerHTML = "";

  if (!results.length) {
    meta.textContent = "";
    empty.style.display = "block";
    if (serverFramePagination?.partial) {
      empty.textContent = "Tìm kiếm đang dùng dữ liệu chưa đầy đủ. " +
        (serverFramePagination.warnings.length
          ? serverFramePagination.warnings.join("; ")
          : "Một phần dữ liệu tìm kiếm chưa sẵn sàng; xem sever_log.json.");
      return;
    }
    empty.innerHTML = isAndQuery
      ? `No frame matched <strong>all</strong> AND terms: ${safeQueryTerms.join(" + ")}.`
      : isOrQuery
        ? `No frame matched <strong>any</strong> OR term: ${safeQueryTerms.join(" / ")}.`
        : "No frames matched. Try different keywords or filters.";
    return;
  }

  empty.style.display = "none";
  const autoNote = autoFiltered.length
    ? ` &nbsp;·&nbsp; <span style="color:#2d7a4f;font-weight:600">auto-filtered: ${autoFiltered.join(", ")}</span>`
    : "";
  const andNote = isAndQuery
    ? ` &nbsp;·&nbsp; <span style="color:#c56a00;font-weight:700">AND: ${safeQueryTerms.join(" + ")}</span>`
    : "";
  const orNote = isOrQuery
    ? ` &nbsp;·&nbsp; <span style="color:#7d3c98;font-weight:700">OR: ${safeQueryTerms.join(" ∪ ")}</span>`
    : "";
  const similarNote = queryOperator === "similar"
    ? ` &nbsp;·&nbsp; <span style="color:#8e44ad;font-weight:700">VISUALLY SIMILAR</span>`
    : "";
  const dedupeNote = deduplicated
    ? ` &nbsp;·&nbsp; ${deduplicated} nearby duplicates hidden`
    : "";
  meta.innerHTML = `Showing ${results.length} of ${total} frames${similarNote}${andNote}${orNote}${autoNote}${dedupeNote}`;

  if (groupMode) return;

  const maxScore = Math.max(...results.map(r => r.score), 0.01);
  const renderQueue = results.slice();
  const appendBatch = () => {
    if (generation !== renderGeneration) return;
    const fragment = document.createDocumentFragment();
    renderQueue.splice(0, RENDER_BATCH_SIZE).forEach(item => {
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.file = item.file;
    if (isShortlisted(item.file)) card.classList.add("selected");
    const pct = Math.round((item.score / maxScore) * 100);
    const topDet = item.detections.length ? [...item.detections].sort((a,b)=>b.conf-a.conf)[0] : null;
    card.innerHTML = `
      <div class="card-img-wrap">
        <img src="${API}/image/${encodeURIComponent(item.file)}" loading="lazy" alt="${escapeHtml(item.file)}">
        <canvas></canvas>
        <div class="select-badge" title="Add to answer shortlist">✓</div>
      </div>
      <div class="card-info">
        <div class="card-filename" title="${escapeHtml(item.file)}">${escapeHtml(fileToAnswer(item.file).video)} / ${escapeHtml(item.file.split('/').pop())}</div>
        <div class="card-score-bar"><div class="card-score-fill" style="width:${pct}%"></div></div>
        <div class="card-tags">
          <span class="tag">Score: ${Number(item.score).toFixed(4)}</span>
          ${item.source === 'summary' ? `<span class="tag">Summary · ${new Date(item.timestamp * 1000).toISOString().slice(11,19)} · RRF ${item.score.toFixed(4)}</span>` : ''}
          ${topDet
            ? `<span class="tag" style="background:#f0f7ee;color:#2d7a4f">${escapeHtml(topDet.class)}</span>
               <span class="tag">${Math.round(topDet.conf*100)}% conf</span>`
            : ""}
          ${item.ocr_matched ? `<span class="ocr-badge">📄 OCR</span>` : ""}
        </div>
      </div>`;
    if (item.summary) card.title = item.summary;
    const imgEl = card.querySelector("img");
    setupResultImage(imgEl);
    const cvs   = card.querySelector("canvas");
    const ctx   = cvs.getContext("2d");
    imgEl.addEventListener("mouseenter", () => drawBoxes(imgEl, cvs, ctx, item.detections, item.ocr_texts));
    imgEl.addEventListener("mousemove",  e => updateTooltip(e, imgEl, item.detections, item.ocr_texts));
    imgEl.addEventListener("mouseleave", () => { ctx.clearRect(0,0,cvs.width,cvs.height); hideTooltip(); });

    // click checkmark badge → toggle shortlist (UI refresh handled inside toggleShortlist)
    card.querySelector(".select-badge").onclick = (e) => {
      e.stopPropagation();
      toggleShortlist(item.file);
    };
    card.querySelector(".card-info").onclick = () => openLightbox(item);
    imgEl.onclick = () => openLightbox(item);
    makeKeyboardAction(card.querySelector(".card-info"), () => openLightbox(item), `Open ${item.file}`);
    makeKeyboardAction(card.querySelector(".select-badge"), () => toggleShortlist(item.file), `Toggle answer selection for ${item.file}`);

      fragment.appendChild(card);
    });
    grid.appendChild(fragment);
    if (!renderQueue.length && activeResultsScrollHandler) {
      resultsWrap.removeEventListener("scroll", activeResultsScrollHandler);
      activeResultsScrollHandler = null;
    }
  };
  activeResultsScrollHandler = () => {
    if (generation !== renderGeneration) return;
    const nearBottom = resultsWrap.scrollTop + resultsWrap.clientHeight >= resultsWrap.scrollHeight - 700;
    if (nearBottom) appendBatch();
  };
  resultsWrap.addEventListener("scroll", activeResultsScrollHandler);
  appendBatch();
  // Fill tall viewports without waiting for a scroll that cannot happen yet.
  const fillViewport = () => {
    if (generation !== renderGeneration || !renderQueue.length) return;
    if (resultsWrap.scrollHeight <= resultsWrap.clientHeight + 700) {
      appendBatch();
      requestAnimationFrame(fillViewport);
    }
  };
  requestAnimationFrame(fillViewport);
}

// ── bounding boxes ────────────────────────────────────────────────────────────
function drawBoxes(imgEl, cvs, ctx, detections, ocrTexts = []) {
  cvs.width  = imgEl.offsetWidth;
  cvs.height = imgEl.offsetHeight;
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  const W = cvs.width, H = cvs.height;

  detections.forEach(d => {
    const [x1,y1,x2,y2] = d.box;
    const col = colorFor(d.class);
    const px1=x1*W, py1=y1*H, pw=(x2-x1)*W, ph=(y2-y1)*H;
    ctx.strokeStyle=col; ctx.lineWidth=1.5; ctx.strokeRect(px1,py1,pw,ph);
    const label=`${d.class} ${Math.round(d.conf*100)}%`;
    ctx.font="bold 10px system-ui";
    const tw=ctx.measureText(label).width, lh=13;
    const ly = py1>lh ? py1-lh : py1+ph;
    ctx.fillStyle=col; ctx.fillRect(px1,ly,tw+6,lh);
    ctx.fillStyle="#fff"; ctx.fillText(label,px1+3,ly+lh-3);
  });

  ocrTexts.forEach(t => {
    const [x1,y1,x2,y2] = t.box;
    const px1=x1*W, py1=y1*H, pw=(x2-x1)*W, ph=(y2-y1)*H;
    ctx.strokeStyle="#e67e22"; ctx.lineWidth=1.5;
    ctx.setLineDash([4,3]); ctx.strokeRect(px1,py1,pw,ph); ctx.setLineDash([]);
    const label=t.text.length>20?t.text.slice(0,20)+"…":t.text;
    ctx.font="10px system-ui";
    const tw=ctx.measureText(label).width, lh=13, ly=py1+ph;
    ctx.fillStyle="#e67e22"; ctx.fillRect(px1,ly,tw+6,lh);
    ctx.fillStyle="#fff"; ctx.fillText(label,px1+3,ly+lh-3);
  });
}

// ── tooltip ───────────────────────────────────────────────────────────────────
function updateTooltip(e, imgEl, detections, ocrTexts = []) {
  const rect = imgEl.getBoundingClientRect();
  const mx=(e.clientX-rect.left)/rect.width, my=(e.clientY-rect.top)/rect.height;
  let hit=null, minArea=Infinity;
  detections.forEach(d => {
    const [x1,y1,x2,y2]=d.box;
    if (mx>=x1&&mx<=x2&&my>=y1&&my<=y2) {
      const a=(x2-x1)*(y2-y1);
      if (a<minArea) { minArea=a; hit={label:`${d.class}  ${(d.conf*100).toFixed(1)}%`,color:"#333"}; }
    }
  });
  if (!hit) ocrTexts.forEach(t => {
    const [x1,y1,x2,y2]=t.box;
    if (mx>=x1&&mx<=x2&&my>=y1&&my<=y2) {
      const a=(x2-x1)*(y2-y1);
      if (a<minArea) { minArea=a; hit={label:`📄 "${t.text}"  ${(t.conf*100).toFixed(1)}%`,color:"#e67e22"}; }
    }
  });
  const tip=document.getElementById("tooltip");
  if (hit) {
    tip.textContent=hit.label; tip.style.color=hit.color; tip.style.display="block";
    tip.style.left=(e.clientX+12)+"px"; tip.style.top=(e.clientY-28)+"px";
  } else hideTooltip();
}
function hideTooltip() { document.getElementById("tooltip").style.display="none"; }

// ── lightbox ──────────────────────────────────────────────────────────────────
let currentLightboxItem = null;
let lightboxRequestToken = 0;
function openLightbox(item) {
  if (!currentLightboxItem) document.getElementById("lb-timeline").checked = document.getElementById("inp-timeline").checked;
  currentLightboxItem = item;
  const lbImg=document.getElementById("lb-img");
  const lbCvs=document.getElementById("lb-canvas");
  const ctx=lbCvs.getContext("2d");
  const detections = item.detections || [];
  const ocrTexts = item.ocr_texts || [];
  lbImg.onload = () => {
    lbCvs.width=lbImg.offsetWidth; lbCvs.height=lbImg.offsetHeight;
    drawBoxes(lbImg, lbCvs, ctx, detections, ocrTexts);
  };
  lbImg.src=`${API}/image/${encodeURIComponent(item.file)}`;
  const top=[...detections].sort((a,b)=>b.conf-a.conf)[0];
  const ans = fileToAnswer(item.file);
  const answerLabel = item.answerLabel || `${ans.video},${ans.frame}`;
  const scoreMeta = Number.isFinite(item.score)
    ? ` &nbsp;·&nbsp; ${(item.score*100).toFixed(1)}% match`
    : "";
  document.getElementById("lb-meta").innerHTML =
    `<strong>${escapeHtml(item.file)}</strong>${scoreMeta}` +
    (top ? ` &nbsp;·&nbsp; ${escapeHtml(top.class)} <strong>${Math.round(top.conf*100)}%</strong>` : "") +
    ` &nbsp;·&nbsp; <span style="font-family:monospace;color:#888">${escapeHtml(answerLabel)}</span>` +
    ` &nbsp; <button id="lb-shortlist-btn" class="primary-button">${isShortlisted(item.file) ? '✓ Added' : '+ Add to answer'}</button>`;
  document.getElementById("lb-shortlist-btn").onclick = () => toggleShortlist(item.file).then(() => updateLbShortlistBtn(item.file));
  const similarButton = document.createElement("button");
  similarButton.id = "lb-similar-btn";
  similarButton.textContent = "◎ Find Similar";
  similarButton.style.cssText =
    "margin-left:6px;background:#8e44ad;color:#fff;border:none;padding:4px 10px;border-radius:4px;font-size:11px;cursor:pointer";
  similarButton.onclick = findSimilarFromLightbox;
  document.getElementById("lb-meta").appendChild(similarButton);
  document.getElementById("lightbox").classList.add("open");
  lbImg.alt = item.file;
  dialogFocus.open(document.getElementById("lightbox"), document.getElementById("lb-close"));
  loadFrameNeighbors(item.file);
}
async function loadFrameNeighbors(file) {
  const token = ++lightboxRequestToken;
  const strip = document.getElementById("lb-neighbors");
  strip.textContent = "Loading nearby frames…";
  try {
    const params = new URLSearchParams({ file, radius: 6 });
    const useTimeline = document.getElementById("lb-timeline").checked;
    const response = await fetch(`${API}/frame/${useTimeline ? "timeline" : "neighbors"}?${params}`);
    const data = await response.json();
    if (token !== lightboxRequestToken || currentLightboxItem?.file !== file) return;
    if (!response.ok || data.error) throw new Error(data.error || "Neighbor lookup failed");
    strip.innerHTML = "";
    data.results.forEach(item => {
      const button = document.createElement("button");
      button.className = `lb-neighbor${item.is_center ? " center" : ""}`;
      button.title = item.file + (item.timestamp == null ? "" : ` · ${fmtTime(item.timestamp)}`);
      const image = document.createElement("img");
      image.src = `${API}/image/${encodeURIComponent(item.file)}`;
      image.alt = item.file;
      image.loading = "lazy";
      button.appendChild(image);
      button.onclick = () => openAsrFrame(item.file);
      strip.appendChild(button);
    });
    strip.querySelector(".center")?.scrollIntoView({ inline: "center", block: "nearest" });
  } catch (error) {
    if (token === lightboxRequestToken) strip.textContent = "Nearby frames unavailable";
  }
}

function reloadLightboxContext() {
  if (currentLightboxItem) loadFrameNeighbors(currentLightboxItem.file);
}

async function findSimilarFromLightbox() {
  const source = currentLightboxItem;
  const button = document.getElementById("lb-similar-btn");
  if (!source || !button) return;
  const requestSerial = ++frameRequestSerial;
  frameRequestController?.abort();
  const controller = new AbortController();
  frameRequestController = controller;
  clearSearchError();
  button.disabled = true;
  button.textContent = "Finding…";
  try {
    const params = new URLSearchParams({ file: source.file, page: 1 });
    const response = await trackedSearch(`${API}/similar?${params}`, {signal: controller.signal});
    const data = await readJsonResponse(response);
    if (requestSerial !== frameRequestSerial) return;
    lastResults = data.results || [];
    lastResultContext = {
      total: data.total || lastResults.length,
      autoFiltered: [],
      operator: "similar",
      terms: [],
      deduplicated: 0,
    };
    closeLightbox();
    renderResults(lastResults, data.total || lastResults.length, [], "similar", [], 0,
      {page: data.page || 1, total: data.total, endpoint: "/similar", params: params.toString()});
    updateSidebarCounts(lastResults);
    document.getElementById("results-wrap").scrollTop = 0;
  } catch (error) {
    if (requestSerial === frameRequestSerial && error.name !== 'AbortError') {
      closeLightbox();
      showSearchError(error.message);
    }
  } finally {
    if (requestSerial === frameRequestSerial) setLoading(false);
    button.disabled = false;
    button.textContent = "◎ Find Similar";
  }
}

function updateLbShortlistBtn(file) {
  const btn = document.getElementById("lb-shortlist-btn");
  if (!btn) return;
  const added = isShortlisted(file);
  btn.textContent = added ? "✓ Added" : "+ Add to answer";
  btn.style.background = added ? "#27ae60" : "#4a90e2";
}
function closeLightbox() {
  lightboxRequestToken++;
  currentLightboxItem = null;
  document.getElementById("lightbox").classList.remove("open");
  dialogFocus.close(document.getElementById("lightbox"));
}
document.getElementById("lightbox").addEventListener("click", e => { if(e.target===e.currentTarget) closeLightbox(); });
document.addEventListener("keydown", e => {
  if (e.key === "Escape") {
    if (currentLightboxItem) closeLightbox();
    else closeMemoryPanel(true);
  }
});

// ── helpers ───────────────────────────────────────────────────────────────────
function setLoading(v) {
  document.getElementById("btn-search").disabled = v || !searchReady;
  document.getElementById("spinner").classList.toggle("active", v);
  document.getElementById("results-wrap").setAttribute("aria-busy", String(v));
  if (v) {
    renderGeneration++;
    if (activeResultsScrollHandler) {
      document.getElementById("results-wrap").removeEventListener("scroll", activeResultsScrollHandler);
      activeResultsScrollHandler = null;
    }
    resultImages.prune(document.getElementById("grid"));
    document.getElementById("grid").innerHTML="";
    document.getElementById("results-meta").textContent = "";
    document.getElementById("frame-pagination").hidden = true;
    document.getElementById("empty").style.display="none";
  }
}
function switchTab(name) {
  document.body.classList.toggle('dres-active', name === 'dres');
  document.querySelectorAll(".tab").forEach(tab => {
    const selected = tab.dataset.page === name;
    tab.classList.toggle("active", selected);
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  document.querySelectorAll(".page").forEach(page => page.classList.toggle("active", page.id === `page-${name}`));
  if (name === "settings") {
    refreshHealth();
    refreshBackupList();
    pollIndexProgress();
  }
  if (name === "kisc" && !kiscSessionId) {
    kiscNewSession();
  }
}

// ── ASR ───────────────────────────────────────────────────────────────────────
async function openAsrFrame(file) {
  try {
    const meta = await fetch(`${API}/frame/meta/${encodeURIComponent(file)}`).then(r => r.json());
    openLightbox({
      file,
      score: 1,
      detections: meta.detections || [],
      ocr_texts: meta.ocr_texts || [],
    });
  } catch {
    openLightbox({ file, score: 1, detections: [], ocr_texts: [] });
  }
}

async function loadAsrConfig() {
  const input = document.getElementById("inp-asr-folder");
  const status = document.getElementById("asr-index-status");
  const transcribeBtn = document.getElementById("btn-asr-transcribe");
  try {
    const response = await fetch(`${API}/asr/config`);
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
    if (input) input.value = data.videos_folder || "";
    if (transcribeBtn) {
      transcribeBtn.disabled = !data.transcription_enabled;
      transcribeBtn.title = data.transcription_ready
        ? (data.transcription_enabled ? "" : "Transcription is disabled in serving mode")
        : "Original videos are unavailable; cached ASR Search remains usable";
    }
  } catch (error) {
    // Runtime transcription is optional: imported transcripts must remain
    // searchable even when the source-video configuration route is absent.
    if (transcribeBtn) transcribeBtn.disabled = true;
  }
  await fetchAsrStatus();
}

async function fetchAsrStatus() {
  const el = document.getElementById("asr-index-status");
  try {
    const response = await fetch(`${API}/asr/status`);
    const d = await response.json();
    if (!response.ok || d.error) throw new Error(d.error || `HTTP ${response.status}`);
    if (d.search_ready) {
      const sourceNote = d.videos_folder_exists
        ? ""
        : " · original videos removed";
      el.textContent = `✓ ASR cache ready: ${d.valid_cached_videos} video(s)${sourceNote}`;
      el.style.color = "#27ae60";
    } else if (d.cache_exists) {
      el.textContent = `ASR cache has ${d.invalid_cached_videos} invalid video(s)`;
      el.style.color = "#e67e22";
    } else {
      el.textContent = "ASR cache not found";
      el.style.color = "#e74c3c";
    }
  } catch {
    el.textContent = "Could not load ASR status";
    el.style.color = "#e74c3c";
  }
}

async function doTranscribeAll() {
  const btn     = document.getElementById("btn-asr-transcribe");
  const statusEl= document.getElementById("asr-index-status");
  btn.disabled  = true; btn.textContent="Transcribing…";
  statusEl.textContent="Transcribing — check terminal for progress…"; statusEl.style.color="#8e44ad";
  try {
    const d = await fetch(`${API}/asr/transcribe`,{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"}).then(r=>r.json());
    if (d.error) alert(d.error);
    else { statusEl.textContent=`✓ ${d.transcribed} new, ${d.skipped_cached} cached`; statusEl.style.color="#27ae60"; }
  } catch(e) { alert("Transcribe failed: "+e); }
  btn.textContent="⚡ Transcribe All";
  await loadAsrConfig();
}

const asrSearchState = {page: 1, total: 0, pageSize: 1000, phrase: "", operator: "and", loading: false};
function readAsrPageSize() {
  return 1000;
}
function setAsrPage(page) {
  if (asrSearchState.loading) return;
  const totalPages = Math.max(1, Math.ceil(asrSearchState.total / asrSearchState.pageSize));
  doAsrSearch(Math.max(1, Math.min(totalPages, parseInt(page) || 1)), true);
}
function updateAsrPagination() {
  const totalPages = Math.max(1, Math.ceil(asrSearchState.total / asrSearchState.pageSize));
  document.getElementById("asr-pagination").hidden = !asrSearchState.total;
  document.getElementById("inp-asr-page").value = asrSearchState.page;
  document.getElementById("inp-asr-page").max = totalPages;
  document.getElementById("inp-asr-page").disabled = asrSearchState.loading;
  document.getElementById("asr-total-pages").textContent = ` / ${totalPages}`;
  document.getElementById("btn-asr-prev").disabled = asrSearchState.loading || asrSearchState.page <= 1;
  document.getElementById("btn-asr-next").disabled = asrSearchState.loading || asrSearchState.page >= totalPages;
}
async function doAsrSearch(page = 1, reuse = false) {
  if (asrSearchState.loading) return;
  const phrase = reuse ? asrSearchState.phrase : document.getElementById("inp-asr-phrase").value.trim();
  const operator = reuse ? asrSearchState.operator : activeAsrOperator;
  const pageSize = readAsrPageSize();
  let targetPage = typeof page === "number" ? page : 1;
  if (phrase !== asrSearchState.phrase || pageSize !== asrSearchState.pageSize || operator !== asrSearchState.operator) targetPage = 1;
  const requestOffset = (targetPage - 1) * pageSize;

  if (!phrase) {
    alert("Enter a phrase.");
    return;
  }

  const btn = document.getElementById("btn-asr-search");
  const spinner = document.getElementById("asr-spinner");
  const grid = document.getElementById("asr-grid");
  const meta = document.getElementById("asr-meta");
  const empty = document.getElementById("asr-empty");

  asrSearchState.loading = true;
  updateAsrPagination();
  btn.disabled = true;
  spinner.classList.add("active");
  resultImages.prune(grid);
  grid.innerHTML = "";
  meta.textContent = "";
  empty.style.display = "none";

  try {
    const response = await trackedSearch(`${API}/asr/frames`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ phrase, operator,
        page: targetPage,
      }),
    });

    const data = await response.json();
    if (!response.ok || data.error) {
      throw new Error(data.error || `HTTP ${response.status}`);
    }

    targetPage = data.page;
    Object.assign(asrSearchState, {page: targetPage, total: data.total, pageSize, phrase, operator: data.operator});
    updateAsrPagination();
    if (!data.results || !data.results.length) {
      empty.style.display = "block";
      empty.innerHTML =
        `No 2-second ASR window found for "<strong>${escapeHtml(phrase)}</strong>"`;
      return;
    }

    meta.textContent =
      `${data.results.length} / ${data.total} ảnh · Trang ${targetPage} · ` +
      `${data.matched_intervals} đoạn khớp · ${data.unmapped_intervals} đoạn không có keyframe`;

    data.results.forEach(item => {
      const card = document.createElement("div");
      card.className = "asr-card";

      const framesHtml = (item.frames || []).length
        ? item.frames.map((frame, index) => `
          <div class="asr-frame-wrap"
               data-file="${escapeHtml(frame.file)}"
               title="${escapeHtml(frame.file)}">
            <img src="${API}/image/${encodeURIComponent(frame.file)}"
                 loading="lazy"
                 alt="${escapeHtml(frame.file)}">
            <span class="asr-frame-label">
              ${frame.asr_position === 'before' ? 'Trước đoạn' : frame.asr_position === 'after' ? 'Sau đoạn' : `Frame ${index + 1}/${item.frames.length}`}
            </span>
            <span class="asr-frame-ts">${fmtTime(frame.timestamp)}</span>
          </div>
        `).join("")
        : `
          <div class="asr-frame-empty">
            ${escapeHtml(
              item.frame_warning ||
              "No existing keyframe is mapped inside this spoken interval."
            )}
          </div>
        `;

      card.innerHTML = `
        <div class="asr-card-header">
          <span class="asr-video-name">🎬 ${escapeHtml(item.video)}</span>
          <span class="asr-time">
            ${fmtTime(item.start)} → ${fmtTime(item.end)}
          </span>
          <span class="asr-duration">
            ${item.duration}s · ${item.frame_count} frame(s)
          </span>
        </div>
        <div class="asr-text">${highlightPhrase(item.text, phrase)}</div>
        <div>BM25: ${Number(item.score).toFixed(6)}</div>
        <div class="asr-frames">${framesHtml}</div>
      `;

      card.querySelectorAll(".asr-frame-wrap").forEach(frameElement => {
        frameElement.addEventListener(
          "click",
          () => openAsrFrame(frameElement.dataset.file)
        );
      });
      card.querySelectorAll(".asr-frame-wrap").forEach(element => makeKeyboardAction(element, () => openAsrFrame(element.dataset.file), `Open ${element.dataset.file}`));
      card.querySelectorAll("img").forEach(setupResultImage);
      grid.appendChild(card);
    });
  } catch (error) {
    alert("ASR search failed: " + error.message);
  } finally {
    asrSearchState.loading = false;
    updateAsrPagination();
    btn.disabled = false;
    spinner.classList.remove("active");
  }
}

document.getElementById("btn-asr-search").onclick = doAsrSearch;
["inp-asr-phrase"].forEach(id =>
  document.getElementById(id).addEventListener("keydown", e => { if (e.key === "Enter") doAsrSearch(); })
);

// ── video match ───────────────────────────────────────────────────────────────
let matchAbortController = null;

let matchPageState = {page: 1, total: 0, query: ""};
async function doVideoMatch(page = 1, reuse = false) {
  const desc = reuse ? matchPageState.query : document.getElementById("inp-match-desc").value.trim();

  if (!desc) { alert("Describe what you saw in the clip."); return; }

  const btn = document.getElementById("btn-match");
  const spinner = document.getElementById("match-spinner");
  const results = document.getElementById("match-results");
  const meta = document.getElementById("match-meta");
  const empty = document.getElementById("match-empty");

  // Cancel an older request when the user immediately submits a new query.
  if (matchAbortController) matchAbortController.abort();
  const controller = new AbortController();
  matchAbortController = controller;

  const oldLabel = "🔍 Find Video";
  btn.disabled = true;
  btn.textContent = "Searching…";
  spinner.textContent = "Searching CLIP/FAISS index…";
  spinner.style.display = "block";
  resultImages.prune(results);
  results.innerHTML = "";
  meta.textContent = "";
  empty.style.display = "none";

  try {
    const params = new URLSearchParams({
      query: desc,
      page: String(page),
    });
    const response = await trackedSearch(`${API}/video-match/frames?${params}`, { signal: controller.signal });
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        detail = body.error || body.detail || detail;
      } catch {}
      throw new Error(detail);
    }

    const d = await response.json();
    if (matchAbortController !== controller) return;
    if (d.error) throw new Error(d.error);

    matchPageState = {page: d.page, total: d.total, query: desc};
    renderSearchPager("match-pagination", d.page, d.total, p => doVideoMatch(p, true));
    const videos = d.videos || [];
    if (!videos.length) {
      empty.style.display = "block";
      empty.textContent = "No matching frames found. Try a different visual description.";
      return;
    }

    const totalMs = d.timings?.total_ms ?? "?";
    meta.textContent = `${d.returned_frames} / ${d.total} ảnh · Trang ${d.page} · ${d.total_videos} video`;

    const rankColors = ["#c0392b", "#e67e22", "#f1c40f", "#bdc3c7"];
    results.innerHTML = videos.map((g, i) => {
      const col = rankColors[Math.min(i, rankColors.length - 1)];
      const frameHtml = (g.frames || []).map(f => {
        const selected = isShortlisted(f.file) ? " selected" : "";
        return `
          <div class="match-frame${selected}" data-file="${escapeHtml(f.file)}"
               style="position:relative;width:100%;min-width:0;">
            <img src="${API}/image/${encodeURIComponent(f.file)}" loading="lazy"
              style="width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:4px;cursor:pointer;display:block;"
              alt="${escapeHtml(f.file)}" title="${escapeHtml(f.file)}">
            <button type="button" class="select-badge" title="Add to answer shortlist"
                    aria-label="Toggle answer selection for ${escapeHtml(f.file)}"></button>
            <span style="position:absolute;right:4px;bottom:4px;background:rgba(0,0,0,.68);
                         color:#fff;font-size:10px;padding:2px 5px;border-radius:3px;pointer-events:none;">
              ${(f.score * 100).toFixed(1)}%
            </span>
          </div>`;
      }).join("");

      return `
        <div style="display:flex;align-items:stretch;background:#fff;border:1px solid #e0e0e0;
                    border-radius:5px;margin-bottom:8px;overflow:hidden;border-left:4px solid ${col}">
          <div style="display:flex;align-items:center;justify-content:center;
                      min-width:40px;font-size:20px;font-weight:800;color:${col};padding:0 6px;">#${i + 1}</div>
          <div style="flex:1;padding:8px 6px;min-width:0;">
            <div style="font-weight:700;font-size:13px;margin-bottom:6px;white-space:nowrap;
                        overflow:hidden;text-overflow:ellipsis;">🎬 ${escapeHtml(g.video)}</div>
            <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;width:100%;">
              ${frameHtml}
            </div>
          </div>
          <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                      min-width:92px;padding:0 10px;border-left:1px solid #eee;">
            <div style="font-size:21px;font-weight:700;color:${i === 0 ? '#c0392b' : '#555'}">
              ${(g.score * 100).toFixed(1)}%
            </div>
            <div style="font-size:10px;color:#888;text-align:center;">video score</div>
            <div style="font-size:10px;color:#aaa;margin-top:3px;">${g.candidate_frame_count} candidates</div>
          </div>
        </div>`;
    }).join("");
    const matchedFrames = new Map(videos.flatMap(group => group.frames || []).map(frame => [frame.file, frame]));
    results.querySelectorAll('.match-frame').forEach(element => {
      const frame = matchedFrames.get(element.dataset.file);
      const image = element.querySelector('img');
      image.onclick = () => openLightbox(frame);
      makeKeyboardAction(image, () => openLightbox(frame), `Open ${frame.file}`);
      element.querySelector('button').onclick = () => toggleMatchShortlist(frame.file);
      setupResultImage(image);
    });
    updateMatchSelectionState();
  } catch (e) {
    if (matchAbortController === controller && e.name !== "AbortError") {
      empty.style.display = "block";
      empty.textContent = "Video search failed.";
      alert("Search failed: " + e.message);
    }
  } finally {
    // Only the newest request is allowed to change the shared loading state.
    if (matchAbortController === controller) {
      matchAbortController = null;
      btn.disabled = false;
      btn.textContent = oldLabel;
      spinner.style.display = "none";
    }
  }
}

async function toggleMatchShortlist(file) {
  await toggleShortlist(file);
  updateMatchSelectionState();
}

function updateMatchSelectionState() {
  document.querySelectorAll(".match-frame").forEach(frame => {
    frame.classList.toggle("selected", isShortlisted(frame.dataset.file));
  });
}

// ── Conversational KIS (KISC) ────────────────────────────────────────────────
let kiscSessionId = null;
let kiscLastResults = [];
let kiscBusy = false;

async function kiscNewSession() {
  try {
    const d = await fetch(`${API}/kis/start`, { method: "POST" }).then(r => r.json());
    kiscSessionId = d.session_id;
    kiscLastResults = [];
    document.getElementById("kisc-pagination").hidden = true;
    document.getElementById("kisc-messages").innerHTML =
      `<div class="kisc-msg bot">New session started. Describe what you're looking for — you don't need to get it right the first time, I'll ask follow-up questions to narrow it down.</div>`;
    document.getElementById("kisc-slots").innerHTML = "";
    document.getElementById("kisc-grid").innerHTML = "";
    document.getElementById("kisc-meta").textContent = "";
    document.getElementById("kisc-empty").style.display = "block";
  } catch (e) {
    kiscAddMessage("bot", "⚠ Could not start session: " + e, "error");
  }
}

function kiscAddMessage(role, text, extraClass = "") {
  const box = document.getElementById("kisc-messages");
  const div = document.createElement("div");
  div.className = `kisc-msg ${role}${extraClass ? " " + extraClass : ""}`;
  div.textContent = text;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

async function kiscSend() {
  if (kiscBusy) return;
  const inp = document.getElementById("inp-kisc");
  const message = inp.value.trim();

  if (!message) return;
  if (!kiscSessionId) await kiscNewSession();

  kiscAddMessage("user", message);
  inp.value = "";
  kiscBusy = true;
  document.getElementById("kisc-send").disabled = true;

  try {
    const d = await trackedSearch(`${API}/kis/turn`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: kiscSessionId, message }),
    }).then(r => r.json());

    if (d.error) {
      kiscAddMessage("bot", "⚠ " + d.error, "error");
    } else {
      if (d.clarifying) {
        kiscAddMessage("bot", d.question, "question");
      } else {
        kiscAddMessage("bot", `Narrowed to ${d.pool_size} matching frame(s) — here are the top results.`);
      }
      kiscRenderSlots(d.slots_filled, d.filters);
      kiscLastResults = d.results || [];
      kiscRenderGrid(kiscLastResults, d.pool_size, d.clarifying);
      renderSearchPager("kisc-pagination", 1, d.pool_size, kiscLoadPage);
    }
  } catch (e) {
    kiscAddMessage("bot", "⚠ Request failed: " + e, "error");
  }

  kiscBusy = false;
  document.getElementById("kisc-send").disabled = false;
  inp.focus();
}

function kiscRenderSlots(slotsFilled, filters) {
  const el = document.getElementById("kisc-slots");
  const chips = [];
  Object.entries(slotsFilled || {}).forEach(([slot, val]) => chips.push(`${slot}: ${val}`));
  (filters?.colors || []).forEach(c => chips.push(`color: ${c}`));
  (filters?.objects || []).forEach(o => chips.push(`object: ${o}`));
  (filters?.categories || []).forEach(c => chips.push(`category: ${c}`));
  el.innerHTML = chips.length
    ? "Filters so far: " + chips.map(c => `<span class="chip">${escapeHtml(c)}</span>`).join("")
    : "";
}

function kiscRenderGrid(results, poolSize, clarifying) {
  const grid  = document.getElementById("kisc-grid");
  const meta  = document.getElementById("kisc-meta");
  const empty = document.getElementById("kisc-empty");
  resultImages.prune(grid);
  grid.innerHTML = "";

  if (!results.length) {
    meta.textContent = "";
    empty.style.display = "block";
    empty.textContent = "No frames matched yet — try adding more detail.";
    return;
  }
  empty.style.display = "none";
  meta.textContent = clarifying
    ? `${poolSize} frame(s) still in play — preview below, answer the question to narrow further.`
    : `${results.length} of ${poolSize} frame(s) shown.`;

  const maxScore = Math.max(...results.map(r => r.score), 0.01);
  results.forEach(item => {
    const card = document.createElement("div");
    card.className = "card";
    if (isShortlisted(item.file)) card.classList.add("selected");
    const pct = Math.round((item.score / maxScore) * 100);
    card.innerHTML = `
      <div class="card-img-wrap">
        <img src="${API}/image/${encodeURIComponent(item.file)}" loading="lazy" alt="${escapeHtml(item.file)}">
        <div class="select-badge" title="Add to answer shortlist">✓</div>
      </div>
      <div class="card-info">
        <div class="card-filename" title="${escapeHtml(item.file)}">${escapeHtml(item.file)}</div>
        <div class="card-score-bar"><div class="card-score-fill" style="width:${pct}%"></div></div>
        <div class="card-tags">
          ${(item.objects || []).slice(0, 2).map(o => `<span class="tag">${escapeHtml(o)}</span>`).join("")}
          ${(item.colors  || []).slice(0, 2).map(c => `<span class="tag" style="background:#fdf2e9;color:#e67e22">${escapeHtml(c)}</span>`).join("")}
        </div>
      </div>`;
    card.querySelector(".select-badge").onclick = (e) => { e.stopPropagation(); toggleShortlist(item.file); };
    const openIt = () => kiscOpenFrame(item);
    card.querySelector(".card-info").onclick = openIt;
    card.querySelector("img").onclick = openIt;
    makeKeyboardAction(card.querySelector(".card-info"), openIt, `Open ${item.file}`);
    makeKeyboardAction(card.querySelector(".select-badge"), () => toggleShortlist(item.file), `Toggle answer selection for ${item.file}`);
    card.querySelectorAll("img").forEach(setupResultImage);
    grid.appendChild(card);
  });
}

// KIS turn results don't carry detections/ocr_texts (kept light for fast
// turns) — fetch full metadata on demand so the shared lightbox still works.
async function kiscOpenFrame(item) {
  try {
    const meta = await fetch(`${API}/frame/meta/${encodeURIComponent(item.file)}`).then(r => r.json());
    openLightbox({ ...item, detections: meta.detections || [], ocr_texts: meta.ocr_texts || [] });
  } catch {
    openLightbox({ ...item, detections: [], ocr_texts: [] });
  }
}

["inp-kisc"].forEach(id =>
  document.getElementById(id).addEventListener("keydown", e => {
    if (e.key === "Enter") { e.preventDefault(); kiscSend(); }
  })
);

["inp-match-desc"].forEach(id =>
  document.getElementById(id).addEventListener("keydown", e => { if (e.key === "Enter") doVideoMatch(); })
);

// ── Answer Mode ───────────────────────────────────────────────────────────────
let answerMode    = false;
let answerCursor  = 0;   // which result we're currently viewing

function toggleAnswerMode() {
  answerMode = !answerMode;
  document.getElementById("btn-answer-toggle").classList.toggle("active", answerMode);
  document.getElementById("btn-answer-toggle").setAttribute("aria-pressed", String(answerMode));
  document.getElementById("answer-mode-banner").classList.toggle("active", answerMode);
  document.getElementById("main").style.display = answerMode ? "none" : "flex";
  if (answerMode && lastResults.length) {
    answerCursor = 0;
    showAnswer(lastResults[answerCursor]);
  }
}

function answerNav(dir) {
  if (!lastResults.length) return;
  answerCursor = Math.max(0, Math.min(lastResults.length - 1, answerCursor + dir));
  showAnswer(lastResults[answerCursor]);
}

// Keyboard arrow navigation in answer mode
document.addEventListener("keydown", e => {
  if (!answerMode || e.target.closest("input, textarea, select, button, [contenteditable], [role=tab]")) return;
  if (e.key === "ArrowRight" || e.key === "ArrowDown") { e.preventDefault(); answerNav(1); }
  if (e.key === "ArrowLeft"  || e.key === "ArrowUp")   { e.preventDefault(); answerNav(-1); }
  if (e.key === "Enter") { e.preventDefault(); answerSelect(); }
});

function answerSelect() {
  if (!lastResults.length) return;
  const item = lastResults[answerCursor];
  toggleShortlist(item.file);
  // Flash the button to confirm
  const btn = document.getElementById("btn-answer-select");
  btn.textContent = "✓ Added!";
  btn.style.background = "#2ecc71";
  setTimeout(() => { btn.textContent = "✓ Select as Answer"; btn.style.background = "#27ae60"; }, 800);
}

async function showAnswer(item) {
  if (!item) return;
  document.getElementById("answer-frame").src = `${API}/image/${encodeURIComponent(item.file)}`;
  document.getElementById("answer-score").textContent = `Score: ${(item.score*100).toFixed(1)}%`;
  document.getElementById("answer-nav-label").textContent = `${answerCursor + 1} / ${lastResults.length}`;

  // Show raw file path while resolving
  document.getElementById("answer-filename").textContent = item.file;

  // Resolve to video + frame_idx via CSV map
  const base = fileToAnswer(item.file);
  try {
    const d = await fetch(`${API}/frame-idx?video=${encodeURIComponent(base.video)}&n=${base.frame}`).then(r => r.json());
    document.getElementById("answer-filename").textContent = `${base.video}, ${d.frame_idx}`;
    document.getElementById("answer-score").textContent =
      `Score: ${(item.score*100).toFixed(1)}%  ·  ${d.mapped ? "✓ mapped" : "⚠ raw (no CSV map)"}`;
  } catch {
    document.getElementById("answer-filename").textContent = `${base.video}, ${base.frame}`;
  }

  // Update select button state to reflect whether this frame is already shortlisted
  const btn = document.getElementById("btn-answer-select");
  if (isShortlisted(item.file)) {
    btn.textContent = "✕ Remove from Answer";
    btn.style.background = "#c0392b";
  } else {
    btn.textContent = "✓ Select as Answer";
    btn.style.background = "#27ae60";
  }
}

// ── On-demand indexing ────────────────────────────────────────────────────────
let _indexPollTimer = null;

async function runIndexing() {
  try {
    const d = await fetch(`${API}/index/run`, { method: "POST" }).then(r=>r.json());
    if (d.status === "already_running") {
      document.getElementById("index-status-text").textContent = "Indexing already in progress…";
    } else {
      document.getElementById("index-status-text").textContent = "Indexing started…";
    }
    pollIndexProgress();
  } catch(e) {
    document.getElementById("index-status-text").textContent = "Failed to start: " + e;
  }
}

async function pollIndexProgress() {
  if (_indexPollTimer) clearInterval(_indexPollTimer);
  const update = async () => {
    try {
      const d = await fetch(`${API}/index/progress`).then(r=>r.json());
      const row  = document.getElementById("index-progress-row");
      const fill = document.getElementById("index-progress-fill");
      const text = document.getElementById("index-progress-text");
      const statusText = document.getElementById("index-status-text");
      const pill = document.getElementById("index-pill");

      if (d.running) {
        row.style.display = "flex";
        pill.style.display = "block";
        const pct = d.total > 0 ? Math.round((d.done/d.total)*100) : 0;
        fill.style.width = pct + "%";
        const eta = d.eta_sec != null ? ` · ETA ${Math.round(d.eta_sec)}s` : "";
        text.textContent = `${d.done} / ${d.total} (${pct}%)${eta}`;
        pill.textContent = `⚡ Indexing ${pct}%`;
        statusText.textContent = `Stage: ${d.stage}`;
      } else {
        row.style.display = "none";
        pill.style.display = "none";
        if (d.error) {
          statusText.textContent = "Error: " + d.error;
        } else {
          statusText.textContent = `Idle — ${d.collection_count} frames in index.`;
          if (_indexPollTimer) { clearInterval(_indexPollTimer); _indexPollTimer = null; }
          fetchStatus(); // refresh sidebar counts
        }
      }
    } catch {}
  };
  update();
  _indexPollTimer = setInterval(update, 1500);
}

// ── System health ─────────────────────────────────────────────────────────────
async function refreshHealth() {
  const grid = document.getElementById("health-grid");
  grid.innerHTML = `<div class="health-item">Loading…</div>`;
  try {
    const [d, s] = await Promise.all([
      fetch(`${API}/health`).then(r=>r.json()),
      fetch(`${API}/status`).then(r=>r.json()),
    ]);
    const items = [
      { label: "Frames", value: d.frames, ok: true },
      { label: "GPU", value: d.gpu ? "✓ " + (d.gpu_name||"") : "✗ CPU only", ok: d.gpu },
      { label: "YOLO cfg", value: d.profile ? `${d.profile.yolo_model} @${d.profile.yolo_imgsz}` : "—", ok: true },
      { label: "CLIP cfg", value: d.profile ? `${d.profile.clip_model.split('/').pop()} · b${d.profile.clip_batch_size}` : "—", ok: true },
      { label: "FAISS cfg", value: d.profile ? `${String(d.profile.faiss_index_type || "—").toUpperCase()} · ef${d.profile.faiss_ef_search ?? "—"}` : "—", ok: true },
      { label: "ASR cfg", value: d.profile ? d.profile.whisper_model : "—", ok: true },
      { label: "OCR engine", value: d.profile ? `${d.profile.ocr_engine || "—"} · b${d.profile.ocr_batch_size ?? "—"}` : "—", ok: true },
      { label: "OCR GPU", value: d.profile ? (d.profile.ocr_gpu ? "Enabled" : "Disabled") : "—", ok: d.profile ? !!d.profile.ocr_gpu : true },
      { label: "OCR all", value: d.profile ? (d.profile.ocr_all_frames ? "All frames" : "Text candidates") : "—", ok: true },
      { label: "Search mode", value: d.profile ? (d.profile.default_search_mode || "hybrid") : "—", ok: true },
      { label: "OCR ASCII", value: s.ocr_ascii_count!=null ? `${s.ocr_ascii_count} indexed` : "Enabled", ok: true },
      { label: "YOLO loaded", value: d.models.yolo ? "Loaded" : "Not loaded", ok: true },
      { label: "CLIP loaded", value: d.models.clip_text ? "Text encoder ready" : "Not loaded", ok: !!d.models.clip_text },
      { label: "OCR loaded",  value: d.models.ocr  ? "Loaded" : "Not loaded", ok: true },
      { label: "Whisper loaded", value: d.models.whisper ? "Loaded" : "Not loaded", ok: true },
      { label: "Text candidates", value: s.has_text_count ?? "—", ok: true },
      { label: "OCR cached", value: s.ocr_cached_count!=null ? `${s.ocr_cached_count}/${s.has_text_count}` : "—", ok: true },
    ];
    grid.innerHTML = items.map(it => `
      <div class="health-item ${it.ok ? 'ok' : 'bad'}">
        <div style="font-size:10px;color:#999;margin-bottom:2px;">${it.label}</div>
        <div>${it.value}</div>
      </div>`).join("");
  } catch {
    grid.innerHTML = `<div class="health-item bad">Backend offline</div>`;
  }
}

// ── Backup ────────────────────────────────────────────────────────────────────
async function createBackup() {
  const status = document.getElementById("backup-status");
  status.textContent = "Creating backup… this may take a moment for large indexes.";
  try {
    const d = await fetch(`${API}/backup`, { method: "POST" }).then(r=>r.json());
    status.textContent = `✓ Backup created: ${d.size_mb} MB`;
    refreshBackupList();
  } catch(e) {
    status.textContent = "Backup failed: " + e;
  }
}

async function refreshBackupList() {
  const list = document.getElementById("backup-list");
  try {
    const d = await fetch(`${API}/backup/list`).then(r=>r.json());
    if (!d.backups.length) { list.innerHTML = `<div style="font-size:12px;color:#aaa;">No backups yet.</div>`; return; }
    list.innerHTML = d.backups.map(b => `
      <div class="backup-row">
        <span style="flex:1;">${b.name}</span>
        <span style="color:#888;">${b.size_mb} MB</span>
        <a href="${API}/backup/download/${b.name}" download style="color:#4a90e2;text-decoration:none;">⬇ Download</a>
      </div>`).join("");
  } catch {
    list.innerHTML = `<div style="font-size:12px;color:#e74c3c;">Failed to load backups.</div>`;
  }
}

// Poll index progress on initial load too, in case indexing is still running from startup
loadAsrConfig();
pollIndexProgress();
