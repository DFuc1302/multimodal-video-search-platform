/* Independent OCR search: it never cancels or replaces a Visual request. */
(() => {
  const PAGE_SIZE = 1000;
  let operator = 'and';
  let controller = null;
  let serial = 0;
  let page = 1;
  let total = 0;

  try { operator = localStorage.getItem('frame-search.ocr-operator.v1') === 'or' ? 'or' : 'and'; } catch (_) {}

  window.setOcrOperator = (value, runSearch = true) => {
    operator = value === 'or' ? 'or' : 'and';
    try { localStorage.setItem('frame-search.ocr-operator.v1', operator); } catch (_) {}
    document.querySelectorAll('.ocr-op').forEach(button => {
      const active = button.dataset.operator === operator;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    if (runSearch && document.getElementById('inp-ocr').value.includes(',')) searchOcr(1);
  };

  function setBusy(busy) {
    document.getElementById('btn-ocr-search').disabled = busy || !textReady;
    document.getElementById('ocr-spinner').hidden = !busy;
    document.getElementById('ocr-results-wrap').setAttribute('aria-busy', String(busy));
  }

  function render(data, requestedPage) {
    page = data.page || requestedPage;
    total = data.total || 0;
    const results = data.results || [];
    document.getElementById('ocr-meta').textContent = `${results.length} / ${total} frames · Trang ${page}`;
    document.getElementById('ocr-empty').hidden = results.length > 0;
    const grid = document.getElementById('ocr-grid');
    grid.replaceChildren();
    for (const item of results) {
      const card = document.createElement('article');
      card.className = 'ocr-result-card';
      const texts = (item.ocr_texts || []).map(value => typeof value === 'string' ? value : value.text).filter(Boolean).join(' · ');
      card.innerHTML = `<img src="${API}/image/${encodeURIComponent(item.file)}" loading="lazy" alt="${escapeHtml(item.file)}">
        <div class="ocr-result-info"><strong title="${escapeHtml(item.file)}">${escapeHtml(item.file)}</strong>
        <div class="ocr-snippet">${highlightPhrase(texts || 'OCR match', document.getElementById('inp-ocr').value)}</div></div>`;
      card.onclick = () => openAsrFrame(item.file);
      grid.appendChild(card);
    }
    const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    const pagination = document.getElementById('ocr-pagination');
    pagination.hidden = total <= PAGE_SIZE;
    document.getElementById('ocr-page-label').textContent = `Page ${page} / ${pages}`;
    document.getElementById('btn-ocr-prev').disabled = page <= 1;
    document.getElementById('btn-ocr-next').disabled = page >= pages;
  }

  async function searchOcr(targetPage = 1) {
    if (!textReady) return;
    const query = document.getElementById('inp-ocr').value.trim();
    if (!query) { document.getElementById('ocr-error').hidden = false; document.getElementById('ocr-error').textContent = 'Nhập chữ cần tìm.'; return; }
    const current = ++serial;
    controller?.abort();
    controller = new AbortController();
    document.getElementById('ocr-error').hidden = true;
    setBusy(true);
    const params = new URLSearchParams({ocr_query:query, ocr_operator:operator, page:targetPage, page_size:PAGE_SIZE});
    const video = document.getElementById('inp-ocr-video').value.trim();
    if (video) params.set('video', video);
    try {
      const response = await trackedSearch(`${API}/search?${params}`, {signal:controller.signal});
      const data = await readJsonResponse(response);
      if (current === serial) render(data, targetPage);
    } catch (error) {
      if (current === serial && error.name !== 'AbortError') {
        const node = document.getElementById('ocr-error'); node.hidden = false; node.textContent = error.message;
      }
    } finally { if (current === serial) setBusy(false); }
  }

  window.searchOcr = searchOcr;
  document.getElementById('btn-ocr-search').onclick = () => searchOcr(1);
  document.getElementById('btn-ocr-prev').onclick = () => searchOcr(page - 1);
  document.getElementById('btn-ocr-next').onclick = () => searchOcr(page + 1);
  ['inp-ocr','inp-ocr-video'].forEach(id => document.getElementById(id).addEventListener('keydown', event => { if (event.key === 'Enter') searchOcr(1); }));
  const enable = () => { document.getElementById('btn-ocr-search').disabled = false; document.getElementById('btn-ocr-search').textContent = 'Search OCR'; };
  window.addEventListener('aic-text-ready', enable);
  if (textReady) enable();
  window.setOcrOperator(operator, false);
})();
