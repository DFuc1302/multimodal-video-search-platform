// Shared browser controls. Loaded before app.js; no framework or build step required.
function updateOperatorControl(operator, storageKey, selector, inputId, example) {
  try { localStorage.setItem(storageKey, operator); } catch {}
  document.querySelectorAll(selector).forEach(button => {
    const active = button.dataset.operator === operator;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  const input = document.getElementById(inputId);
  input.placeholder = `e.g. ${example} (comma = ${operator === 'and' ? 'ALL' : 'ANY'})`;
  return input;
}

async function readJsonResponse(response) {
  let data;
  try { data = await response.json(); }
  catch { throw new Error(`The server returned an unreadable response (HTTP ${response.status}). Please try again.`); }
  if (!response.ok || data.error) {
    const detail = data.error || data.detail;
    throw new Error(typeof detail === 'string' ? detail : `Search failed (HTTP ${response.status}). Please try again.`);
  }
  return data;
}

function clearSearchError() {
  document.getElementById('search-error').hidden = true;
}

function showSearchError(message) {
  const banner = document.getElementById('search-error');
  banner.querySelector('span').textContent = message;
  banner.hidden = false;
  document.getElementById('results-wrap').scrollTop = 0;
}

function makeKeyboardAction(element, action, label) {
  element.tabIndex = 0;
  element.setAttribute('role', 'button');
  element.setAttribute('aria-label', label);
  element.addEventListener('keydown', event => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    event.stopPropagation();
    action();
  });
}

function toggleFilters() {
  const expanded = document.getElementById('main').classList.toggle('filters-open');
  document.getElementById('btn-filters').setAttribute('aria-expanded', String(expanded));
}

const dialogFocus = (() => {
  const stack = [];
  const focusable = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';
  document.addEventListener('keydown', event => {
    const current = stack.at(-1);
    if (!current || event.key !== 'Tab') return;
    const controls = [...current.element.querySelectorAll(focusable)]
      .filter(node => !node.disabled && node.getClientRects().length);
    const first = controls[0], last = controls.at(-1);
    if (!first) { event.preventDefault(); return; }
    if (!current.element.contains(document.activeElement) ||
        (event.shiftKey && document.activeElement === first) ||
        (!event.shiftKey && document.activeElement === last)) {
      event.preventDefault();
      (event.shiftKey ? last : first).focus();
    }
  });
  return {
    open(element, initialFocus) {
      if (stack.some(entry => entry.element === element)) return;
      stack.push({element, previous: document.activeElement});
      initialFocus.focus();
    },
    close(element) {
      const index = stack.findIndex(entry => entry.element === element);
      if (index < 0) return;
      const wasTop = index === stack.length - 1;
      const [entry] = stack.splice(index, 1);
      if (wasTop && entry.previous?.isConnected) entry.previous.focus();
    },
  };
})();

document.getElementById('tabs').addEventListener('keydown', event => {
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const index = tabs.indexOf(event.target);
  if (index < 0) return;
  let next;
  if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
  else if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
  else if (event.key === 'Home') next = 0;
  else if (event.key === 'End') next = tabs.length - 1;
  else return;
  event.preventDefault();
  tabs[next].focus();
  switchTab(tabs[next].dataset.page);
});
