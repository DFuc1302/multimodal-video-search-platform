// Run with node test_ui.cjs. No browser, packages, models, or production data required.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
for (const [, src, inline] of html.matchAll(/<script\b(?:[^>]*src="([^"]+)")?[^>]*>([\s\S]*?)<\/script>/gi)) {
  new vm.Script(src ? fs.readFileSync(path.join(__dirname, src), 'utf8') : inline, {filename: src || 'inline'});
}
const context = vm.createContext({});
vm.runInContext(fs.readFileSync(path.join(__dirname, 'static/utils.js'), 'utf8'), context);
assert.equal(context.highlightPhrase('hello mark', 'hello, mark'), '<mark>hello</mark> <mark>mark</mark>');
assert.equal(context.highlightPhrase('<hello> & goodbye', 'hello, amp'), '&lt;<mark>hello</mark>&gt; &amp; goodbye');
assert.equal(context.highlightPhrase('hello', 'hello, hell'), '<mark>hello</mark>');
assert.equal(context.highlightPhrase('a+b', 'a+b'), '<mark>a+b</mark>');
assert.equal(context.escapeHtml(`a'"<b>&`), 'a&#39;&quot;&lt;b&gt;&amp;');

async function checkResponses() {
  const controls = vm.createContext({document: {
    addEventListener() {}, getElementById() { return {addEventListener() {}}; },
  }});
  vm.runInContext(fs.readFileSync(path.join(__dirname, 'static/components.js'), 'utf8'), controls);
  assert.deepEqual(await controls.readJsonResponse({ok:true, json:async()=>({results:[]})}), {results:[]});
  await assert.rejects(controls.readJsonResponse({ok:false, status:503, json:async()=>({detail:'Model unavailable'})}), /Model unavailable/);
  await assert.rejects(controls.readJsonResponse({ok:false, status:500, json:async()=>{throw new Error('html')}}), /unreadable response/);
  await assert.rejects(controls.readJsonResponse({ok:true, status:200, json:async()=>({error:'Invalid query'})}), /Invalid query/);

  // A late response must not overwrite the newer search or its loading state.
  const pending = [], rendered = [], errors = [], loading = [];
  const inputs = Object.fromEntries(['inp-query', 'inp-ocr', 'inp-video', 'summary-mode', 'summary-category', 'inp-rerank', 'inp-timeline', 'rerank-status']
    .map(id => [id, {value: id === 'summary-mode' ? 'visual' : '', checked: false}]));
  const searchContext = vm.createContext({
    URLSearchParams, AbortController, Set, searchReady:true, API:'', FRAME_PAGE_SIZE:1000,
    activeCategories:new Set(), activeObjects:new Set(), activeColors:new Set(),
    activeQueryOperator:'and', activeOcrOperator:'and', activeVideoMode:'include',
    confSlider:{value:'0'}, serverFramePagination:null,
    document:{getElementById:id => inputs[id]},
    clearSearchError() {}, showSearchError:message => errors.push(message),
    setLoading:value => loading.push(value), readJsonResponse:controls.readJsonResponse,
    trackedSearch(url, options) { return new Promise(resolve => pending.push({resolve, signal:options.signal})); },
    renderResults:results => rendered.push(results), updateSidebarCounts() {},
  });
  const app = fs.readFileSync(path.join(__dirname, 'static/app.js'), 'utf8');
  vm.runInContext(app.slice(app.indexOf('let frameRequestSerial'), app.indexOf('document.getElementById("btn-search").onclick')), searchContext);
  const first = searchContext.doSearch();
  const second = searchContext.doSearch();
  assert.equal(pending[0].signal.aborted, true);
  pending[1].resolve({ok:true, json:async()=>({results:['new'], total:1})});
  await second;
  pending[0].resolve({ok:true, json:async()=>({results:['old'], total:1})});
  await first;
  assert.deepEqual(rendered, [['new']]);
  assert.deepEqual(loading, [true, true, false]);
  const failed = searchContext.doSearch();
  pending[2].resolve({ok:false, status:503, json:async()=>({detail:'Unavailable'})});
  await failed;
  assert.deepEqual(errors, ['Unavailable']);
  assert.equal(loading.at(-1), false);
}

function checkImageLifecycle() {
  let observerCount = 0, callback;
  const timers = new Map();
  let nextTimer = 0;
  const observed = new Set();
  const ctx = vm.createContext({
    Map, window: {addEventListener() {}},
    setTimeout(fn) { timers.set(++nextTimer, fn); return nextTimer; },
    clearTimeout(id) { timers.delete(id); },
    IntersectionObserver: class {
      constructor(fn) { observerCount++; callback = fn; }
      observe(img) { observed.add(img); }
      unobserve(img) { observed.delete(img); }
      disconnect() { observed.clear(); }
    },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, 'static/images.js'), 'utf8'), ctx);
  const makeImage = () => ({src:'/image/frame.jpg', isConnected:true, complete:false, naturalWidth:0,
    listeners:new Map(), classList:{add() {}},
    addEventListener(name, fn) { this.listeners.set(name, fn); },
    removeEventListener(name) { this.listeners.delete(name); },
  });
  const images = Array.from({length:1000}, makeImage);
  images.forEach(ctx.setupResultImage);
  assert.equal(observerCount, 1, 'all images share one observer');
  assert.equal(timers.size, 0, 'offscreen images do not start timers');
  callback([{target:images[0], isIntersecting:true}]);
  assert.equal(timers.size, 1);
  images[0].listeners.get('load')();
  assert.equal(timers.size, 0, 'successful loads release timers');
  assert.equal(images[0].listeners.size, 0, 'successful loads release listeners');
  callback([{target:images[1], isIntersecting:true}]);
  ctx.detached = images[1];
  images[1].isConnected = false;
  vm.runInContext('resultImages.prune()', ctx);
  assert.equal(timers.size, 0, 'removed cards cancel timers');
  assert.equal(images[1].listeners.size, 0);
  vm.runInContext('resultImages.dispose()', ctx);
  assert.equal(observed.size, 0);
  assert.ok(images.every(img => img.listeners.size === 0));
}

checkImageLifecycle();
checkResponses().then(() => console.log('PASS: script syntax, escaping/highlighting, HTTP errors, stale search protection, and 1,000-image observer lifecycle'))
  .catch(error => { console.error(error); process.exitCode = 1; });
