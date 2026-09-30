const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(__dirname + '/index.html', 'utf8');
assert.match(html, /id="tab-ocr"[^>]+data-page="ocr"/);
assert.match(html, /id="page-ocr"/);
assert.equal((html.match(/id="inp-ocr"/g) || []).length, 1);

const values = { 'inp-ocr':'airport', 'inp-ocr-video':'', 'btn-ocr-search':'',
  'ocr-spinner':'', 'ocr-results-wrap':'', 'ocr-error':'', 'ocr-meta':'', 'ocr-empty':'',
  'ocr-grid':'', 'ocr-pagination':'', 'ocr-page-label':'', 'btn-ocr-prev':'', 'btn-ocr-next':'' };
const elements = Object.fromEntries(Object.entries(values).map(([id,value]) => [id, {
  value, hidden:false, disabled:false, dataset:{operator:id.endsWith('prev')?'and':'or'},
  classList:{toggle(){}}, setAttribute(){}, addEventListener(){}, replaceChildren(){}, appendChild(){},
}]));
const pending = [];
const context = vm.createContext({window:{addEventListener(){}, setOcrOperator:null}, document:{
  getElementById:id=>elements[id], querySelectorAll:()=>[], createElement:()=>({className:'',onclick:null,innerHTML:'',appendChild(){}}),
}, localStorage:{getItem(){return null},setItem(){}}, URLSearchParams, AbortController,
  textReady:true, API:'', escapeHtml:v=>v, highlightPhrase:v=>v, openAsrFrame(){},
  readJsonResponse:async response=>response.json(),
  trackedSearch(url,options){return new Promise(resolve=>pending.push({url,signal:options.signal,resolve}));}
});
vm.runInContext(fs.readFileSync(__dirname + '/static/ocr.js','utf8'), context);
const first=context.window.searchOcr(1), second=context.window.searchOcr(1);
assert.equal(pending[0].signal.aborted,true);
assert.match(pending[1].url,/ocr_query=airport/);
pending[1].resolve({json:async()=>({results:[],total:0,page:1})});
second.then(()=>{pending[0].resolve({json:async()=>({results:[],total:0,page:1})});return first;})
  .then(()=>console.log('PASS: OCR tab uses its own cancel-safe request lifecycle'))
  .catch(error=>{console.error(error);process.exitCode=1;});
