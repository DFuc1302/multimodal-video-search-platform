const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(__dirname + '/static/app.js', 'utf8');
const start = source.indexOf('async function waitForSearchReady()');
const code = source.slice(start, source.indexOf('waitForSearchReady();', start));
async function check(status) {
  const elements = new Map();
  const messages = [];
  const states = [{ready:false,text_ready:true,status}, {ready:true,frames:123}];
  let events = 0;
  const context = vm.createContext({API:'',searchReady:false,visualReady:false,textReady:false,
    document:{getElementById(id) {
      if (!elements.has(id)) elements.set(id, {disabled:true, set textContent(text) { messages.push([id,text]); }});
      return elements.get(id);
    }},
    window:{dispatchEvent(){events++;}}, Event:class{},
    fetch:async()=> { const value=states.shift(); return {ok:value.ready,json:async()=>value}; },
    fetchStatus:async()=>{}, setTimeout(callback) {
      assert.equal(context.searchReady,false);
      assert.equal(context.textReady,true);
      assert.equal(context.visualReady,false);
      assert.equal(elements.get('btn-search').disabled,true);
      callback();
    }});
  vm.runInContext(code,context);
  await context.waitForSearchReady();
  assert.equal(context.visualReady,true);
  assert.equal(events,2);
  assert(messages.some(([id,text])=>id==='competition-startup' && text.includes('OCR/ASR sẵn sàng')));
}
(async()=>{await check('loading_visual');await check('failed');console.log('PASS: UI enables OCR during CLIP loading/failure and switches to full search');})().catch(e=>{console.error(e);process.exitCode=1;});
