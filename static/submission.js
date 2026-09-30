/* Standalone DRES submission UI. Credentials never enter the browser. */
(() => {
  const el = id => document.getElementById(`dres-${id}`);
  const state = {connection: '', preview: null, revision: 0, busy: false, simulations: []};
  const help = {
    textual_kis: 'Chọn một frame đại diện cho sự kiện. Hệ thống đổi sang mili giây.',
    video_kis: 'Chọn một frame đại diện; không chụp hoặc ghi lại video đề để đưa vào công cụ.',
    qa: 'Chọn một frame minh chứng và nhập câu trả lời.',
    trake: 'Nhập một frame cho mỗi sự kiện, cùng video, theo thứ tự tăng dần.',
  };
  const labels = {simulated: 'Mô phỏng', received: 'DRES đã nhận', rejected: 'Bị từ chối', unknown: 'Chưa rõ kết quả', sending: 'Đang gửi / chưa rõ kết quả'};
  function message(id, text, error = false) { el(id).textContent = text; el(id).dataset.error = String(error); }
  function invalidate() {
    state.revision += 1; state.preview = null; el('submit').disabled = true; el('copy').disabled = true;
    el('preview-mode').textContent = 'Cần kiểm tra đáp án'; el('preview-details').textContent = 'Nhấn “Kiểm tra đáp án” để xác minh thông tin mới.';
    el('payload').textContent = 'Chưa có đáp án'; message('status', ''); message('result', '');
  }
  function setBusy(value) {
    state.busy = value;
    ['connect', 'refresh', 'current', 'live', 'evaluation', 'history-refresh'].forEach(id => { el(id).disabled = value; });
    el('answer-fields').disabled = value; el('submit').disabled = value || !state.preview;
  }
  async function api(path, body) {
    const response = await fetch(`/dres/${path}`, {method: body === undefined ? 'GET' : 'POST', headers: {'Content-Type': 'application/json', 'X-DRES-Connection': state.connection}, ...(body === undefined ? {} : {body: JSON.stringify(body)})});
    let data; try { data = await response.json(); } catch (_) { throw new Error('Không đọc được phản hồi server. Nếu vừa nộp thật, hãy kiểm tra lịch sử trước.'); }
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Server từ chối yêu cầu.');
    return data;
  }
  async function run(action, target = 'status') { if (state.busy) return; setBusy(true); try { await action(); } catch (error) { message(target, error.message, true); } finally { setBusy(false); } }
  function updateMode() { invalidate(); const live = el('live').checked; el('mode-badge').textContent = live ? 'NỘP THẬT' : 'MÔ PHỎNG'; el('mode-badge').classList.toggle('live', live); el('submit').textContent = live ? 'Nộp đáp án lên DRES' : 'Mô phỏng nộp'; }
  function updateType() { invalidate(); const type = el('type').value; el('type-help').textContent = help[type]; el('answer-label').hidden = type !== 'qa'; el('frames-label').textContent = (el('frame-mode').value === 'n' ? 'ID ảnh (n)' : 'idx gốc (frame_idx)') + (type === 'trake' ? ' cho từng sự kiện, cách nhau bằng dấu phẩy' : ''); el('frames').placeholder = type === 'trake' ? '90, 261, 351' : 'Ví dụ: 261'; }
  async function evaluations() {
    const data = await api('evaluations'); const previous = el('evaluation').value; const active = data.evaluations.filter(item => item.status === 'ACTIVE');
    el('evaluation').replaceChildren(new Option('Chọn phiên đang ACTIVE', ''));
    for (const item of data.evaluations) { const option = new Option(`${item.name} · ${item.status}`, item.id); option.disabled = item.status !== 'ACTIVE'; el('evaluation').add(option); }
    if (active.some(item => item.id === previous)) el('evaluation').value = previous; else if (active.length === 1) el('evaluation').value = active[0].id;
    invalidate(); message('connection-status', active.length ? `Đã kết nối · ${active.length} phiên ACTIVE.` : 'Đã kết nối. Chưa có phiên ACTIVE; vẫn có thể mô phỏng.');
  }
  async function connect() { message('connection-status', 'Đang kết nối DRES…'); const result = await api('auto-login', {}); state.connection = result.connection; await evaluations(); await history(); }
  async function history() {
    let remote = []; if (state.connection) remote = (await api('history')).history;
    const entries = [...state.simulations, ...remote].sort((a, b) => b.created - a.created);
    if (!entries.length) { el('history').textContent = 'Chưa có lượt nộp.'; return; }
    el('history').innerHTML = `<div class="dres-table-wrap"><table><thead><tr><th>Thời gian</th><th>Dạng / câu</th><th>Đáp án</th><th>Trạng thái / phản hồi</th></tr></thead><tbody>${entries.map(row => { let task = row.task; try { task = JSON.parse(task).name; } catch (_) {} return `<tr><td>${escapeHtml(new Date(row.created * 1000).toLocaleTimeString())}</td><td>${escapeHtml(row.mode)}<br>${escapeHtml(task || '')}</td><td>${escapeHtml(JSON.stringify(row.payload))}</td><td><strong>${escapeHtml(labels[row.state] || row.state)}</strong><br>${escapeHtml(JSON.stringify(row.response || {}))}</td></tr>`; }).join('')}</tbody></table></div>`;
  }
  el('connect').addEventListener('click', () => run(connect, 'connection-status'));
  el('refresh').addEventListener('click', () => run(evaluations, 'connection-status'));
  el('current').addEventListener('click', () => run(async () => { const task = await api(`current?evaluation_id=${encodeURIComponent(el('evaluation').value)}`); message('current-task', `${task.name} · ${task.taskType} · ${task.taskGroup}`); }));
  el('type').addEventListener('change', updateType); el('frame-mode').addEventListener('change', updateType); el('live').addEventListener('change', updateMode);
  ['video', 'frames', 'answer', 'evaluation'].forEach(id => el(id).addEventListener('input', invalidate)); el('evaluation').addEventListener('change', invalidate);
  el('preview').addEventListener('click', () => run(async () => {
    invalidate(); const revision = state.revision; const raw = el('frames').value.trim();
    if (!/^\d+(?:\s*[,\s]\s*\d+)*$/.test(raw)) throw new Error('Nhập Frame ID nguyên không âm; phân tách bằng dấu phẩy hoặc khoảng trắng.');
    const frames = raw.split(/[,\s]+/).map(Number); if (frames.some(frame => !Number.isSafeInteger(frame))) throw new Error('Frame ID quá lớn.');
    message('status', 'Đang xác minh frame và thời gian…');
    const data = await api('preview', {task_type: el('type').value, video: el('video').value, frames, frame_mode: el('frame-mode').value, answer: el('answer').value, live: el('live').checked, evaluation_id: el('evaluation').value});
    if (revision !== state.revision) return; state.preview = data; el('preview-mode').textContent = `${data.live ? 'NỘP THẬT' : 'MÔ PHỎNG'} · hiệu lực 2 phút`;
    const rows = data.mapping.frames; el('preview-details').innerHTML = `<p><strong>${escapeHtml(data.mapping.video_id)}</strong>${data.task ? ` · Câu: ${escapeHtml(data.task.name)} · ${escapeHtml(data.task.taskType)}` : ''}</p><div class="dres-table-wrap"><table><thead><tr><th>Sự kiện</th><th>idx của bạn</th><th>Frame ID BTC</th><th>Lệch (ms)</th><th>Ảnh n</th><th>Giây</th><th>Mili giây</th></tr></thead><tbody>${rows.map((row, i) => `<tr><td>${i + 1}</td><td>${row.frame_idx}</td><td>${row.official_frame_idx ?? '—'}</td><td>${row.delta_ms ?? '—'}</td><td>${row.n}</td><td>${row.pts_time}</td><td>${row.time_ms}</td></tr>`).join('')}</tbody></table></div>`;
    el('payload').textContent = JSON.stringify(data.payload, null, 2); el('copy').disabled = false; message('status', ['Đã xác minh frame bằng CSV.', ...data.warnings].join('\n'));
  }));
  el('submit').addEventListener('click', () => run(async () => {
    if (!state.preview) throw new Error('Kiểm tra đáp án trước.'); const preview = state.preview; state.preview = null; el('copy').disabled = true;
    message('result', preview.live ? 'Đang gửi một lần lên DRES…' : 'Đang mô phỏng…'); const result = await api('submit', {preview_id: preview.preview_id});
    message('result', `${result.message || labels[result.state]}${result.response ? '\n' + JSON.stringify(result.response) : ''}`, ['unknown', 'rejected'].includes(result.state));
    if (result.state === 'simulated') state.simulations.unshift({created: Date.now() / 1000, mode: preview.task_type, state: 'simulated', payload: result.payload}); el('preview-mode').textContent = labels[result.state] || result.state; await history();
  }, 'result'));
  el('copy').addEventListener('click', () => run(async () => { if (state.preview) await navigator.clipboard.writeText(JSON.stringify(state.preview.payload, null, 2)); }));
  let frameCursor = 0; ['click', 'keyup', 'select', 'input'].forEach(event => el('frames').addEventListener(event, () => { frameCursor = el('frames').selectionStart || 0; }));
  for (const [id, direction] of [['minus', -1], ['plus', 1]]) el(id).addEventListener('click', () => { try { const result = DresInput.stepFrame(el('frames').value, frameCursor, direction, Number(el('step').value)); el('frames').value = result.text; frameCursor = result.start; invalidate(); el('frames').focus(); el('frames').setSelectionRange(result.start, result.end); } catch (error) { message('status', error.message, true); } });
  el('history-refresh').addEventListener('click', () => run(history));
  run(connect, 'connection-status');
})();
