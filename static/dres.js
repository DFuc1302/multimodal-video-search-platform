/* DRES submission is explicit; changing any answer invalidates its preview. */
(() => {
  const el = id => document.getElementById(`dres-${id}`);
  const state = {connection: '', preview: null, revision: 0, busy: false, simulations: []};
  try { state.connection = sessionStorage.getItem('aic.dres.connection') || ''; } catch (_) {}
  const help = {
    textual_kis: 'Chọn một frame đại diện cho sự kiện. Hệ thống đổi sang mili giây.',
    video_kis: 'Mô tả video đề bằng quan sát; không chụp hoặc ghi lại video đề để đưa vào công cụ.',
    qa: 'Chọn một frame minh chứng và nhập câu trả lời. Thời gian được đổi sang mili giây.',
    trake: 'Nhập một frame cho mỗi sự kiện, trong cùng video, theo thứ tự tăng dần. Ví dụ: 90, 261, 351.',
  };
  const labels = {simulated: 'Mô phỏng', received: 'DRES đã nhận', rejected: 'Bị từ chối', unknown: 'Chưa rõ kết quả', sending: 'Đang gửi / chưa rõ kết quả'};
  function message(id, text, error = false) {
    el(id).textContent = text;
    el(id).dataset.error = String(error);
  }
  function invalidate() {
    state.revision++;
    state.preview = null;
    el('submit').disabled = true;
    el('copy').disabled = true;
    el('preview-mode').textContent = 'Cần kiểm tra đáp án';
    el('preview-details').textContent = 'Nhấn “Kiểm tra đáp án” để xác minh thông tin mới.';
    el('payload').textContent = 'Chưa có đáp án';
    message('status', '');
    message('result', '');
  }
  function setBusy(value) {
    state.busy = value;
    ['preview', 'login', 'disconnect', 'refresh', 'current', 'import', 'live', 'evaluation', 'history-refresh', 'host', 'user', 'password'].forEach(id => el(id).disabled = value);
    el('answer-fields').disabled = value;
    el('submit').disabled = value || !state.preview;
  }
  async function api(path, body) {
    const response = await fetch(`${API}/dres/${path}`, {
      method: body === undefined ? 'GET' : 'POST',
      headers: {'Content-Type': 'application/json', 'X-DRES-Connection': state.connection},
      ...(body === undefined ? {} : {body: JSON.stringify(body)}),
    });
    let data;
    try { data = await response.json(); } catch (_) { throw new Error('Không đọc được phản hồi server. Nếu vừa nộp thật, kiểm tra lịch sử trước khi gửi lại.'); }
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Server từ chối yêu cầu.');
    return data;
  }
  function updateMode() {
    invalidate();
    const live = el('live').checked;
    el('mode-badge').textContent = live ? 'NỘP THẬT' : 'MÔ PHỎNG';
    el('mode-badge').classList.toggle('live', live);
    el('submit').textContent = live ? 'Nộp đáp án lên DRES' : 'Mô phỏng nộp';
  }
  function updateType() {
    invalidate();
    const type = el('type').value;
    el('type-help').textContent = help[type];
    el('answer-label').hidden = type !== 'qa';
    el('frames-label').textContent = type === 'trake' ? 'Frame ID cho từng sự kiện, cách nhau bằng dấu phẩy' : 'Frame ID gốc';
    el('frames').placeholder = type === 'trake' ? '90, 261, 351' : 'Ví dụ: 261';
  }
  async function evaluations() {
    const data = await api('evaluations');
    const previous = el('evaluation').value;
    el('evaluation').replaceChildren(new Option('Chọn phiên đang ACTIVE', ''));
    const active = data.evaluations.filter(item => item.status === 'ACTIVE');
    for (const item of data.evaluations) {
      const option = new Option(`${item.name} · ${item.status}`, item.id);
      option.disabled = item.status !== 'ACTIVE';
      el('evaluation').add(option);
    }
    if (active.some(item => item.id === previous)) el('evaluation').value = previous;
    else if (active.length === 1) el('evaluation').value = active[0].id;
    invalidate();
    message('connection-status', active.length ? `Đã kết nối · ${active.length} phiên ACTIVE.` : 'Đã kết nối. Chưa có phiên ACTIVE; vẫn có thể mô phỏng.');
  }
  async function history() {
    let remote = [];
    if (state.connection) remote = (await api('history')).history;
    const entries = [...state.simulations, ...remote].sort((a, b) => b.created - a.created);
    if (!entries.length) { el('history').textContent = 'Chưa có lượt nộp.'; return; }
    el('history').innerHTML = `<div class="dres-table-wrap"><table><thead><tr><th>Thời gian</th><th>Dạng / câu</th><th>Đáp án</th><th>Trạng thái / phản hồi</th></tr></thead><tbody>${entries.map(row => {
      let task = row.task;
      try { task = JSON.parse(task).name; } catch (_) {}
      return `<tr><td>${escapeHtml(new Date(row.created * 1000).toLocaleTimeString())}</td><td>${escapeHtml(row.mode)}<br>${escapeHtml(task || '')}</td><td>${escapeHtml(JSON.stringify(row.payload))}</td><td><strong>${escapeHtml(labels[row.state] || row.state)}</strong><br>${escapeHtml(JSON.stringify(row.response || {}))}</td></tr>`;
    }).join('')}</tbody></table></div>`;
  }
  async function run(action, target = 'status') {
    if (state.busy) return;
    setBusy(true);
    try { await action(); } catch (error) { message(target, error.message, true); }
    finally { setBusy(false); }
  }
  el('login-form').addEventListener('submit', event => {
    event.preventDefault();
    run(async () => {
      invalidate();
      message('connection-status', 'Đang đăng nhập…');
      const body = {base_url: el('host').value, username: el('user').value, password: el('password').value};
      el('password').value = '';
      const result = await api('login', body);
      state.connection = result.connection;
      try { sessionStorage.setItem('aic.dres.connection', state.connection); } catch (_) {}
      await evaluations();
      await history();
    }, 'connection-status');
  });
  el('disconnect').addEventListener('click', () => run(async () => {
    await api('disconnect', {});
    state.connection = '';
    try { sessionStorage.removeItem('aic.dres.connection'); } catch (_) {}
    el('live').checked = false;
    updateMode();
    el('evaluation').replaceChildren(new Option('Chưa có phiên thi', ''));
    message('connection-status', 'Đã ngắt kết nối.');
    await history();
  }));
  el('refresh').addEventListener('click', () => run(evaluations, 'connection-status'));
  el('current').addEventListener('click', () => run(async () => {
    const task = await api(`current?evaluation_id=${encodeURIComponent(el('evaluation').value)}`);
    message('current-task', `${task.name} · ${task.taskType} · ${task.taskGroup}`);
  }));
  el('type').addEventListener('change', updateType);
  el('live').addEventListener('change', updateMode);
  ['video', 'frames', 'answer', 'evaluation', 'host', 'user'].forEach(id => el(id).addEventListener('input', invalidate));
  el('evaluation').addEventListener('change', invalidate);
  el('import').addEventListener('click', () => run(async () => {
    if (!shortlist.length) throw new Error('Chọn frame ở Frame Search trước.');
    if (shortlist.some(row => row.mapped !== true || !Number.isInteger(row.frameIdx))) throw new Error('Có frame chưa ánh xạ chính xác. Không thể chuyển sang đáp án DRES.');
    if (new Set(shortlist.map(row => row.video)).size !== 1) throw new Error('Chọn các frame của cùng một video.');
    if (el('type').value !== 'trake' && shortlist.length !== 1) throw new Error('KIS và Q&A cần chọn đúng một frame.');
    el('video').value = shortlist[0].video;
    el('frames').value = shortlist.map(row => row.frameIdx).join(', ');
    if (el('type').value === 'qa') el('answer').value = document.getElementById('inp-vqa-answer')?.value || el('answer').value;
    invalidate();
    message('status', 'Đã lấy frame. Kiểm tra thứ tự sự kiện rồi nhấn “Kiểm tra đáp án”.');
  }));
  el('preview').addEventListener('click', () => run(async () => {
    invalidate();
    const revision = state.revision;
    const raw = el('frames').value.trim();
    if (!/^\d+(?:\s*[,\s]\s*\d+)*$/.test(raw)) throw new Error('Nhập Frame ID nguyên không âm; phân tách bằng dấu phẩy hoặc khoảng trắng.');
    const frames = raw.split(/[,\s]+/).map(Number);
    if (frames.some(frame => !Number.isSafeInteger(frame))) throw new Error('Frame ID quá lớn.');
    message('status', 'Đang xác minh frame và thời gian…');
    const data = await api('preview', {task_type: el('type').value, video: el('video').value,
      frames, answer: el('answer').value, live: el('live').checked, evaluation_id: el('evaluation').value});
    if (revision !== state.revision) return;
    state.preview = data;
    el('preview-mode').textContent = `${data.live ? 'NỘP THẬT' : 'MÔ PHỎNG'} · hiệu lực 2 phút`;
    const rows = data.mapping.frames;
    el('preview-details').innerHTML = `<p><strong>${escapeHtml(data.mapping.video_id)}</strong>${data.task ? ` · Câu: ${escapeHtml(data.task.name)} · ${escapeHtml(data.task.taskType)}` : ''}</p><div class="dres-table-wrap"><table><thead><tr><th>Sự kiện</th><th>Frame ID gốc</th><th>Thứ tự ảnh n</th><th>Giây (PTS)</th><th>Mili giây</th></tr></thead><tbody>${rows.map((row, i) => `<tr><td>${i + 1}</td><td>${row.frame_idx}</td><td>${row.n}</td><td>${row.pts_time}</td><td>${row.time_ms}</td></tr>`).join('')}</tbody></table></div>`;
    el('payload').textContent = JSON.stringify(data.payload, null, 2);
    el('copy').disabled = false;
    message('status', ['Đã xác minh các frame được chọn bằng CSV.', ...data.warnings,
      ...(data.mapping.map_warnings.length ? ['Video có dòng trùng ở vị trí khác; các frame đang chọn không có thời gian mâu thuẫn.'] : [])].join('\n'));
  }));
  el('submit').addEventListener('click', () => run(async () => {
    if (!state.preview) throw new Error('Kiểm tra đáp án trước.');
    const preview = state.preview;
    state.preview = null;
    el('copy').disabled = true;
    message('result', preview.live ? 'Đang gửi một lần lên DRES…' : 'Đang mô phỏng…');
    const result = await api('submit', {preview_id: preview.preview_id});
    message('result', `${result.message || labels[result.state]}${result.response ? '\n' + JSON.stringify(result.response) : ''}`, ['unknown', 'rejected'].includes(result.state));
    if (result.state === 'simulated') state.simulations.unshift({created: Date.now() / 1000, mode: preview.task_type, state: 'simulated', payload: result.payload});
    el('preview-mode').textContent = labels[result.state] || result.state;
    await history();
  }, 'result'));
  el('copy').addEventListener('click', () => run(async () => {
    if (!state.preview) return;
    await navigator.clipboard.writeText(JSON.stringify(state.preview.payload, null, 2));
    message('status', 'Đã sao chép JSON.');
  }));
  el('history-refresh').addEventListener('click', () => run(history));
  if (state.connection) message('connection-status', 'Có kết nối từ tab này. Nhấn “Tải lại phiên thi” để kiểm tra.');
  if (location.hash === '#dres') switchTab('dres');
})();
