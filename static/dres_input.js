/* Pure input editing shared by the UI and regression tests. */
(function (root) {
  function stepFrame(text, cursor, direction, step) {
    if (!Number.isSafeInteger(step) || step < 1 || ![-1, 1].includes(direction)) throw new Error('Bước tăng/giảm phải là số nguyên dương.');
    if (!/^\d+(?:\s*[,\s]\s*\d+)*$/.test(text.trim())) throw new Error('Nhập ID/idx nguyên không âm trước khi dùng −/+.');
    const tokens = [...text.matchAll(/\d+/g)];
    const token = tokens.find(t => t.index <= cursor && cursor <= t.index + t[0].length) || tokens.find(t => t.index > cursor) || tokens[tokens.length - 1];
    const value = Number(token[0]) + direction * step;
    if (!Number.isSafeInteger(value) || value < 0) throw new Error('ID/idx phải là số nguyên không âm, trong giới hạn an toàn.');
    const replacement = String(value);
    return {text: text.slice(0, token.index) + replacement + text.slice(token.index + token[0].length), start: token.index, end: token.index + replacement.length};
  }
  root.DresInput = {stepFrame};
  if (typeof module !== 'undefined') module.exports = {stepFrame};
})(typeof window === 'undefined' ? globalThis : window);
