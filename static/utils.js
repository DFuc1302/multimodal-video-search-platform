function fmtTime(sec) {
  const m=Math.floor(sec/60).toString().padStart(2,"0");
  const s=(sec%60).toFixed(1).padStart(4,"0");
  return `${m}:${s}`;
}
function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function highlightPhrase(text, phrase) {
  text = String(text ?? "");
  const terms = String(phrase ?? "").split(",").map(t => t.trim()).filter(Boolean);
  if (!terms.length) return escapeHtml(text);
  const pattern = terms.sort((a, b) => b.length - a.length)
    .map(term => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
  let res = "", offset = 0;
  for (const match of text.matchAll(new RegExp(pattern, "gi"))) {
    res += escapeHtml(text.slice(offset, match.index)) + "<mark>" + escapeHtml(match[0]) + "</mark>";
    offset = match.index + match[0].length;
  }
  return res + escapeHtml(text.slice(offset));
}

