// Minimal Markdown -> HTML for the experiment reports: headings, bullet lists (with indented
// continuation), tables, `code`, **bold**. Everything is escaped first; no raw HTML passes through.
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const inline = (s) => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
const isRule = (cells) => cells.every((c) => /^:?-+:?$/.test(c));
const cellsOf = (line) => line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
const numeric = (c) => /^[-+−]?[\d.]+%?( ± [\d.+−-]+%?)?$/.test(c) || c === "—";

export function renderMarkdown(text) {
  const lines = (text || "").split("\n");
  const out = [];
  for (let i = 0; i < lines.length; ) {
    const ln = lines[i];
    if (ln.startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].startsWith("|")) {
        const c = cellsOf(lines[i]);
        if (!isRule(c)) rows.push(c);
        i++;
      }
      if (rows.length) {
        const head = rows[0].map((c) => `<th>${inline(c)}</th>`).join("");
        const body = rows.slice(1).map((r) => "<tr>" + r.map((c) => `<td${numeric(c) ? ' class="num"' : ""}>${inline(c)}</td>`).join("") + "</tr>").join("");
        out.push(`<div class="table-wrap"><table class="grid"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`);
      }
      continue;
    }
    if (ln.startsWith("- ")) {
      const items = [];
      while (i < lines.length && (lines[i].startsWith("- ") || (lines[i].startsWith("  ") && items.length))) {
        if (lines[i].startsWith("- ")) items.push(inline(lines[i].slice(2)));
        else items[items.length - 1] += "<br>" + inline(lines[i].trim());
        i++;
      }
      out.push("<ul>" + items.map((x) => `<li>${x}</li>`).join("") + "</ul>");
      continue;
    }
    const h = ln.match(/^(#{1,4}) (.*)/);
    if (h) out.push(`<h${h[1].length + 1}>${inline(h[2])}</h${h[1].length + 1}>`);
    else if (ln.trim()) out.push(`<p>${inline(ln)}</p>`);
    i++;
  }
  return out.join("\n");
}
