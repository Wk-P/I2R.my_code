async function get(url, fallback = null) {
  try {
    const res = await fetch(url);
    return res.ok ? await res.json() : fallback;
  } catch {
    return fallback;
  }
}

const q = (branch) => (branch ? `?branch=${encodeURIComponent(branch)}` : "");

export const getBranch = () => get("/api/branch", { current: null, branches: [] });
export const getSystem = () => get("/api/system");
export const getBatches = (all = false, branch = null) => {
  const qs = new URLSearchParams();
  if (all) qs.set("all", "true");
  if (branch) qs.set("branch", branch);
  const s = qs.toString();
  return get(`/api/batches${s ? "?" + s : ""}`, { batches: [] });
};
export const getBatch = (name) => get(`/api/batch_progress/${encodeURIComponent(name)}`);
export const getExperiments = (branch) => get(`/api/experiments${q(branch)}`, []);
export const getHistory = (scenario, algo, branch) => get(`/api/history/${scenario}/${algo}${q(branch)}`, []);
export const getTags = () => get("/api/tags", []);
export const getTagDoc = (tag, lang = "zh") => get(`/api/tags/${encodeURIComponent(tag)}/doc?lang=${lang}`);
// Plots can be redrawn in place, and the Cloudflare tunnel rewrites Cache-Control to max-age=14400 for .png,
// so a per-page-load token makes every load fetch the current image (browser and edge cache key on the URL).
const BUST = Date.now();
export const resultFile = (branch, scenario, algo, run, file) =>
  `/api/results/${scenario}/${algo}/${run}/${file}${q(branch)}${branch ? "&" : "?"}v=${BUST}`;
export const getResults = (branch) => get(`/api/results${q(branch)}`, []);
// v4.4.14: experiment reports (paper_contents/v*/report.md)
export const getReports = () => get("/api/reports", { reports: [] });
export const getReport = (version, file = null, lang = "zh") =>
  get(`/api/reports/${encodeURIComponent(version)}?lang=${lang}${file ? "&file=" + encodeURIComponent(file) : ""}`);
// serial experiment queue (read-only; managed by src/scripts/queue_runner.py)
export const getGpuHistory = (since = 0) => get(`/api/gpu_history?since=${since}`, { interval: 2, window: 1800, samples: [] });
export const getCurves = (lang = "zh") => get(`/api/curves?lang=${lang}`, { versions: [] });
export const getQueue = () => get("/api/queue", { items: [], paused: false, runner: "unknown" });
