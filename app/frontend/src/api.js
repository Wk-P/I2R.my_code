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
export const getTagDoc = (tag) => get(`/api/tags/${encodeURIComponent(tag)}/doc`);
export const resultFile = (branch, scenario, algo, run, file) =>
  `/api/results/${scenario}/${algo}/${run}/${file}${q(branch)}`;
export const getResults = (branch) => get(`/api/results${q(branch)}`, []);
