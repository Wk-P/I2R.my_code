// /api/v2 (app/backend/v2/api.py); every call resolves to `fallback` on failure.
async function get(url, fallback = null) {
  try {
    const res = await fetch(url);
    return res.ok ? await res.json() : fallback;
  } catch {
    return fallback;
  }
}
const e = encodeURIComponent;
const runPath = (r) => `/api/v2/runs/${e(r.space)}/${e(r.scenario)}/${e(r.algo)}/${e(r.exp_id)}`;

export const getMeta = () => get("/api/v2/meta", { stages: [], spaces: [] });
export const getOverview = () => get("/api/v2/overview");
export const getBatches = (stage = "current", tests = false) => get(`/api/v2/batches?stage=${e(stage)}&tests=${tests}`, { batches: [] });
export const getBatch = (name) => get(`/api/v2/batches/${e(name)}`);
export const getReport = (name, file, lang) => get(`/api/v2/batches/${e(name)}/report?file=${e(file)}&lang=${lang}`);
export const getReportData = (name) => get(`/api/v2/batches/${e(name)}/report_data`, { report: null, sweeps: [] });
export const getRun = (r) => get(runPath(r));
export const getTrace = (r, points = 300, window = 1000) => get(`${runPath(r)}/trace?points=${points}&window=${window}`);
export const runFile = (r, f) => `${runPath(r)}/file/${e(f)}?v=${Date.now() >> 16}`;
export const getSystem = () => get("/api/v2/system");
export const getGpuHistory = (since = 0) => get(`/api/v2/gpu_history?since=${since}`, { samples: [] });
