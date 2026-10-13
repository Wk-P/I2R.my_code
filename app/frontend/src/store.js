// App-wide state polled once and shared by every page: git branch (with the
// branch the user picked for browsing results) and the machine/process
// snapshot shown in the top bar.
import { reactive } from "vue";
import { getBranch, getSystem } from "./api.js";

export const store = reactive({
  branch: null,          // checked-out branch
  branches: [],
  viewBranch: null,      // space browsed on the 结果对比 page (its own selector; other pages show every space)
  resultBranches: [],    // [{name, runs}] branches that have a results/ tree
  system: null,
  updatedAt: null,
});

let started = false;
export function startPolling() {
  if (started) return;
  started = true;
  const loadBranch = async () => {
    const b = await getBranch();
    store.branch = b.current;
    store.branches = b.branches || [];
    store.resultBranches = b.result_branches || [];
    if (!store.viewBranch) {
      let saved = null;
      try { saved = localStorage.getItem("viewBranch"); } catch {}
      const ok = (n) => store.resultBranches.some((r) => r.name === n);
      store.viewBranch = saved && ok(saved) ? saved : b.current;
    }
  };
  const loadSystem = async () => {
    store.system = await getSystem();
    store.updatedAt = new Date();
  };
  loadBranch();
  loadSystem();
  setInterval(loadBranch, 15000);
  setInterval(loadSystem, 5000);
}

// Live training processes: scenarios/*/run_all.py (old) or python -m paper_rl.train (v4.3.1.3+).
export const trainingProcs = () =>
  (store.system?.processes || []).filter((p) => p.scenario && p.algo);

export function setViewBranch(b) {
  if (!b || b === store.viewBranch) return;
  store.viewBranch = b;
  try { localStorage.setItem("viewBranch", b); } catch {}
}

// "阶段5 · 论文最终实验" for a results space name (falls back to the name).
export const spaceLabel = (name) => store.resultBranches.find((b) => b.name === name)?.label ?? name;

// Project stages (result_spaces.json "stage"): batches and spaces are grouped by stage, test spaces (smoke / check)
// belong to the current stage. [{stage, label, current}] newest first; label = the stage's main space label.
export const stageOf = (space) => store.resultBranches.find((b) => b.name === space)?.stage ?? null;
export const currentStage = () => stageOf(store.branch);
export function stages() {
  const out = new Map();
  for (const b of store.resultBranches)
    if (b.stage != null && b.kind === "branch" && !out.has(b.stage))
      out.set(b.stage, { stage: b.stage, label: b.label, current: b.stage === currentStage() });
  return [...out.values()].sort((a, b) => b.stage - a.stage);
}
