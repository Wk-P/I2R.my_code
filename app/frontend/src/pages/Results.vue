<script setup>
// 实验结果: hierarchy tree (版本 → 批次) on the left, filters + detail list /
// pivot comparison on the right. Every filter lives in the URL query
// (#/results?branch=..&batch=..&scen=lt,eq&algo=..&view=pivot) so a view
// can be bookmarked or linked from other pages.
import { ref, reactive, computed, watch, onMounted, onUnmounted } from "vue";
import { store, setViewBranch, spaceLabel } from "../store.js";
import { getExperiments } from "../api.js";
import { SCENARIOS, algoLabel, algoIndex, scenarioIndex, zeroRepairViol } from "../labels.js";
import { pct, fmt, shortTime, steps } from "../format.js";

const NO_VERSION = "未标注版本";
const NO_BATCH = "（未归属批次）";

// ── filter state <-> URL ────────────────────────────────────────────────
let keepScope = false;
const f = reactive({ version: "", batch: "", scen: [], algo: [], variant: "", q: "", view: "list", metric: "success" });
function readUrl() {
  const qs = new URLSearchParams(window.location.hash.split("?")[1] || "");
  // a link that names its branch keeps its own version/batch filters
  if (qs.get("branch") && qs.get("branch") !== store.viewBranch) { keepScope = true; setViewBranch(qs.get("branch")); }
  f.version = qs.get("version") || "";
  f.batch = qs.get("batch") || "";
  f.scen = (qs.get("scen") || "").split(",").filter(Boolean);
  f.algo = (qs.get("algo") || "").split(",").filter(Boolean);
  f.variant = qs.get("variant") || "";
  f.q = qs.get("q") || "";
  f.view = qs.get("view") || "list";
  f.metric = qs.get("metric") || "success";
}
function writeUrl() {
  const qs = new URLSearchParams();
  if (store.viewBranch) qs.set("branch", store.viewBranch);
  for (const k of ["version", "batch", "variant", "q"]) if (f[k]) qs.set(k, f[k]);
  if (f.scen.length) qs.set("scen", f.scen.join(","));
  if (f.algo.length) qs.set("algo", f.algo.join(","));
  if (f.view !== "list") qs.set("view", f.view);
  if (f.metric !== "success") qs.set("metric", f.metric);
  const s = qs.toString();
  history.replaceState(null, "", "#/results" + (s ? "?" + s : ""));
}
readUrl();
watch(f, writeUrl, { deep: true });
const onHash = () => { if (window.location.hash.startsWith("#/results")) readUrl(); };
onMounted(() => window.addEventListener("hashchange", onHash));
onUnmounted(() => window.removeEventListener("hashchange", onHash));

// ── data ───────────────────────────────────────────────────────────────
const rows = ref([]);
const loading = ref(true);
async function load() {
  loading.value = true;
  rows.value = (await getExperiments(branchShown.value)).map(zeroRepairViol);
  loading.value = false;
}
const branchShown = computed(() => store.viewBranch);
let firstLoad = true;
watch(branchShown, (b) => {
  if (!b) return;
  // tree nodes belong to the old branch -- unless the branch came with the URL
  if (!firstLoad && !keepScope) Object.assign(f, { version: "", batch: "" });
  firstLoad = keepScope = false;
  writeUrl();
  load();
}, { immediate: true });

const vOf = (r) => r.version || NO_VERSION;
const bOf = (r) => r.batch || NO_BATCH;

// ── hierarchy tree: version -> batch ───────────────────────────────────
const tree = computed(() => {
  const vs = new Map();
  for (const r of rows.value) {
    const v = vOf(r);
    if (!vs.has(v)) vs.set(v, { version: v, count: 0, latest: "", batches: new Map() });
    const node = vs.get(v);
    node.count++;
    if ((r.created_at ?? "") > node.latest) node.latest = r.created_at ?? "";
    const b = bOf(r);
    if (!node.batches.has(b)) node.batches.set(b, { batch: b, count: 0, latest: "" });
    const bn = node.batches.get(b);
    bn.count++;
    if ((r.created_at ?? "") > bn.latest) bn.latest = r.created_at ?? "";
  }
  return [...vs.values()]
    .sort((a, b) => b.latest.localeCompare(a.latest))
    .map((v) => ({ ...v, batches: [...v.batches.values()].sort((a, b) => b.latest.localeCompare(a.latest)) }));
});
const collapsed = reactive({});
const treeOpen = ref(false);   // mobile: the scope tree is folded behind a button
function pickVersion(v) { f.version = f.version === v && !f.batch ? "" : v; f.batch = ""; }
function pickBatch(v, b) { f.version = v; f.batch = f.batch === b ? "" : b; }

// ── filtering ──────────────────────────────────────────────────────────
const inTree = computed(() => rows.value.filter((r) =>
  (!f.version || vOf(r) === f.version) && (!f.batch || bOf(r) === f.batch)));
const algosAvail = computed(() => [...new Set(inTree.value.map((r) => r.algo))].sort((a, b) => algoIndex(a) - algoIndex(b)));
const variantsAvail = computed(() => [...new Set(inTree.value.map((r) => r.variant).filter(Boolean))].sort());
const scensAvail = computed(() => SCENARIOS.filter((s) => inTree.value.some((r) => r.scenario === s)));

const filtered = computed(() => {
  const q = f.q.trim().toLowerCase();
  return inTree.value.filter((r) =>
    (!f.scen.length || f.scen.includes(r.scenario)) &&
    (!f.algo.length || f.algo.includes(r.algo)) &&
    (!f.variant || r.variant === f.variant) &&
    (!q || [r.exp_id, r.run, r.batch, r.algo, algoLabel(r.algo), r.variant].some((x) => (x ?? "").toLowerCase().includes(q))));
});
const toggle = (arr, v) => { const i = arr.indexOf(v); i === -1 ? arr.push(v) : arr.splice(i, 1); };
function reset() {
  Object.assign(f, { version: "", batch: "", scen: [], algo: [], variant: "", q: "" });
}
const activeFilters = computed(() =>
  [f.version, f.batch, f.variant, f.q].filter(Boolean).length + f.scen.length + f.algo.length);

// ── detail list: sorting + paging ──────────────────────────────────────
const gap = (r) => (r.ilp_ar == null || r.test_ar_mean == null ? null : r.ilp_ar - r.test_ar_mean);
const COLS = [
  { key: "created_at", label: "完成时间", get: (r) => r.created_at ?? "" },
  { key: "batch", label: "批次", get: (r) => vOf(r) + bOf(r) },
  { key: "scenario", label: "场景", get: (r) => scenarioIndex(r.scenario) },
  { key: "algo", label: "算法", get: (r) => algoIndex(r.algo) },
  { key: "variant", label: "变体", get: (r) => r.variant ?? "" },
  { key: "seed", label: "种子", get: (r) => r.seed ?? 0, num: true },
  { key: "train_steps", label: "步数", get: (r) => r.train_steps ?? 0, num: true },
  { key: "test_success_rate", label: "success_rate", get: (r) => r.test_success_rate ?? -1, num: true },
  { key: "test_ar_mean", label: "AR", get: (r) => r.test_ar_mean ?? -1, num: true },
  { key: "ilp_ar", label: "ILP AR", get: (r) => r.ilp_ar ?? -1, num: true },
  { key: "gap", label: "AR gap", get: (r) => gap(r) ?? 9, num: true },
  { key: "test_cap_viol_rate", label: "容量违规", get: (r) => r.test_cap_viol_rate ?? -1, num: true },
  { key: "test_conflict_viol_rate", label: "冲突违规", get: (r) => r.test_conflict_viol_rate ?? -1, num: true },
];
const TIPS = {
  test_ar_mean: "AR = average resource utilization（平均资源利用率，优化目标）",
  test_success_rate: "success_rate = 测试实例中 M 个服务全部合法放置且无违规的比例",
  gap: "AR gap = ILP AR − AR",
};
const sort = reactive({ key: "created_at", dir: -1 });
function sortBy(c) {
  if (sort.key === c.key) sort.dir = -sort.dir;
  else { sort.key = c.key; sort.dir = c.key === "created_at" || c.num ? -1 : 1; }
}
const sorted = computed(() => {
  const c = COLS.find((x) => x.key === sort.key);
  return [...filtered.value].sort((a, b) => {
    const x = c.get(a), y = c.get(b);
    return (x < y ? -1 : x > y ? 1 : 0) * sort.dir;
  });
});
const PAGE = 50;
const page = ref(1);
watch(filtered, () => { page.value = 1; });
const pages = computed(() => Math.max(1, Math.ceil(sorted.value.length / PAGE)));
const pageRows = computed(() => sorted.value.slice((page.value - 1) * PAGE, page.value * PAGE));
const go = (r) => { window.location.hash = `#/run/${branchShown.value}/${r.scenario}/${r.algo}/${r.run}`; };

// ── pivot: (algo, variant) x scenario ──────────────────────────────────
const METRICS = {
  success: { label: "success_rate", get: (r) => r.test_success_rate, f: (v) => pct(v), better: 1 },
  ar: { label: "AR", get: (r) => r.test_ar_mean, f: (v) => fmt(v), better: 1 },
  gap: { label: "AR gap", get: gap, f: (v) => fmt(v), better: -1 },
  cap: { label: "容量违规", get: (r) => r.test_cap_viol_rate, f: (v) => pct(v), better: -1 },
  conflict: { label: "冲突违规", get: (r) => r.test_conflict_viol_rate, f: (v) => pct(v), better: -1 },
};
function stats(vals) {
  vals = vals.filter((v) => v != null);
  if (!vals.length) return null;
  const m = vals.reduce((s, v) => s + v, 0) / vals.length;
  const sd = vals.length > 1 ? Math.sqrt(vals.reduce((s, v) => s + (v - m) ** 2, 0) / (vals.length - 1)) : 0;
  return { m, sd, n: vals.length };
}
const pivot = computed(() => {
  const M = METRICS[f.metric];
  const scens = SCENARIOS.filter((s) => filtered.value.some((r) => r.scenario === s));
  const keys = new Map();
  for (const r of filtered.value) {
    const k = r.algo + "|" + (r.variant ?? "");
    if (!keys.has(k)) keys.set(k, { algo: r.algo, variant: r.variant ?? "", cells: {} });
    (keys.get(k).cells[r.scenario] ??= []).push(r);
  }
  const body = [...keys.values()]
    .sort((a, b) => algoIndex(a.algo) - algoIndex(b.algo) || a.variant.localeCompare(b.variant))
    .map((row) => ({ ...row, stats: Object.fromEntries(scens.map((s) => [s, stats((row.cells[s] || []).map(M.get))])) }));
  // best value per scenario column
  const best = {};
  for (const s of scens) {
    const ms = body.map((b) => b.stats[s]?.m).filter((v) => v != null);
    best[s] = ms.length ? (M.better > 0 ? Math.max(...ms) : Math.min(...ms)) : null;
  }
  const hasVariant = body.some((b) => b.variant);
  return { scens, body, best, M, hasVariant };
});
function drill(row, s) {
  Object.assign(f, { scen: [s], algo: [row.algo], variant: row.variant, view: "list" });
}
</script>

<template>
  <div class="page-head">
    <h1>实验结果</h1>
    <div class="page-sub">层级：版本 → 批次 → 场景 → 算法 → 单次运行 · 先选范围（版本或批次），再按场景、算法筛选，可切换明细 / 汇总对比</div>
  </div>

  <div class="results-layout">
    <!-- hierarchy -->
    <button class="btn btn--ghost tree-toggle" @click="treeOpen = !treeOpen">
      选择范围：{{ f.batch || f.version || "全部实验" }} {{ treeOpen ? "▴" : "▾" }}
    </button>
    <aside class="panel tree" :class="{ 'tree--open': treeOpen }" @click="($event.target.closest('.tree-label, .tree-root') && (treeOpen = false))">
      <div class="tree-branch"><b>{{ spaceLabel(branchShown) }}</b><div class="dim small mono">results/{{ branchShown }}/ · 顶栏切换</div></div>
      <div class="tree-node tree-root" :class="{ active: !f.version && !f.batch }" @click="f.version = ''; f.batch = ''">
        全部实验 <span class="tree-count">{{ rows.length }}</span>
      </div>
      <div v-for="v in tree" :key="v.version" class="tree-group">
        <div class="tree-node tree-version" :class="{ active: f.version === v.version && !f.batch }">
          <span class="tree-caret" @click.stop="collapsed[v.version] = !collapsed[v.version]">{{ collapsed[v.version] ? "▸" : "▾" }}</span>
          <span class="tree-label" @click="pickVersion(v.version)">{{ v.version }}</span>
          <span class="tree-count">{{ v.count }}</span>
        </div>
        <template v-if="!collapsed[v.version]">
          <div v-for="b in v.batches" :key="b.batch" class="tree-node tree-batch"
            :class="{ active: f.version === v.version && f.batch === b.batch }" @click="pickBatch(v.version, b.batch)" :title="b.batch">
            <span class="tree-label">{{ b.batch }}</span><span class="tree-count">{{ b.count }}</span>
          </div>
        </template>
      </div>
    </aside>

    <div class="results-main">
      <!-- filters -->
      <section class="panel filters">
        <div class="filter-row">
          <span class="filter-label">场景</span>
          <button v-for="s in scensAvail" :key="s" class="chip-btn" :class="{ on: f.scen.includes(s) }" @click="toggle(f.scen, s)">{{ s.toUpperCase() }}</button>
          <span class="filter-sep"></span>
          <span class="filter-label">算法</span>
          <button v-for="a in algosAvail" :key="a" class="chip-btn" :class="{ on: f.algo.includes(a) }" @click="toggle(f.algo, a)">{{ algoLabel(a) }}</button>
        </div>
        <div class="filter-row">
          <template v-if="variantsAvail.length">
            <span class="filter-label">变体</span>
            <select v-model="f.variant"><option value="">全部</option><option v-for="v in variantsAvail" :key="v" :value="v">{{ v }}</option></select>
            <span class="filter-sep"></span>
          </template>
          <span class="filter-label">搜索</span>
          <input v-model="f.q" class="search" placeholder="exp_id / 批次 / 算法…" />
          <button v-if="activeFilters" class="btn btn--ghost" @click="reset">清除筛选（{{ activeFilters }}）</button>
          <span class="filter-grow"></span>
          <div class="seg">
            <button :class="{ on: f.view === 'list' }" @click="f.view = 'list'">明细</button>
            <button :class="{ on: f.view === 'pivot' }" @click="f.view = 'pivot'">汇总对比</button>
          </div>
        </div>
        <div class="scope">
          范围：<b>{{ spaceLabel(branchShown) }}</b>
          <template v-if="f.version"> / <b>{{ f.version }}</b></template>
          <template v-if="f.batch"> / <a :href="`#/batches/${f.batch}`">{{ f.batch }}</a></template>
          · 共 <b>{{ filtered.length }}</b> 条运行
        </div>
      </section>

      <div v-if="loading" class="panel empty">加载中…</div>
      <div v-else-if="!filtered.length" class="panel empty">没有符合条件的运行</div>

      <!-- detail list -->
      <section v-else-if="f.view === 'list'" class="panel">
        <div class="table-scroll">
          <table class="grid">
            <thead>
              <tr>
                <th v-for="c in COLS" :key="c.key" :class="[{ num: c.num }, 'sortable', { sorted: sort.key === c.key }]" :title="TIPS[c.key]" @click="sortBy(c)">
                  {{ c.label }}<span class="sort-ind">{{ sort.key === c.key ? (sort.dir > 0 ? "▲" : "▼") : "" }}</span>
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="r in pageRows" :key="r.scenario + r.algo + r.run" class="clickable" @click="go(r)">
                <td>{{ shortTime(r.created_at) }}</td>
                <td :title="`版本 ${vOf(r)}`">{{ bOf(r) }}</td>
                <td><span class="scen" :class="`scen--${r.scenario}`">{{ r.scenario.toUpperCase() }}</span></td>
                <td>{{ algoLabel(r.algo) }}<span v-if="r.is_bc" class="tag">BC</span></td>
                <td><span v-if="r.variant" class="tag">{{ r.variant }}</span></td>
                <td class="num">{{ r.seed ?? "—" }}</td>
                <td class="num">{{ steps(r.train_steps) }}</td>
                <td class="num">{{ pct(r.test_success_rate) }}</td>
                <td class="num">{{ fmt(r.test_ar_mean) }}</td>
                <td class="num dim">{{ fmt(r.ilp_ar) }}</td>
                <td class="num">{{ fmt(gap(r)) }}</td>
                  <td class="num" :class="{ warn: r.test_cap_viol_rate }">{{ pct(r.test_cap_viol_rate) }}</td>
                  <td class="num" :class="{ warn: r.test_conflict_viol_rate }">{{ pct(r.test_conflict_viol_rate) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="pager" v-if="pages > 1">
          <button class="btn btn--ghost" :disabled="page === 1" @click="page--">上一页</button>
          <span>第 {{ page }} / {{ pages }} 页</span>
          <button class="btn btn--ghost" :disabled="page === pages" @click="page++">下一页</button>
        </div>
      </section>

      <!-- pivot -->
      <section v-else class="panel">
        <div class="panel-head">
          <h2>汇总对比 · {{ pivot.M.label }}</h2>
          <div class="seg">
            <button v-for="(m, k) in METRICS" :key="k" :class="{ on: f.metric === k }" @click="f.metric = k">{{ m.label }}</button>
          </div>
        </div>
        <table class="grid pivot">
          <thead>
            <tr><th>算法</th><th v-if="pivot.hasVariant">变体</th><th v-for="s in pivot.scens" :key="s" class="num">{{ s.toUpperCase() }}</th></tr>
          </thead>
          <tbody>
            <tr v-for="row in pivot.body" :key="row.algo + row.variant">
              <td>{{ algoLabel(row.algo) }}</td>
              <td v-if="pivot.hasVariant"><span v-if="row.variant" class="tag">{{ row.variant }}</span></td>
              <td v-for="s in pivot.scens" :key="s" class="num" :class="{ best: row.stats[s] && row.stats[s].m === pivot.best[s] }">
                <a v-if="row.stats[s]" class="cell-link" @click="drill(row, s)" title="查看这些运行">
                  {{ pivot.M.f(row.stats[s].m) }}<span v-if="row.stats[s].n > 1" class="sd"> ± {{ pivot.M.f(row.stats[s].sd) }}</span>
                  <span class="n">n={{ row.stats[s].n }}</span>
                </a>
                <span v-else class="dim">—</span>
              </td>
            </tr>
          </tbody>
        </table>
        <div class="note">单元格为筛选范围内所有运行的均值 ± 样本标准差（n 为运行数），绿色为该列最优。点击单元格进入对应明细。AR = average resource utilization（优化目标），AR gap = ILP AR − AR；success_rate = M 个服务全部合法放置且无违规的测试实例比例。跨批次汇总时请先在左侧限定批次，避免混入不同代码版本。</div>
      </section>
    </div>
  </div>
</template>
