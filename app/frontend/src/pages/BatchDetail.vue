<script setup>
import { ref, computed, watch, onUnmounted } from "vue";
import { getBatch } from "../api.js";
import { SCENARIOS, SCENARIO_LABEL, algoLabel, algoIndex, learnerLabel, mechLabel, learnerOf, mechOf, LEARNERS, MECHANISMS, MECH_LABEL, statusLabel, statusClass, zeroRepairViol, variantLabel, netLabel, reporting, reportText, reportPct, ROLES, ROLE_LABEL, GROUP_LABEL, groupLabel } from "../labels.js";
import { langRef as curLang } from "../i18n.js";
import { pct, fmt, pm, shortTime, secToHuman, steps } from "../format.js";

const props = defineProps({ name: { type: String, required: true } });

const data = ref(null);
const missing = ref(false);
const scen = ref("all");
let timer = null;

async function load() {
  const d = await getBatch(props.name);
  (d?.rows || []).forEach(zeroRepairViol);
  missing.value = !d;
  data.value = d;
}
watch(() => props.name, () => {
  data.value = null;
  scen.value = "all";
  clearInterval(timer);
  load();
  timer = setInterval(load, 10000);
}, { immediate: true });
onUnmounted(() => clearInterval(timer));

const roleOf = (r) => r.role || "main";
const roleCount = (k) => (data.value?.rows || []).filter((r) => roleOf(r) === k).length;
const multiRole = computed(() => ROLES.filter((k) => roleCount(k)).length > 1);
const inScen = (r) => scen.value === "all" || r.scenario === scen.value;
// run details: every run of the batch (all roles), sweep and references after the formal runs
const rows = computed(() => {
  const rs = (data.value?.rows || []).filter(inScen);
  return [...rs].sort((a, b) =>
    ROLES.indexOf(roleOf(a)) - ROLES.indexOf(roleOf(b)) ||
    SCENARIOS.indexOf(a.scenario) - SCENARIOS.indexOf(b.scenario) ||
    algoIndex(a.algo) - algoIndex(b.algo) || a.variant.localeCompare(b.variant) ||
    (a.params?.lambda_max ?? 0) - (b.params?.lambda_max ?? 0));
});

// ── main view: per scenario, learner (rows) x constraint handling (columns); one cell = one formal result ──
// a reference (another batch's run) fills a cell only when this batch has no run for it, and says where it is from
const cellRows = computed(() => (data.value?.rows || []).filter((r) => roleOf(r) !== "sweep" && LEARNERS.includes(learnerOf(r.algo))));
const cellKey = (r) => `${r.scenario}|${learnerOf(r.algo)}|${mechOf(r.algo)}`;
const cells = computed(() => {
  const out = {};
  for (const r of cellRows.value) {
    const k = cellKey(r);
    if (roleOf(r) === "reference" && out[k]) continue;
    if (out[k] && roleOf(out[k]) !== "reference") { out[k] = "dup"; continue; }
    out[k] = r;
  }
  return out;
});
// the matrix only works when every formal (scenario, learner, mechanism) has exactly one run group
const matrixOk = computed(() => cellRows.value.length > 0 && !Object.values(cells.value).includes("dup"));
const matrixScens = computed(() => SCENARIOS.filter((s) => (scen.value === "all" || s === scen.value) && cellRows.value.some((r) => r.scenario === s)));
const matrixLearners = (s) => LEARNERS.filter((l) => cellRows.value.some((r) => r.scenario === s && learnerOf(r.algo) === l));
const matrixMechs = computed(() => MECHANISMS.filter((m) => cellRows.value.some((r) => mechOf(r.algo) === m)));
const cell = (s, l, m) => cells.value[`${s}|${l}|${m}`];
// a cell's state when it has no metrics yet: the most advanced seed status
function cellState(r) {
  const ss = r.seeds || [];
  for (const k of ["running", "waiting", "queued", "stopped", "skipped"]) {
    const s = ss.find((x) => x.status === k);
    if (s) return s;
  }
  return null;
}
const cellLink = (r) => {
  const s = (r.seeds || []).find((x) => x.status === "done");
  return roleOf(r) === "reference" || !s ? null : runLink(r, s);
};

// matrix tags: a Lagrangian cell says that lambda adapts (dual ascent) up to lambda_max, and where it ended
function cellTags(r) {
  const out = paramTags(r).filter((x) => !x.startsWith("λ_max="));
  const lm = r.params?.lambda_max;
  if (lm != null) out.unshift(`λ 自适应 · 上限 ${lm}`);
  if (r.final_lambda_mean != null) out.push(`末 λ=${r.final_lambda_mean.toFixed(2)}`);
  return out;
}

// ── parameter sweeps (role "sweep"): per learner, one row per lambda_max, one column per scenario ──
const sweepRows = computed(() => (data.value?.rows || []).filter((r) => roleOf(r) === "sweep"));
const sweepLearners = computed(() => LEARNERS.filter((l) => sweepRows.value.some((r) => learnerOf(r.algo) === l)));
const sweepValues = (l) => [...new Set(sweepRows.value.filter((r) => learnerOf(r.algo) === l).map((r) => r.params?.lambda_max))].sort((a, b) => a - b);
const sweepCell = (l, v, s) => sweepRows.value.find((r) => learnerOf(r.algo) === l && r.params?.lambda_max === v && r.scenario === s);
const sweepDone = computed(() => sweepRows.value.filter((r) => r.n_done === r.n_total).length);
// lambda_max chosen per learner (v5.0.3: {learner: value}; older manifests: one value for all)
const chosen = (l) => {
  const v = data.value?.lambda_max;
  const x = v != null && typeof v === "object" ? v[l] : v;
  return x == null || x !== x ? null : x;
};

// docs/data_spec.md §4: params as badges; a legacy variant equal to the row's network is already in the 网络 column,
// steps only when they differ from the batch's default
function paramTags(r) {
  const out = [];
  for (const [k, v] of Object.entries(r.params || {})) {
    if (k === "variant") { if (v && v !== (r.net ?? "graph")) out.push(variantLabel(v)); }   // a run without results: graph (the default net)
    else if (k === "lambda_max") { if (v != null) out.push(`λ_max=${v}`); }                  // null: still decided by a sweep (shown as 等待 …)
    else out.push(`${k}=${v}`);
  }
  if (r.steps && data.value?.steps && r.steps !== data.value.steps) out.push(`步数=${steps(r.steps)}`);
  return out;
}
const hasParams = computed(() => rows.value.some((r) => paramTags(r).length || (r.waiting_for || []).length));
const scenCount = (s) => (data.value?.rows || []).filter((r) => r.scenario === s && (isEval.value || roleOf(r) !== "sweep")).length;
const refLabel = (r) => (r.label ? (curLang.value === "en" ? r.label.en : r.label.zh) : null);

// first row of each scenario group gets a divider
const firstOfLearner = (i) =>
  firstOfScen(i) || learnerOf(rows.value[i - 1].algo) !== learnerOf(rows.value[i].algo);
const firstOfScen = (i) => i === 0 || rows.value[i - 1].scenario !== rows.value[i].scenario || roleOf(rows.value[i - 1]) !== roleOf(rows.value[i]);

// evaluation batches (kind === "eval"): best-of-K / single deterministic re-evaluation
const isEval = computed(() => data.value?.kind === "eval");
const evalRows = computed(() => {
  const rs = (data.value?.rows || []).filter((r) => scen.value === "all" || r.scenario === scen.value);
  const idx = (a) => (a === "greedy" ? -1 : algoIndex(a));
  return [...rs].sort((a, b) => SCENARIOS.indexOf(a.scenario) - SCENARIOS.indexOf(b.scenario) || idx(a.algo) - idx(b.algo));
});
const evalFirstOfScen = (i) => i === 0 || evalRows.value[i - 1].scenario !== evalRows.value[i].scenario;
const evalLabel = (a) => (a === "greedy" ? "贪心" : algoLabel(a));
const ks = computed(() => (data.value?.ks || [1]).map(String));
const ilpOf = (s) => data.value?.ilp?.[s];
const m = (r, k, key) => r.metrics?.[k]?.[key] || [null, null];
const speedup = (r, k) => {
  const ilp = ilpOf(r.scenario)?.ms_mean;
  return ilp == null || r.ms[0] == null ? null : ilp / (Number(k) * r.ms[0]);
};

const gap = (r) =>
  r.ilp_ar_mean == null || r.test_ar_mean_mean == null ? null : r.ilp_ar_mean - r.test_ar_mean_mean;
const relGap = (r) => (gap(r) == null || !r.ilp_ar_mean ? null : gap(r) / r.ilp_ar_mean);
const hasNet = computed(() => (data.value?.rows || []).some((r) => r.net));

const seedTitle = (s) =>
  `种子 ${s.seed ?? "—"} · ${statusLabel(s.status)}${s.progress_pct != null ? ` ${s.progress_pct}%` : ""}${s.exp_id ? " · " + s.exp_id : ""}`;
const runLink = (r, s) =>
  s.status === "done" ? `#/run/${s.branch || data.value.branch}/${r.scenario}/${r.algo}/${s.exp_id}` : null;
const resultsLink = computed(() => `#/results?branch=${encodeURIComponent(data.value?.branch ?? "")}&version=${encodeURIComponent(data.value?.version ?? "未标注版本")}&batch=${encodeURIComponent(props.name)}&view=pivot`);
</script>

<template>
  <div class="crumbs"><a href="#/batches">批次</a> / <template v-if="data?.version"><a :href="`#/batches?version=${encodeURIComponent(data.version)}`">v{{ String(data.version).replace(/^v/, "") }}</a> / </template>{{ name }}</div>
  <div v-if="missing" class="empty">找不到批次 {{ name }}</div>
  <div v-else-if="!data" class="empty">加载中…</div>
  <template v-else>
    <div class="page-head page-head--row">
      <div>
        <h1>{{ name }} <span class="badge badge--lg" :class="statusClass(data.status)">{{ statusLabel(data.status) }}</span></h1>
        <div class="page-sub" v-if="!isEval">版本 {{ data.version ?? "—" }}<template v-if="data.purpose && data.purpose !== 'main'"> · {{ data.purpose === "smoke" ? "冒烟测试" : "测试" }}</template> · {{ steps(data.steps) }} 步 · 开始于 {{ shortTime(data.started_at) }}</div>
        <div class="page-sub" v-else>评估批次（不训练）· 版本 {{ data.version ?? "—" }} · {{ data.workers ?? "—" }} 个并行进程 · 开始于 {{ shortTime(data.started_at) }}</div>
      </div>
      <div v-if="!isEval" class="head-links">
        <a v-if="data.version" class="btn btn--ghost" :href="`#/versions/v${String(data.version).replace(/^v/, '')}`">版本报告 →</a>
        <a class="btn" :href="resultsLink">在结果对比中查看 →</a>
      </div>
    </div>

    <div class="kpis">
      <div class="kpi"><div class="kpi-label">总进度</div><div class="kpi-value">{{ data.overall_pct }}%</div>
        <div class="bar bar--kpi"><div class="bar-fill" :style="{ width: data.overall_pct + '%' }"></div></div></div>
      <div class="kpi"><div class="kpi-label">已完成</div><div class="kpi-value">{{ data.done }}<span class="kpi-unit">/ {{ data.total_runs }}</span></div></div>
      <div class="kpi"><div class="kpi-label">运行 / 排队 / 等待依赖 / 中断{{ data.skipped ? " / 未运行" : "" }}</div><div class="kpi-value kpi-value--sm">{{ data.running }} / {{ data.queued }} / {{ data.waiting ?? 0 }} / {{ data.stopped }}{{ data.skipped ? " / " + data.skipped : "" }}</div></div>
      <div v-if="data.report && !data.report.finished" class="kpi kpi--wide" data-no-i18n>
        <div class="kpi-label">{{ curLang === "en" ? "Report (after training)" : "生成报告（训练结束后）" }}</div>
        <div class="kpi-value kpi-value--sm">{{ reportPct(data.report).toFixed(0) }}%</div>
        <div class="bar bar--kpi"><div class="bar-fill bar-fill--report" :style="{ width: reportPct(data.report) + '%' }"></div></div>
        <div class="kpi-foot">{{ reportText(data.report, curLang) }}</div></div>
      <div class="kpi"><div class="kpi-label">已用时</div><div class="kpi-value kpi-value--sm">{{ secToHuman(data.elapsed_seconds) }}</div>
        <div class="kpi-foot">最后更新 {{ shortTime(data.last_updated) }}</div></div>
    </div>

    <div class="tabs">
      <button class="tab" :class="{ active: scen === 'all' }" @click="scen = 'all'">全部场景</button>
      <button v-for="s in SCENARIOS" :key="s" v-show="scenCount(s)" class="tab" :class="{ active: scen === s }" @click="scen = s">
        {{ s.toUpperCase() }}
      </button>
    </div>

    <section v-if="isEval" class="panel">
      <p v-if="data.description" class="eval-desc">{{ data.description }}</p>
      <p class="dim small">报告：<code>{{ data.report ?? "—" }}</code>（全部完成后生成）。指标在整个评估结束、写出原始结果后显示；运行中只显示各任务的状态与单次耗时。</p>
      <div class="table-scroll"><table class="grid">
        <thead>
          <tr>
            <th>场景</th><th>方法</th><th>种子</th>
            <th class="num" title="单个 episode（一个测试实例、一个解）的平均耗时，单线程">单次耗时 ms</th>
            <th class="num k-first" title="每个测试实例取 K 个解中最好的一个；K = 1 即确定性输出">K</th>
            <th class="num" title="K 个解中至少有一个成功的测试实例比例">成功率</th>
            <th class="num" title="AR = average resource utilization，只在成功的测试实例上平均">AR</th>
            <th class="num" title="同一批成功实例上 ILP 最优解的 AR">ILP AR</th>
            <th class="num" title="AR gap = ILP AR − AR">AR gap</th>
            <th class="num" title="K × 单次耗时（逐个运行；合并成 batch 时更低）">耗时 ms/实例</th>
            <th class="num" title="ILP 平均耗时 ÷ 本方法耗时">比 ILP 快</th>
          </tr>
        </thead>
        <tbody>
          <template v-for="(r, i) in evalRows" :key="r.scenario + r.algo">
            <tr v-if="evalFirstOfScen(i) && ilpOf(r.scenario)" class="row-sep ilp-row">
              <td><span class="scen" :class="`scen--${r.scenario}`">{{ r.scenario.toUpperCase() }}</span></td>
              <td><b>ILP（最优）</b></td>
              <td class="dim small" colspan="3">{{ ilpOf(r.scenario).source }}</td>
              <td class="num">100%</td>
              <td class="num">{{ fmt(ilpOf(r.scenario).ar_mean) }}</td>
              <td class="num">{{ fmt(ilpOf(r.scenario).ar_mean) }}</td>
              <td class="num">0</td>
              <td class="num">{{ fmt(ilpOf(r.scenario).ms_mean, 1) }}<span class="dim small">（中位数 {{ fmt(ilpOf(r.scenario).ms_median, 1) }}）</span></td>
              <td class="num">1×</td>
            </tr>
            <tr v-for="(k, j) in ks" :key="r.scenario + r.algo + k"
                :class="{ 'row-sep': j === 0 && evalFirstOfScen(i) && !ilpOf(r.scenario), 'method-sep': j === 0 }">
              <td><span v-if="j === 0 && evalFirstOfScen(i) && !ilpOf(r.scenario)" class="scen" :class="`scen--${r.scenario}`">{{ r.scenario.toUpperCase() }}</span></td>
              <td><b v-if="j === 0">{{ evalLabel(r.algo) }}</b></td>
              <td>
                <template v-if="j === 0">
                  <span class="seeds">
                    <span v-for="s in r.seeds" :key="s.seed" class="seed" :class="`seed--${s.status}`" :title="`种子 ${s.seed} · ${statusLabel(s.status)}`">{{ s.seed }}</span>
                  </span>
                  <span class="dim small">{{ r.n_done }}/{{ r.n_total }}</span>
                </template>
              </td>
              <td class="num"><template v-if="j === 0">{{ pm(r.ms[0], r.ms[1], (v) => fmt(v, 1)) }}</template></td>
              <td class="num k-first">{{ k }}</td>
              <td class="num">{{ pm(...m(r, k, "success"), pct) }}</td>
              <td class="num">{{ pm(...m(r, k, "ar")) }}</td>
              <td class="num dim">{{ fmt(m(r, k, "ilp_ar")[0]) }}</td>
              <td class="num">{{ pm(...m(r, k, "gap")) }}</td>
              <td class="num">{{ r.ms[0] == null ? "—" : fmt(Number(k) * r.ms[0], 1) }}</td>
              <td class="num" :class="{ warn: speedup(r, k) != null && speedup(r, k) < 1 }">{{ speedup(r, k) == null ? "—" : fmt(speedup(r, k), 1) + "×" }}</td>
            </tr>
          </template>
        </tbody>
      </table></div>
      <div class="legend">
        <span><i class="seed seed--done"></i>已完成</span>
        <span><i class="seed seed--running"></i>运行中</span>
        <span><i class="seed seed--queued"></i>排队中</span>
        <span v-if="data.skipped"><i class="seed seed--skipped"></i>未运行（评估已结束）</span>
        <span>K 个解取最好：先选成功的，再选 AR 最高的；ILP AR 只用于计分。K = 1 即确定性输出（贪心为纯贪心规则）。AR 与 ILP AR 都只在成功的实例上平均，AR gap = ILP AR − AR。指标为各种子的均值 ± 标准差。"比 ILP 快"小于 1× 时标红，表示比 ILP 还慢。ILP 耗时来自 v4.3.1.7，与本批次不是同一时间、同一负载下测得。</span>
      </div>
    </section>

    <template v-else>
    <section v-if="matrixOk" class="panel">
      <div class="panel-title">结果矩阵 <span class="dim small">每格一个模型（场景 × 模型 × 约束处理）：相对 gap 与测试集违约率；未完成的格子显示当前状态</span></div>
      <div v-for="s in matrixScens" :key="s" class="mx-block">
        <div class="mx-scen"><span class="scen" :class="`scen--${s}`">{{ s.toUpperCase() }}</span> <span class="dim small">{{ SCENARIO_LABEL[s] }}</span></div>
        <div class="table-scroll"><table class="grid mx">
          <thead><tr><th>模型</th><th v-for="m in matrixMechs" :key="m">{{ MECH_LABEL[m] }}</th></tr></thead>
          <tbody>
            <tr v-for="l in matrixLearners(s)" :key="l">
              <td class="mx-learner"><b>{{ l.toUpperCase() }}</b></td>
              <td v-for="m in matrixMechs" :key="m" class="mx-cell">
                <template v-if="cell(s, l, m)">
                  <component :is="cellLink(cell(s, l, m)) ? 'a' : 'div'" :href="cellLink(cell(s, l, m))" class="mx-in">
                    <template v-if="cell(s, l, m).n_done">
                      <div class="mx-gap">{{ pct(relGap(cell(s, l, m))) }}<span class="mx-unit">gap</span></div>
                      <div class="mx-viol">
                        <span :class="{ warn: cell(s, l, m).test_cap_viol_rate_mean }">容量 {{ pct(cell(s, l, m).test_cap_viol_rate_mean) }}</span>
                        · <span :class="{ warn: cell(s, l, m).test_conflict_viol_rate_mean }">隐私 {{ pct(cell(s, l, m).test_conflict_viol_rate_mean) }}</span>
                        <template v-if="cell(s, l, m).test_exit_rate_mean"> · <span class="warn">EXIT {{ pct(cell(s, l, m).test_exit_rate_mean) }}</span></template>
                      </div>
                      <div v-if="cell(s, l, m).n_done < cell(s, l, m).n_total" class="dim small">{{ cell(s, l, m).n_done }}/{{ cell(s, l, m).n_total }} 个种子</div>
                    </template>
                    <div v-else-if="cellState(cell(s, l, m))" class="mx-state" :class="`mx-state--${cellState(cell(s, l, m)).status}`">
                      <template v-if="cellState(cell(s, l, m)).status === 'running'">
                        运行中 {{ cellState(cell(s, l, m)).progress_pct != null ? Math.round(cellState(cell(s, l, m)).progress_pct) + "%" : "" }}
                        <div class="bar"><div class="bar-fill" :style="{ width: (cellState(cell(s, l, m)).progress_pct || 0) + '%' }"></div></div>
                      </template>
                      <template v-else-if="cellState(cell(s, l, m)).status === 'waiting'">等待 {{ (cell(s, l, m).waiting_for || []).map((g) => groupLabel(g)).join("、") }}</template>
                      <template v-else>{{ statusLabel(cellState(cell(s, l, m)).status) }}</template>
                    </div>
                    <div class="mx-tags">
                      <span v-for="t in cellTags(cell(s, l, m))" :key="t" class="tag tag--param">{{ t }}</span>
                      <span v-if="roleOf(cell(s, l, m)) === 'reference'" class="tag tag--wait" :title="refLabel(cell(s, l, m))">参照 · 非本批次训练</span>
                    </div>
                  </component>
                </template>
                <span v-else class="dim">—</span>
              </td>
            </tr>
          </tbody>
        </table></div>
      </div>
      <div class="legend"><span>相对 gap = (ILP AR − AR) / ILP AR，只在无违约的测试实例上计算；容量 / 隐私 = 有容量 / 隐私违约的测试实例比例。点击已完成的格子查看该 run。</span></div>
    </section>

    <details v-if="sweepRows.length" class="panel mx-details" :open="!matrixOk">
      <summary>
        <b>参数扫描：λ_max</b>
        <span class="dim small">（每个 learner 各扫一次，{{ steps(sweepRows[0].steps) }} 步；完成 {{ sweepDone }}/{{ sweepRows.length }}）</span>
        <span v-for="l in sweepLearners" :key="l" class="tag tag--param">{{ l.toUpperCase() }}：{{ chosen(l) != null ? "λ_max=" + chosen(l) : "待定" }}</span>
      </summary>
      <div class="table-scroll"><table class="grid mx">
        <thead><tr><th>模型</th><th>λ_max</th><th v-for="s in SCENARIOS" :key="s">{{ s.toUpperCase() }}</th></tr></thead>
        <tbody>
          <template v-for="l in sweepLearners" :key="l">
            <tr v-for="(v, i) in sweepValues(l)" :key="l + v" :class="{ 'mx-chosen': v === chosen(l), 'row-sep': i === 0 }">
              <td><b v-if="i === 0">{{ l.toUpperCase() }}</b></td>
              <td><b>{{ v }}</b><span v-if="v === chosen(l)" class="dim small"> ✓ 选定</span></td>
              <td v-for="s in SCENARIOS" :key="s" class="mx-cell">
                <template v-if="sweepCell(l, v, s)">
                  <template v-if="sweepCell(l, v, s).n_done">
                    <div class="mx-gap">{{ pct(relGap(sweepCell(l, v, s))) }}<span class="mx-unit">gap</span></div>
                    <div class="mx-viol">容量 {{ pct(sweepCell(l, v, s).test_cap_viol_rate_mean) }} · 隐私 {{ pct(sweepCell(l, v, s).test_conflict_viol_rate_mean) }}<template v-if="sweepCell(l, v, s).final_lambda_mean != null"> · 末 λ={{ sweepCell(l, v, s).final_lambda_mean.toFixed(2) }}</template></div>
                  </template>
                  <div v-else-if="cellState(sweepCell(l, v, s))" class="mx-state">{{ statusLabel(cellState(sweepCell(l, v, s)).status) }}{{ cellState(sweepCell(l, v, s)).progress_pct != null ? " " + Math.round(cellState(sweepCell(l, v, s)).progress_pct) + "%" : "" }}</div>
                </template>
                <span v-else class="dim">—</span>
              </td>
            </tr>
          </template>
        </tbody>
      </table></div>
      <div class="legend"><span>λ 在训练中按局对偶上升自适应，λ_max 只是它的上限。每个 learner 选三场景平均"有效 gap"最低的上限（有违约的实例按 AR = 0 计），该 learner 的 Lagrangian 格子随后开始训练；详见报告 lam_sweep_&lt;learner&gt;.md。</span></div>
    </details>

    <details class="panel mx-details" :open="!matrixOk">
      <summary><b>全部 run</b> <span class="dim small">（每个 run 的种子、状态与全部指标，共 {{ rows.length }} 组）</span></summary>
      <div class="table-scroll"><table class="grid">
        <thead>
          <tr>
            <th v-if="multiRole">类别</th><th>场景</th><th>模型</th><th>约束处理</th><th v-if="roleCount('reference')">来源</th><th v-if="hasParams" title="区分同一模型 × 约束处理下不同 run 的参数（docs/data_spec.md）">参数</th><th v-if="hasNet">网络 / 设备</th><th>种子</th>
            <th class="num" title="success_rate = 测试实例中 M 个服务全部合法放置且无违规的比例">success_rate</th><th class="num" title="AR = average resource utilization（平均资源利用率，优化目标）">AR</th><th class="num">ILP AR</th><th class="num" title="AR gap = ILP AR − AR">AR gap</th>
            <th class="num" title="相对 gap = (ILP AR − AR) / ILP AR">相对 gap</th>
            <th class="num">容量违规</th><th class="num">冲突违规</th><th class="num" title="Maskable 选了 EXIT 的测试实例比例">EXIT 率</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(r, i) in rows" :key="roleOf(r) + r.scenario + r.algo + JSON.stringify(r.params) + r.steps + (refLabel(r) || '')" :class="{ 'row-sep': firstOfScen(i) }">
            <td v-if="multiRole" class="small">{{ firstOfScen(i) ? ROLE_LABEL[roleOf(r)] : "" }}</td>
            <td><span v-if="firstOfScen(i)" class="scen" :class="`scen--${r.scenario}`">{{ r.scenario.toUpperCase() }}</span></td>
            <td><b v-if="firstOfLearner(i)">{{ learnerLabel(r.algo) }}</b></td>
            <td :class="{ dim: mechLabel(r.algo) === '无约束' }">{{ mechLabel(r.algo) }}</td>
            <td v-if="roleCount('reference')" class="small">{{ refLabel(r) }}</td>
            <td v-if="hasParams">
              <span v-for="t in paramTags(r)" :key="t" class="tag tag--param">{{ t }}</span>
              <span v-for="g in r.waiting_for || []" :key="g" class="tag tag--wait">等待 {{ groupLabel(g) }}</span>
            </td>
            <td v-if="hasNet">{{ netLabel(r.net) }}<span v-if="r.device" class="dim small"> · {{ r.device.toUpperCase() }}</span></td>
            <td>
              <span class="seeds">
                <component :is="runLink(r, s) ? 'a' : 'span'" v-for="s in r.seeds" :key="`${s.seed}-${s.exp_id}`" :href="runLink(r, s)"
                  class="seed" :class="`seed--${s.status}`" :title="seedTitle(s)">
                  {{ s.status === "running" && s.progress_pct != null ? Math.round(s.progress_pct) : s.seed ?? "·" }}
                </component>
              </span>
              <span class="dim small">{{ r.n_done }}/{{ r.n_total }}</span>
            </td>
            <td class="num">{{ pm(r.test_success_rate_mean, r.test_success_rate_std, pct) }}</td>
            <td class="num">{{ pm(r.test_ar_mean_mean, r.test_ar_mean_std) }}</td>
            <td class="num dim">{{ fmt(r.ilp_ar_mean) }}</td>
            <td class="num">{{ fmt(gap(r)) }}</td>
            <td class="num">{{ pct(relGap(r)) }}</td>
            <td class="num" :class="{ warn: r.test_cap_viol_rate_mean }">{{ pct(r.test_cap_viol_rate_mean) }}</td>
            <td class="num" :class="{ warn: r.test_conflict_viol_rate_mean }">{{ pct(r.test_conflict_viol_rate_mean) }}</td>
            <td class="num" :class="{ warn: r.test_exit_rate_mean }">{{ pct(r.test_exit_rate_mean) }}</td>
          </tr>
        </tbody>
      </table></div>
      <div class="legend">
        <span><i class="seed seed--done"></i>已完成（可点击查看）</span>
        <span><i class="seed seed--running"></i>运行中（数字为训练进度 %）</span>
        <span><i class="seed seed--queued"></i>排队中</span>
        <span v-if="data.waiting"><i class="seed seed--waiting"></i>等待依赖（例如等参数扫描结束）</span>
        <span v-if="data.skipped"><i class="seed seed--skipped"></i>未运行（批次已结束）</span>
        <span><i class="seed seed--stopped"></i>已中断</span>
        <span>指标为已完成种子的均值 ± 标准差；AR = average resource utilization；AR gap = ILP AR − AR；相对 gap = (ILP AR − AR) / ILP AR；success_rate = M 个服务全部合法放置且无违规的测试实例比例</span>
      </div>
    </details>
    </template>
  </template>
</template>

<style scoped>
.eval-desc { margin: 0 0 6px; }
.method-sep td { border-top: 1px solid var(--border, #e3e6ec); }
.k-first { border-left: 1px solid var(--border, #ddd); }
.ilp-row td { background: var(--bg-soft, rgba(127,127,127,.06)); }
.panel-title { font-weight: 600; margin-bottom: 10px; }
.head-links { display: flex; gap: 8px; flex-wrap: wrap; }
.mx-block { margin-bottom: 14px; }
.mx-scen { margin-bottom: 6px; }
.mx { table-layout: fixed; min-width: 640px; }
.mx th:first-child, .mx td:first-child { width: 80px; }
.mx td.mx-cell { vertical-align: top; white-space: normal; padding: 8px 10px; }
.mx-in { display: block; color: inherit; text-decoration: none; }
a.mx-in:hover .mx-gap { color: var(--primary, #2f6fde); }
.mx-gap { font-size: 18px; font-weight: 600; line-height: 1.2; }
.mx-unit { font-size: 11px; font-weight: 400; color: var(--text-dim); margin-left: 4px; }
.mx-viol { font-size: 12px; color: var(--text-dim); }
.mx-state { font-size: 13px; color: var(--text-dim); }
.mx-state--running { color: var(--primary, #2f6fde); }
.mx-state .bar { margin-top: 4px; }
.mx-tags { margin-top: 4px; }
.mx-tags .tag { margin-right: 4px; }
.mx-chosen td { background: var(--primary-soft, #eef4ff); }
.mx-details > summary { cursor: pointer; padding: 2px 0 8px; }
.mx-details[open] > summary { margin-bottom: 6px; }
</style>
