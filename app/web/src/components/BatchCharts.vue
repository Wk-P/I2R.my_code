<script setup>
// Chart tab of a batch, built for learner x constraint-handling studies (v5.0.x):
//   1. gap vs violation trade-off (finished runs)     4. Lagrangian: lambda and violations during training
//   2. learning curves (report_data.json checkpoints) 5. lambda_max sweep (lam_sweep_<learner>.json / runs)
//   3. training process (training_curve.csv)          6. per-instance time vs the ILP (report_data.json)
import { ref, computed, onMounted, watch } from "vue";
import { t, lang } from "../i18n.js";
import { getTrace } from "../api.js";
import { steps as fsteps } from "../format.js";
import { SCENARIOS, RL_ALGORITHMS, STRATEGIES, scenName, modelName, STRATEGY_COLOR, RL_SYMBOL, SCEN_COLOR, fromDir } from "../labels.js";
import Chart from "./Chart.vue";

const props = defineProps({ b: Object, data: Object });
const scens = computed(() => SCENARIOS.filter((s) => props.b.runs.some((r) => r.scenario === s)));
const main = computed(() => props.b.runs.filter((r) => r.role === "main" && RL_ALGORITHMS.includes(r.RL_algorithm)));
const learners = computed(() => RL_ALGORITHMS.filter((l) => main.value.some((r) => r.RL_algorithm === l)));
const mechs = computed(() => STRATEGIES.filter((m) => main.value.some((r) => r.strategy === m)));
const P = (v) => (v == null ? null : +(100 * v).toFixed(3));
const axisPct = (name) => ({ type: "value", name, axisLabel: { formatter: "{value}%" }, nameGap: 10 });
const tip = { trigger: "axis", confine: true, valueFormatter: (v) => (v == null ? "—" : typeof v === "number" ? v.toFixed(2) : v) };
const grid = { left: 48, right: 14, top: 30, bottom: 30 };
const stepAxis = (max) => ({ type: "value", min: 0, max: max || undefined, axisLabel: { formatter: (v) => fsteps(v) } });

// ── 1. trade-off ──────────────────────────────────────────────────────────────
const doneMain = computed(() => [...main.value, ...props.b.references].filter((r) => r.status === "done" && r.metrics));
const tradeoff = (s) => {
  lang.value;
  const series = [];
  for (const l of learners.value)
    for (const m of mechs.value) {
      const rs = doneMain.value.filter((r) => r.scenario === s && r.RL_algorithm === l && r.strategy === m);
      if (!rs.length) continue;
      series.push({ type: "scatter", name: modelName(l, m), symbol: RL_SYMBOL[l], symbolSize: 13,
        itemStyle: { color: STRATEGY_COLOR[m], opacity: 0.9, borderColor: "#fff", borderWidth: 1 },
        data: rs.map((r) => [P((r.metrics.cap_viol ?? 0) + (r.metrics.priv_viol ?? 0)), P(r.metrics.rel_gap), r.seed]) });
    }
  return {
    grid: { left: 52, right: 18, top: 34, bottom: 44 },
    tooltip: { trigger: "item", formatter: (p) => `<b>${p.seriesName}</b><br>${t("违约", "violations")} ${p.value[0]?.toFixed(1)}% · gap ${p.value[1]?.toFixed(1)}%` },
    xAxis: { ...axisPct(t("违约率（容量 + 隐私）", "violation rate (capacity + privacy)")), nameLocation: "middle", nameGap: 28, min: 0 },
    yAxis: { ...axisPct(t("相对 gap", "relative gap")), scale: true },
    series: [...series, { type: "line", data: [], markLine: { silent: true, symbol: "unconstraint", lineStyle: { type: "dashed", color: "#059669" },
      label: { formatter: t("目标 5%", "goal 5%"), position: "insideEndTop" }, data: [{ yAxis: 5 }] } }],
  };
};

// ── 2. learning curves (report_data.json) ─────────────────────────────────────
const curveMetric = ref("test_gap");
const CURVE_METRICS = computed(() => [["test_gap", t("测试集相对 gap", "Test relative gap")], ["train_gap", t("训练集相对 gap", "Training-set relative gap")],
  ["test_fail", t("测试未通过率（EXIT / 违约）", "Test failure rate (EXIT / violation)")]]);
const curves = computed(() => (props.data?.report?.curves || []).filter((p) => p.key && props.b.runs.some((r) => r.exp_id === p.exp_id)));
const curveOpt = (l, s) => {
  lang.value;
  const series = mechs.value.map((m) => {
    const pts = curves.value.filter((p) => p.scenario === s && fromDir(p.algo)[0] === l && fromDir(p.algo)[1] === m)
      .sort((a, b) => a.step - b.step);
    return { type: "line", name: m, color: STRATEGY_COLOR[m], symbolSize: 6, data: pts.map((p) => [p.step, P(p[curveMetric.value])]) };
  });
  return { grid, tooltip: tip, xAxis: stepAxis(props.b.params?.steps), yAxis: { ...axisPct(""), scale: true }, series };
};

// ── 3 & 4. training traces (training_curve.csv) ──────────────────────────────
const traces = ref({});
const traceWindow = ref(1000);
const loading = ref(false);
async function loadTraces() {
  loading.value = true;
  const rs = props.b.runs.filter((r) => r.exp_id && r.space && ["done", "running", "stopped"].includes(r.status) && (r.role === "main" || r.role === "sweep"));
  traces.value = {};
  // drawn as they arrive (a 20M run is ~80 MB of csv; the backend caches the result after the first read)
  const w = traceWindow.value;
  await Promise.all(rs.map(async (r) => {
    const tr = await getTrace(r, 240, w);
    if (tr && w === traceWindow.value) traces.value = { ...traces.value, [r.exp_id]: tr };
  }));
  loading.value = false;
}
onMounted(loadTraces);
watch(traceWindow, loadTraces);
const traceMetric = ref("ar");
const TRACE_METRICS = computed(() => [["ar", t("训练局 AR", "Training-episode AR")], ["viol", t("违约局占比（容量 + 隐私）", "Episodes with a violation (capacity + privacy)")],
  ["reward", t("每局回报", "Episode return")], ["success", t("无违约完成率", "Violation-free completion")]]);
const traceVal = (tr, k) => {
  const S = tr.series;
  if (k === "viol") return S.cap_viol?.map((v, i) => (v == null ? null : P(v + (S.priv_viol?.[i] ?? 0))));
  if (k === "ar" || k === "success") return S[k]?.map((v) => P(v));
  return S[k];
};
const traceOpt = (l, s) => {
  lang.value;
  const series = mechs.value.map((m) => {
    const r = main.value.find((x) => x.scenario === s && x.RL_algorithm === l && x.strategy === m);
    const tr = r && traces.value[r.exp_id];
    const ys = tr ? traceVal(tr, traceMetric.value) || [] : [];
    return { type: "line", name: m, color: STRATEGY_COLOR[m], showSymbol: false, lineWidth: 1.5, data: tr ? tr.step.map((x, i) => [x, ys[i]]) : [] };
  });
  const pctAxis = traceMetric.value !== "reward";
  return { grid, tooltip: tip, xAxis: stepAxis(props.b.params?.steps), yAxis: pctAxis ? { ...axisPct(""), scale: true } : { type: "value", scale: true }, series };
};
const lagRuns = (l) => main.value.filter((r) => r.RL_algorithm === l && r.strategy === "lagrangian");
const lamMax = (l) => { const x = props.b.params?.lambda_max; return x == null ? null : typeof x === "object" ? x[l] : x; };
const lamOpt = (l) => {
  lang.value;
  const series = scens.value.map((s) => {
    const r = lagRuns(l).find((x) => x.scenario === s);
    const tr = r && traces.value[r.exp_id];
    return { type: "line", name: s.toUpperCase(), color: SCEN_COLOR[s], showSymbol: false,
      data: tr && tr.series.lambda ? tr.step.map((x, i) => [x, tr.series.lambda[i]]) : [] };
  });
  const cap = lamMax(l) ?? lagRuns(l)[0]?.params?.lambda_max;
  if (cap != null && !Number.isNaN(cap))
    series.push({ type: "line", data: [], markLine: { silent: true, symbol: "unconstraint", lineStyle: { type: "dashed", color: "#b91c1c" },
      label: { formatter: `λ_max = ${cap}`, position: "insideEndTop" }, data: [{ yAxis: cap }] } });
  return { grid, legend: { top: 0, right: 0, itemWidth: 14, textStyle: { fontSize: 11 } }, tooltip: tip, xAxis: stepAxis(props.b.params?.steps), yAxis: { type: "value", name: "λ", min: 0 }, series };
};
const jcOpt = (l) => {
  lang.value;
  const series = scens.value.map((s) => {
    const r = lagRuns(l).find((x) => x.scenario === s);
    const tr = r && traces.value[r.exp_id];
    return { type: "line", name: s.toUpperCase(), color: SCEN_COLOR[s], showSymbol: false,
      data: tr ? tr.step.map((x, i) => [x, tr.series.cap_viol?.[i] == null ? null : +(tr.series.cap_viol[i] + (tr.series.priv_viol?.[i] ?? 0)).toFixed(5)]) : [] };
  });
  series.push({ type: "line", data: [], markLine: { silent: true, symbol: "unconstraint", lineStyle: { type: "dashed", color: "#059669" },
    label: { formatter: "d = 0.001", position: "insideEndTop" }, data: [{ yAxis: 0.001 }] } });
  return { grid, legend: { top: 0, right: 0, itemWidth: 14, textStyle: { fontSize: 11 } }, tooltip: tip, xAxis: stepAxis(props.b.params?.steps),
    yAxis: { type: "log", name: "J_C", min: 1e-4, logBase: 10, axisLabel: { formatter: (v) => (v >= 0.01 ? v : v.toExponential(0)) } }, series };
};
const hasLag = computed(() => learners.value.some((l) => lagRuns(l).some((r) => traces.value[r.exp_id]?.series?.lambda)));

// ── 5. lambda_max sweep ──────────────────────────────────────────────────────
const sweepRuns = computed(() => props.b.runs.filter((r) => r.role === "sweep"));
const sweepLearners = computed(() => RL_ALGORITHMS.filter((l) => sweepRuns.value.some((r) => r.RL_algorithm === l)));
const sweepOpt = (l) => {
  lang.value;
  const js = (props.data?.sweeps || []).find((x) => x.RL_algorithm === l);
  const vals = [...new Set(sweepRuns.value.filter((r) => r.RL_algorithm === l).map((r) => r.params?.lambda_max))].sort((a, b) => a - b);
  const val = (v, s) => {
    const row = js?.rows.find((r) => r.lambda_max === v && r.scenario === s);
    if (row) return P(row.eff_gap);
    const r = sweepRuns.value.find((x) => x.RL_algorithm === l && x.scenario === s && x.params?.lambda_max === v && x.metrics);
    return r ? P(r.metrics.rel_gap) : null;
  };
  const series = scens.value.map((s) => ({ type: "line", name: s.toUpperCase(), color: SCEN_COLOR[s], symbolSize: 7, data: vals.map((v) => val(v, s)) }));
  if (js) series.push({ type: "line", name: t("平均", "mean"), color: "#171c2b", lineStyle: { width: 3 }, symbolSize: 9, data: vals.map((v) => P(js.score[String(v)])) });
  const ch = lamMax(l);
  const idx = vals.indexOf(ch);
  if (idx >= 0) series[0].markLine = { silent: true, symbol: "unconstraint", lineStyle: { color: "#4f46e5", type: "solid", width: 2, opacity: 0.35 },
    label: { formatter: t("选定", "chosen"), position: "end" }, data: [{ xAxis: idx }] };
  return { grid: { ...grid, top: 34 }, legend: { top: 0, right: 0, itemWidth: 14, textStyle: { fontSize: 11 } }, tooltip: tip,
    xAxis: { type: "category", name: "λ_max", data: vals.map(String) }, yAxis: { ...axisPct(js ? t("有效 gap", "effective gap") : "gap"), scale: true }, series };
};

// ── sweep-only batch (v5.0.3): lambda and violations of every sweep run, one line per lambda_max ───────────────
const sweepOnly = computed(() => !main.value.length && sweepRuns.value.length > 0);
const sweepScen = ref("lt");
const VAL_COLOR = ["#4f46e5", "#d97706", "#059669", "#dc2626", "#0891b2"];
const sweepVals = (l) => [...new Set(sweepRuns.value.filter((r) => r.RL_algorithm === l).map((r) => r.params?.lambda_max))].sort((a, b) => a - b);
const sweepTraceOpt = (l, what) => {
  lang.value;
  const series = sweepVals(l).map((v, k) => {
    const r = sweepRuns.value.find((x) => x.RL_algorithm === l && x.scenario === sweepScen.value && x.params?.lambda_max === v);
    const tr = r && traces.value[r.exp_id];
    const S = tr?.series || {};
    const ys = what === "lambda" ? S.lambda : what === "ar" ? S.ar?.map(P)
      : S.cap_viol?.map((c, i) => (c == null ? null : +(c + (S.priv_viol?.[i] ?? 0)).toFixed(5)));
    return { type: "line", name: `λ_max ${v}`, color: VAL_COLOR[k % VAL_COLOR.length], showSymbol: false, data: tr ? tr.step.map((x, i) => [x, ys?.[i] ?? null]) : [] };
  });
  if (what === "jc") series.push({ type: "line", data: [], markLine: { silent: true, symbol: "unconstraint", lineStyle: { type: "dashed", color: "#059669" },
    label: { formatter: "d = 0.001", position: "insideEndTop" }, data: [{ yAxis: 0.001 }] } });
  const maxStep = Math.max(0, ...sweepRuns.value.map((r) => r.params?.steps || 0)) || undefined;
  const y = what === "lambda" ? { type: "value", name: "λ", min: 0 } : what === "ar" ? { ...axisPct(""), scale: true }
    : { type: "log", name: "J_C", min: 1e-4, axisLabel: { formatter: (v) => (v >= 0.01 ? v : v.toExponential(0)) } };
  return { grid, legend: { top: 0, right: 0, itemWidth: 14, textStyle: { fontSize: 11 } }, tooltip: tip, xAxis: stepAxis(maxStep), yAxis: y, series };
};

// ── 6. time vs ILP ───────────────────────────────────────────────────────────
const timing = computed(() => props.data?.report?.timing || []);
const ilpT = computed(() => props.data?.report?.ilp_timing || {});
const timeOpt = (s) => {
  lang.value;
  const rows = timing.value.filter((x) => x.scenario === s && x.key).map((x) => ({ ...x, lm: fromDir(x.dir) }))
    .sort((a, b) => b.median_ms - a.median_ms);
  const names = [...rows.map((x) => modelName(...x.lm)), "ILP"];
  const data = [...rows.map((x) => ({ value: +x.median_ms.toFixed(2), itemStyle: { color: STRATEGY_COLOR[x.lm[1]] } })),
    { value: +(ilpT.value[s]?.median_ms ?? 0).toFixed(2), itemStyle: { color: "#171c2b" } }];
  return { grid: { left: 150, right: 60, top: 10, bottom: 34 }, tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v) => `${v} ms` },
    xAxis: { type: "log", name: t("每实例中位数 ms（对数）", "median ms per instance (log)"), nameLocation: "middle", nameGap: 24 },
    yAxis: { type: "category", data: names, axisLabel: { fontSize: 11 } },
    series: [{ type: "bar", data, barMaxWidth: 14, label: { show: true, position: "right", fontSize: 10, formatter: (p) => `${p.value}` } }] };
};
const chartH = (n) => Math.max(160, 26 * n + 50);
</script>

<template>
  <div class="stack">
    <div v-if="sweepOnly" class="card">
      <div class="card-h">
        <h2>{{ t("扫描 run 的训练过程", "Sweep runs during training") }}</h2>
        <span class="faint small">{{ t("每条线一个 λ_max；训练局的滑动平均（窗口 1000 局）", "one line per λ_max; moving average over training episodes (1000)") }}</span>
        <span class="spacer" />
        <div class="seg"><button v-for="s in scens" :key="s" :class="{ on: sweepScen === s }" @click="sweepScen = s">{{ s.toUpperCase() }}</button></div>
      </div>
      <div class="card-b">
        <div v-if="loading" class="faint small" style="margin-bottom: 6px">{{ t("读取训练曲线…", "Loading training curves…") }} {{ Object.keys(traces).length }}</div>
        <div class="grid-3">
          <div v-for="l in sweepLearners" :key="l">
            <b>{{ l }} · lagrangian</b>
            <h3 style="margin-top: 6px">λ</h3><Chart :option="sweepTraceOpt(l, 'lambda')" :height="190" />
            <h3>{{ t("J_C：每局违约次数（对数轴）", "J_C: violations per episode (log axis)") }}</h3><Chart :option="sweepTraceOpt(l, 'jc')" :height="190" />
            <h3>{{ t("训练局 AR", "Training-episode AR") }}</h3><Chart :option="sweepTraceOpt(l, 'ar')" :height="190" />
          </div>
        </div>
      </div>
      <div class="card-note">{{ t("λ 顶到 λ_max 且 J_C 仍高于 d，说明该上限太低；不同 λ_max 下 λ 轨迹重合，说明上限没有起作用。", "λ pinned at λ_max while J_C stays above d means that cap is too low; identical λ curves for different λ_max mean the cap never binds.") }}</div>
    </div>

    <template v-if="!sweepOnly">
    <!-- legend shared by the model charts -->
    <div class="card"><div class="card-b row" style="gap: 18px">
      <b class="small">{{ t("图例", "Legend") }}</b>
      <span v-for="m in mechs" :key="m" class="small"><i class="sw" :style="{ background: STRATEGY_COLOR[m] }" />{{ m }}</span>
      <span class="faint">|</span>
      <span v-for="l in learners" :key="l" class="small"><span class="sym">{{ { circle: "●", triangle: "▲", rect: "■" }[RL_SYMBOL[l]] }}</span>{{ l }}</span>
      <span class="spacer" />
      <span class="faint small">{{ t("颜色 = strategy，形状 = RL_algorithm（取舍图）", "colour = strategy, shape = RL_algorithm (trade-off)") }}</span>
    </div></div>

    <div class="card">
      <div class="card-h"><h2>1 · {{ t("gap 与违约的取舍", "Gap vs violation trade-off") }}</h2><span class="faint small">{{ t("越靠左下越好；Maskable 落在 x = 0", "lower-left is better; Maskable sits at x = 0") }}</span></div>
      <div class="card-b">
        <div v-if="!doneMain.length" class="empty">{{ t("还没有完成的 run。", "No finished run yet.") }}</div>
        <div v-else class="grid-3">
          <div v-for="s in scens" :key="s"><h3>{{ scenName(s) }}</h3><Chart :option="tradeoff(s)" :height="300" /></div>
        </div>
      </div>
      <div class="card-note">{{ t("横轴：有容量违约的测试实例比例 + 有隐私违约的比例；纵轴：无违约实例上的相对 gap。", "x: share of test instances with a capacity violation + share with a privacy violation; y: relative gap on violation-free instances.") }}</div>
    </div>

    <div class="card">
      <div class="card-h">
        <h2>2 · {{ t("学习曲线（checkpoint 评估）", "Learning curves (checkpoint evaluation)") }}</h2>
        <span class="spacer" />
        <div class="seg"><button v-for="[k, n] in CURVE_METRICS" :key="k" :class="{ on: curveMetric === k }" @click="curveMetric = k">{{ n }}</button></div>
      </div>
      <div class="card-b">
        <div v-if="!curves.length" class="empty">{{ t("报告生成后显示（来自 report_data.json：每个 checkpoint 在测试集与训练集上的确定性评估）。", "Shown once the report is written (report_data.json: deterministic evaluation of every checkpoint on the test and training sets).") }}</div>
        <div v-else v-for="l in learners" :key="l" style="margin-bottom: 8px">
          <b>{{ l }}</b>
          <div class="grid-3"><div v-for="s in scens" :key="s"><h3>{{ scenName(s) }}</h3><Chart :option="curveOpt(l, s)" :height="220" /></div></div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-h">
        <h2>3 · {{ t("训练过程", "During training") }}</h2>
        <span class="faint small">{{ t("训练局的滑动平均", "moving average over training episodes") }}</span>
        <span class="spacer" />
        <div class="seg"><button v-for="[k, n] in TRACE_METRICS" :key="k" :class="{ on: traceMetric === k }" @click="traceMetric = k">{{ n }}</button></div>
        <select v-model.number="traceWindow" :title="t('滑动窗口（局）', 'window (episodes)')">
          <option :value="100">{{ t("窗口 100 局", "window 100") }}</option><option :value="1000">{{ t("窗口 1000 局", "window 1000") }}</option><option :value="10000">{{ t("窗口 10000 局", "window 10000") }}</option>
        </select>
      </div>
      <div class="card-b">
        <div v-if="loading" class="faint small" style="margin-bottom: 6px">{{ t("读取训练曲线…", "Loading training curves…") }} {{ Object.keys(traces).length }}</div>
        <div v-for="l in learners" :key="l" style="margin-bottom: 8px">
          <b>{{ l }}</b>
          <div class="grid-3"><div v-for="s in scens" :key="s"><h3>{{ scenName(s) }}</h3><Chart :option="traceOpt(l, s)" :height="220" /></div></div>
        </div>
      </div>
      <div class="card-note">{{ t("训练时的随机策略（PPO 采样、DQN ε-greedy），与测试集确定性评估不同；横轴为训练步数 0 – 总步数。", "Stochastic training policy (PPO sampling, DQN ε-greedy), unlike the deterministic test evaluation; x axis is 0 – total training steps.") }}</div>
    </div>

    <div v-if="learners.some((l) => lagRuns(l).length)" class="card">
      <div class="card-h"><h2>4 · {{ t("lagrangian：λ 与违约", "lagrangian: λ and violations") }}</h2><span class="faint small">{{ t("λ ← clip(λ + η(J_C − d), 0, λ_max)，按局更新", "λ ← clip(λ + η(J_C − d), 0, λ_max), per episode") }}</span></div>
      <div class="card-b">
        <div v-if="!hasLag" class="empty">{{ t("lagrangian 正式 run 开始训练后显示。", "Shown once the main lagrangian runs are training.") }}</div>
        <div v-else class="grid-3">
          <div v-for="l in learners.filter((x) => lagRuns(x).length)" :key="l">
            <b>{{ l }} · lagrangian</b>
            <h3 style="margin-top: 6px">λ</h3><Chart :option="lamOpt(l)" :height="200" />
            <h3>{{ t("J_C：每局违约次数（滑动平均，对数轴）", "J_C: violations per episode (moving average, log axis)") }}</h3><Chart :option="jcOpt(l)" :height="200" />
          </div>
        </div>
      </div>
      <div class="card-note">{{ t("λ 顶到 λ_max 且 J_C 仍高于 d，说明上限太低；λ 很快回到 0，说明约束已基本满足。", "λ pinned at λ_max while J_C stays above d means the cap is too low; λ falling back to 0 means the constraint is essentially met.") }}</div>
    </div>

    </template>

    <div v-if="sweepLearners.length" class="card">
      <div class="card-h"><h2>{{ sweepOnly ? "" : "5 · " }}{{ t("λ_max 扫描", "λ_max sweep") }}</h2><span class="faint small">{{ t("每个 RL_algorithm 各自选择", "chosen per RL_algorithm") }}</span></div>
      <div class="card-b grid-3">
        <div v-for="l in sweepLearners" :key="l"><b>{{ l }} · lagrangian</b><Chart :option="sweepOpt(l)" :height="230" /></div>
      </div>
      <div class="card-note">{{ t("有效 gap：有违约的实例按 AR = 0 计（来自 lam_sweep_<算法>.json）；没有该文件时显示无违约实例上的 gap。", "Effective gap: a violating instance counts as AR = 0 (from lam_sweep_<algorithm>.json); without it, the gap on violation-free instances.") }}</div>
    </div>

    <div v-if="!sweepOnly" class="card">
      <div class="card-h"><h2>6 · {{ t("推理时间与 ILP", "Inference time vs ILP") }}</h2><span class="faint small">{{ t("单线程 CPU，每实例中位数", "one CPU thread, median per instance") }}</span></div>
      <div class="card-b">
        <div v-if="!timing.length" class="empty">{{ t("最终报告生成后显示（计时只在最终报告里做）。", "Shown once the final report is written (timing is done for the final report only).") }}</div>
        <div v-else class="grid-3">
          <div v-for="s in scens" :key="s"><h3>{{ scenName(s) }}</h3><Chart :option="timeOpt(s)" :height="chartH(timing.filter((x) => x.scenario === s && x.key).length + 1)" /></div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.sym { margin-right: 5px; color: var(--dim); }
h3 { margin: 4px 0 0; }
</style>
