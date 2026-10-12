<script setup>
// Learning curves of one version (parsed from its report by /api/curves): one panel per scenario (+ the
// three-scenario mean when the report has it), shared y axis, mean line with a +-std band, the 5% goal.
// Many series (v5.0.2: 4 mechanisms x 3 learners) -> one row of panels per learner, colour = mechanism.
import { ref, computed, watch } from "vue";
import { langRef as curLang } from "../../i18n.js";
import { CAT, TARGET, useWidth, niceMax, fmt1, learnerOf, mechSlot, MECH_NAMES } from "./common.js";

const props = defineProps({ versions: { type: Array, default: () => [] } });
const L = (zh, en) => (curLang.value === "en" ? en : zh);
const METRICS = { gap: ["测试集相对 gap", "Test relative gap"], train_gap: ["训练集相对 gap", "Training-set relative gap"],
  fail: ["测试未通过率（EXIT / 违约）", "Test failure rate (EXIT / violation)"] };
const SCEN_ORDER = ["LT", "EQ", "GT", "AVG"];
const scenName = (s) => (s === "AVG" ? L("三场景平均", "3-scenario mean") : s);

const version = ref(null);
const metric = ref("gap");
const hidden = ref(new Set());
watch(() => props.versions, (vs) => { if (!version.value && vs.length) version.value = vs[0].version; }, { immediate: true });
const cur = computed(() => props.versions.find((v) => v.version === version.value) ?? null);
const metrics = computed(() => Object.keys(METRICS).filter((m) => cur.value?.curves?.[m]?.length));
watch(cur, () => { if (!metrics.value.includes(metric.value)) metric.value = metrics.value[0] ?? "gap"; hidden.value = new Set(); });
const curves = computed(() => cur.value?.curves?.[metric.value] ?? []);

// series and their colours
const labels = computed(() => [...new Set(curves.value.map((c) => c.label))]);
const grouped = computed(() => labels.value.length > 6 && labels.value.every((s) => learnerOf(s) && mechSlot(s) !== null));
const color = (lab) => (grouped.value ? CAT[mechSlot(lab)] : CAT[labels.value.indexOf(lab) % CAT.length]);
const rows = computed(() => (grouped.value ? ["PPO", "DQN", "DDQN"].filter((g) => labels.value.some((s) => learnerOf(s) === g)) : [null]));
const scens = computed(() => SCEN_ORDER.filter((s) => curves.value.some((c) => c.scen === s)));
const legend = computed(() => (grouped.value
  ? MECH_NAMES.map(([zh, en], k) => ({ key: `m${k}`, name: L(zh, en), color: CAT[k], match: (s) => mechSlot(s) === k }))
      .filter((m) => labels.value.some(m.match))
  : labels.value.map((s) => ({ key: s, name: s, color: color(s), match: (x) => x === s }))));
const visible = (lab) => !legend.value.some((m) => m.match(lab) && hidden.value.has(m.key));
const toggle = (k) => { const h = new Set(hidden.value); h.has(k) ? h.delete(k) : h.add(k); hidden.value = h; };
const panelSeries = (row, scen) => curves.value.filter((c) => c.scen === scen && (row === null || learnerOf(c.label) === row) && visible(c.label));

// geometry: shared axes
const { el, width } = useWidth(300);
const cols = computed(() => Math.max(1, Math.min(scens.value.length, Math.floor((width.value + 16) / 300))));
const PW = computed(() => Math.floor((width.value - 16 * (cols.value - 1)) / cols.value));
const PH = 210, M = { t: 14, r: 12, b: 30, l: 42 };
const xMax = computed(() => Math.max(1, ...curves.value.flatMap((c) => c.points.map((p) => p[0]))));
const yMax = computed(() => niceMax(Math.max(metric.value === "fail" ? 2 : 6, ...curves.value.filter((c) => visible(c.label))
  .flatMap((c) => c.points.map((p) => p[1] + (p[2] ?? 0)))), metric.value === "fail" ? 1 : 2));
const xs = (v) => M.l + (v / xMax.value) * (PW.value - M.l - M.r);
const ys = (v) => M.t + (1 - v / yMax.value) * (PH - M.t - M.b);
const yTicks = computed(() => { const st = yMax.value > 16 ? 4 : yMax.value > 8 ? 2 : yMax.value > 4 ? 1 : 0.5; const t = []; for (let v = 0; v <= yMax.value + 1e-9; v += st) t.push(+v.toFixed(2)); return t; });
const xTicks = computed(() => [...new Set(curves.value.flatMap((c) => c.points.map((p) => p[0])))].sort((a, b) => a - b));
// tick labels that keep >= 30 px apart (always the last one); the ticks themselves stay
const xLabels = computed(() => {
  const out = [];
  for (const t of [...xTicks.value].reverse()) if (!out.length || xs(out[out.length - 1]) - xs(t) >= 30) out.push(t);
  return out;
});
const line = (c) => c.points.map((p, i) => `${i ? "L" : "M"}${xs(p[0]).toFixed(1)},${ys(p[1]).toFixed(1)}`).join("");
const band = (c) => {
  if (!c.points.some((p) => p[2] != null)) return "";
  const up = c.points.map((p) => [xs(p[0]), ys(p[1] + (p[2] ?? 0))]);
  const dn = c.points.map((p) => [xs(p[0]), ys(Math.max(0, p[1] - (p[2] ?? 0)))]).reverse();
  return [...up, ...dn].map(([a, b], i) => `${i ? "L" : "M"}${a.toFixed(1)},${b.toFixed(1)}`).join("") + "Z";
};
const fmtStep = (v) => `${+v.toFixed(1)}M`;

// hover: nearest checkpoint in one panel
const hover = ref(null);   // {row, scen, step}
function onMove(ev, row, scen) {
  const r = ev.currentTarget.getBoundingClientRect();
  const v = ((ev.clientX - r.left) - M.l) / (PW.value - M.l - M.r) * xMax.value;
  const step = xTicks.value.reduce((a, b) => (Math.abs(b - v) < Math.abs(a - v) ? b : a), xTicks.value[0]);
  hover.value = { row, scen, step };
}
const tipRows = computed(() => {
  if (!hover.value) return [];
  return panelSeries(hover.value.row, hover.value.scen)
    .map((c) => ({ c, p: c.points.find((p) => p[0] === hover.value.step) }))
    .filter((x) => x.p).sort((a, b) => a.p[1] - b.p[1]);
});
const tableRows = computed(() => curves.value.slice().sort((a, b) => SCEN_ORDER.indexOf(a.scen) - SCEN_ORDER.indexOf(b.scen)));
</script>

<template>
  <div class="lc" data-no-i18n>
    <div class="lc-controls">
      <label>{{ L("版本", "Version") }}
        <select v-model="version">
          <option v-for="v in versions" :key="v.version" :value="v.version">{{ v.version }}{{ v.interim ? L("（中间报告）", " (interim)") : "" }}</option>
        </select>
      </label>
      <div v-if="metrics.length > 1" class="seg" role="group">
        <button v-for="m in metrics" :key="m" :class="{ on: metric === m }" @click="metric = m">{{ L(...METRICS[m]) }}</button>
      </div>
    </div>
    <div v-if="cur?.title" class="dim small lc-title">{{ cur.title }}</div>
    <div class="legend">
      <button v-for="m in legend" :key="m.key" class="chip" :class="{ off: hidden.has(m.key) }" @click="toggle(m.key)"
        :title="L('点击显示 / 隐藏', 'click to show / hide')">
        <span class="sw" :style="{ background: m.color }"></span>{{ m.name }}
      </button>
      <span v-if="metric !== 'fail'" class="chip static"><span class="sw sw--target"></span>{{ L("目标 5%", "goal 5%") }}</span>
      <span v-if="curves.some((c) => c.points.some((p) => p[2] != null))" class="chip static"><span class="sw sw--band"></span>{{ L("± 种子间标准差", "± std over seeds") }}</span>
    </div>

    <div ref="el" class="lc-body">
      <div v-for="row in rows" :key="row ?? 'all'" class="lc-row">
        <div v-if="row" class="row-label">{{ row }}</div>
        <div class="lc-grid" :style="{ gridTemplateColumns: `repeat(${cols}, ${PW}px)` }">
          <div v-for="scen in scens" :key="scen" class="lc-panel">
            <div class="panel-title">{{ scenName(scen) }}</div>
            <svg :width="PW" :height="PH" role="img" :aria-label="`${scenName(scen)} ${L(...METRICS[metric])}`"
              @mousemove="onMove($event, row, scen)" @mouseleave="hover = null">
              <g class="ax">
                <g v-for="t in yTicks" :key="'y' + t">
                  <line :x1="M.l" :x2="PW - M.r" :y1="ys(t)" :y2="ys(t)" />
                  <text :x="M.l - 6" :y="ys(t) + 4" text-anchor="end">{{ t }}%</text>
                </g>
                <line v-for="t in xTicks" :key="'xt' + t" class="tick" :x1="xs(t)" :x2="xs(t)" :y1="PH - M.b" :y2="PH - M.b + 4" />
                <text v-for="t in xLabels" :key="'x' + t" :x="xs(t)" :y="PH - 10" text-anchor="middle">{{ fmtStep(t) }}</text>
              </g>
              <line v-if="metric !== 'fail' && TARGET <= yMax" class="target" :x1="M.l" :x2="PW - M.r" :y1="ys(TARGET)" :y2="ys(TARGET)" />
              <path v-for="c in panelSeries(row, scen)" :key="'b' + c.label" :d="band(c)" :fill="color(c.label)" class="band" />
              <line v-if="hover && hover.row === row && hover.scen === scen" class="cross" :x1="xs(hover.step)" :x2="xs(hover.step)" :y1="M.t" :y2="PH - M.b" />
              <path v-for="c in panelSeries(row, scen)" :key="'l' + c.label" :d="line(c)" :stroke="color(c.label)" class="ln" />
              <g v-for="c in panelSeries(row, scen)" :key="'p' + c.label">
                <circle v-for="p in c.points" :key="p[0]" :cx="xs(p[0])" :cy="ys(p[1])" r="4" :fill="color(c.label)" class="pt" />
              </g>
            </svg>
            <div v-if="hover && hover.row === row && hover.scen === scen && tipRows.length" class="tip"
              :style="{ left: Math.min(xs(hover.step) + 10, PW - 230) + 'px' }">
              <div class="tip-head">{{ scenName(scen) }}{{ row ? ` · ${row}` : "" }} · {{ fmtStep(hover.step) }}</div>
              <div v-for="x in tipRows" :key="x.c.label" class="tip-row">
                <span class="sw" :style="{ background: color(x.c.label) }"></span>
                <span class="tip-name">{{ x.c.label }}</span>
                <b>{{ fmt1(x.p[1]) }}</b><span v-if="x.p[2] != null" class="dim"> ± {{ x.p[2].toFixed(1) }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div v-if="!curves.length" class="empty">{{ L("这个版本的报告里没有学习曲线", "No learning curve in this version's report") }}</div>
    </div>

    <details class="lc-table">
      <summary class="small">{{ L("数据表", "Data table") }}</summary>
      <div class="table-scroll"><table class="grid">
        <thead><tr><th>{{ L("场景", "Scenario") }}</th><th>{{ L("模型", "Model") }}</th><th v-for="t in xTicks" :key="t" class="num">{{ fmtStep(t) }}</th></tr></thead>
        <tbody><tr v-for="c in tableRows" :key="c.scen + c.label"><td>{{ scenName(c.scen) }}</td><td>{{ c.label }}</td>
          <td v-for="t in xTicks" :key="t" class="num">{{ fmt1(c.points.find((p) => p[0] === t)?.[1]) }}</td></tr></tbody>
      </table></div>
    </details>
  </div>
</template>

<style scoped>
.lc-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-bottom: 6px; font-size: 13px; color: var(--text-dim); }
.lc-controls select { margin-left: 6px; }
.lc-title { margin-bottom: 8px; }
.seg { display: inline-flex; border: 1px solid var(--border-strong); border-radius: 6px; overflow: hidden; }
.seg button { border: 0; background: var(--panel); padding: 4px 10px; font-size: 12px; color: var(--text-dim); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border-strong); }
.seg button.on { background: var(--primary-soft); color: var(--primary); font-weight: 600; }
.legend { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.chip { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text); background: var(--panel);
  border: 1px solid var(--border); border-radius: 999px; padding: 2px 10px; cursor: pointer; }
.chip.off { opacity: .4; }
.chip.static { cursor: default; color: var(--text-dim); }
.sw { display: inline-block; width: 14px; height: 3px; border-radius: 2px; }
.sw--target { height: 0; border-top: 2px dashed var(--ok); }
.sw--band { height: 8px; background: var(--text-faint); opacity: .25; }
.lc-row { display: flex; gap: 8px; align-items: flex-start; margin-bottom: 8px; }
.row-label { writing-mode: vertical-rl; transform: rotate(180deg); font-weight: 650; font-size: 13px; color: var(--text-dim); padding: 30px 0; }
.lc-grid { display: grid; gap: 16px; }
.lc-panel { position: relative; }
.panel-title { font-size: 12px; font-weight: 650; color: var(--text-dim); margin-bottom: 2px; }
.ax line { stroke: var(--border); stroke-width: 1; }
.ax .tick { stroke: var(--border-strong); }
.ax text { font-size: 10px; fill: var(--text-faint); font-variant-numeric: tabular-nums; }
.target { stroke: var(--ok); stroke-width: 1.5; stroke-dasharray: 6 4; }
.band { opacity: .12; }
.ln { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.pt { stroke: var(--panel); stroke-width: 2; }
.cross { stroke: var(--border-strong); stroke-width: 1; }
.tip { position: absolute; top: 22px; width: 220px; background: var(--panel); border: 1px solid var(--border-strong); border-radius: 8px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, .08); padding: 6px 8px; font-size: 12px; pointer-events: none; z-index: 3; }
.tip-head { font-weight: 600; margin-bottom: 4px; }
.tip-row { display: flex; align-items: center; gap: 6px; line-height: 1.6; }
.tip-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-dim); }
.lc-table { margin-top: 6px; }
</style>
