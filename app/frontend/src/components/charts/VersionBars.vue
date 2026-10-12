<script setup>
// One version, every model in its report: test relative gap per scenario (grouped horizontal bars, LT / EQ / GT
// in fixed colours), the three-scenario mean written at the end of the row, the 5% goal as a reference line.
import { ref, computed, watch } from "vue";
import { langRef as curLang } from "../../i18n.js";
import { SCEN_COLOR, TARGET, useWidth, niceMax, fmt1 } from "./common.js";

const props = defineProps({ reports: { type: Array, default: () => [] } });
const L = (zh, en) => (curLang.value === "en" ? en : zh);
const SC = ["LT", "EQ", "GT"];

const choices = computed(() => props.reports.filter((r) => (r.models?.length ?? 0) >= 2));
const version = ref(null);
watch(choices, (c) => { if (!version.value && c.length) version.value = c[0].version; }, { immediate: true });
const rep = computed(() => choices.value.find((r) => r.version === version.value) ?? null);
const en = computed(() => curLang.value === "en" && rep.value?.en);
const rows = computed(() => {
  const r = rep.value;
  if (!r) return [];
  const enModels = r.en?.models ?? [];
  return r.models.filter((m) => r.gaps?.[m]).map((m, i) => {
    const g = r.gaps[m];
    const vals = SC.map((s) => g[s]).filter((v) => v != null);
    return { key: m, name: en.value && enModels[i] ? enModels[i] : m, g, avg: vals.length ? vals.reduce((a, b) => a + b) / vals.length : null,
             isNew: m === r.new_model, isBase: m === r.base_model };
  });
});

const { el, width } = useWidth(320);
const LW = computed(() => Math.min(260, Math.max(140, width.value * 0.32)));   // label column
const BAR = 7, GAP = 2, ROW = 3 * BAR + 2 * GAP + 16;
const M = { t: 8, r: 64, b: 26 };
const H = computed(() => M.t + rows.value.length * ROW + M.b);
const xMax = computed(() => niceMax(Math.max(8, ...rows.value.flatMap((r) => SC.map((s) => r.g[s] ?? 0))), 4));
const xs = (v) => LW.value + (v / xMax.value) * (width.value - LW.value - M.r);
const ticks = computed(() => { const t = []; const st = xMax.value > 16 ? 4 : 2; for (let v = 0; v <= xMax.value; v += st) t.push(v); return t; });
const rowY = (i) => M.t + i * ROW + 8;
// bar with a 4px rounded data end, square at the baseline
const bar = (x0, y0, x1, h) => {
  const w = Math.max(0, x1 - x0), r = Math.min(4, w, h / 2);
  return `M${x0},${y0}H${x0 + w - r}Q${x0 + w},${y0} ${x0 + w},${y0 + r}V${y0 + h - r}Q${x0 + w},${y0 + h} ${x0 + w - r},${y0 + h}H${x0}Z`;
};
const hover = ref(null);
</script>

<template>
  <div class="vb" data-no-i18n>
    <div class="vb-controls">
      <label>{{ L("版本", "Version") }}
        <select v-model="version"><option v-for="r in choices" :key="r.version" :value="r.version">{{ r.version }}</option></select>
      </label>
      <span class="legend">
        <span v-for="s in SC" :key="s" class="legend-item"><span class="sw" :style="{ background: SCEN_COLOR[s] }"></span>{{ s }}</span>
        <span class="legend-item"><span class="sw sw--target"></span>{{ L("目标 5%", "goal 5%") }}</span>
        <span class="legend-item dim">{{ L("行末 = 三场景平均", "row end = 3-scenario mean") }}</span>
      </span>
    </div>
    <div v-if="rep" class="dim small vb-title">{{ en ? rep.en.title : rep.title }}</div>
    <div ref="el" class="vb-wrap">
      <svg v-if="rows.length" :width="width" :height="H" role="img" :aria-label="L('各模型三场景测试集相对 gap', 'Test relative gap of every model')"
        @mouseleave="hover = null">
        <g class="ax">
          <g v-for="t in ticks" :key="t">
            <line :x1="xs(t)" :x2="xs(t)" :y1="M.t" :y2="H - M.b" />
            <text :x="xs(t)" :y="H - 8" text-anchor="middle">{{ t }}%</text>
          </g>
        </g>
        <line class="target" :x1="xs(TARGET)" :x2="xs(TARGET)" :y1="M.t" :y2="H - M.b" />
        <g v-for="(r, i) in rows" :key="r.key" @mouseenter="hover = i">
          <rect class="hit" x="0" :y="rowY(i) - 6" :width="width" :height="ROW" />
          <text class="row-label" :class="{ strong: r.isNew }" :x="LW - 10" :y="rowY(i) + 13" text-anchor="end">
            {{ r.name.length > 34 ? r.name.slice(0, 33) + "…" : r.name }}</text>
          <path v-for="(s, k) in SC" :key="s" v-show="r.g[s] != null" :d="bar(LW, rowY(i) + k * (BAR + GAP), xs(r.g[s] ?? 0), BAR)" :fill="SCEN_COLOR[s]" />
          <text class="avg" :x="xs(Math.max(...SC.map((s) => r.g[s] ?? 0))) + 8" :y="rowY(i) + 13">{{ fmt1(r.avg) }}</text>
        </g>
      </svg>
      <div v-if="hover !== null && rows[hover]" class="tip" :style="{ top: rowY(hover) + ROW + 'px', left: LW + 'px' }">
        <div class="tip-head">{{ rows[hover].name }}<span v-if="rows[hover].isNew" class="dim"> · {{ L("本版本的新变体", "this version's new variant") }}</span>
          <span v-else-if="rows[hover].isBase" class="dim"> · {{ L("基线", "baseline") }}</span></div>
        <div v-for="s in SC" :key="s" class="tip-row"><span class="sw" :style="{ background: SCEN_COLOR[s] }"></span>{{ s }} <b>{{ fmt1(rows[hover].g[s]) }}</b></div>
        <div class="tip-row">{{ L("三场景平均", "3-scenario mean") }} <b>{{ fmt1(rows[hover].avg) }}</b></div>
      </div>
      <div v-if="!rows.length" class="empty">{{ L("没有可比较的模型", "No models to compare") }}</div>
    </div>
    <details class="vb-table">
      <summary class="small">{{ L("数据表", "Data table") }}</summary>
      <table class="grid"><thead><tr><th>{{ L("模型", "Model") }}</th><th v-for="s in SC" :key="s" class="num">{{ s }}</th><th class="num">{{ L("平均", "Mean") }}</th></tr></thead>
        <tbody><tr v-for="r in rows" :key="r.key"><td>{{ r.name }}</td><td v-for="s in SC" :key="s" class="num">{{ fmt1(r.g[s]) }}</td><td class="num"><b>{{ fmt1(r.avg) }}</b></td></tr></tbody>
      </table>
    </details>
  </div>
</template>

<style scoped>
.vb-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 16px; font-size: 13px; color: var(--text-dim); margin-bottom: 4px; }
.vb-controls select { margin-left: 6px; }
.vb-title { margin-bottom: 6px; }
.legend { display: inline-flex; flex-wrap: wrap; gap: 12px; font-size: 12px; }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.sw { display: inline-block; width: 12px; height: 8px; border-radius: 2px; }
.sw--target { width: 14px; height: 0; border-top: 2px dashed var(--ok); }
.vb-wrap { position: relative; width: 100%; }
.ax line { stroke: var(--border); stroke-width: 1; }
.ax text { font-size: 10px; fill: var(--text-faint); }
.target { stroke: var(--ok); stroke-width: 1.5; stroke-dasharray: 6 4; }
.row-label { font-size: 12px; fill: var(--text-dim); }
.row-label.strong { fill: var(--text); font-weight: 650; }
.avg { font-size: 12px; font-weight: 650; fill: var(--text); font-variant-numeric: tabular-nums; }
.hit { fill: transparent; }
g:hover > .hit { fill: var(--primary-soft); opacity: .5; }
.tip { position: absolute; background: var(--panel); border: 1px solid var(--border-strong); border-radius: 8px; box-shadow: 0 4px 16px rgba(0, 0, 0, .08);
  padding: 6px 8px; font-size: 12px; pointer-events: none; z-index: 3; min-width: 180px; }
.tip-head { font-weight: 600; margin-bottom: 4px; }
.tip-row { display: flex; align-items: center; gap: 6px; line-height: 1.6; }
.tip-row b { margin-left: auto; }
.vb-table { margin-top: 6px; }
</style>
