<script setup>
// Test relative gap of each version's new variant (LT / EQ / GT and the three-scenario mean), with the
// 5% target and the final-configuration baseline as reference lines. Ordinal x (versions, oldest first).
import { ref, computed, onMounted, onUnmounted } from "vue";
import { langRef as curLang } from "../i18n.js";

const props = defineProps({ reports: { type: Array, default: () => [] } });
const SERIES = [
  { key: "LT", color: "#2a78d6", width: 2 },
  { key: "EQ", color: "#eb6834", width: 2 },
  { key: "GT", color: "#1baf7a", width: 2 },
  { key: "avg", color: "var(--text)", width: 3 },
];
const TARGET = 5;
const showEarly = ref(false);
const hover = ref(null);
const wrap = ref(null);
const W = ref(760);
let ro = null;
onMounted(() => {
  ro = new ResizeObserver(([e]) => (W.value = Math.max(320, Math.floor(e.contentRect.width))));
  ro.observe(wrap.value);
});
onUnmounted(() => ro?.disconnect());

const vkey = (v) => v.replace(/^v/, "").split(".").map(Number);
const cmp = (a, b) => { const x = vkey(a), y = vkey(b); for (let i = 0; i < Math.max(x.length, y.length); i++) { const d = (x[i] ?? 0) - (y[i] ?? 0); if (d) return d; } return 0; };
const en = computed(() => curLang.value === "en");
const L = (zh, e) => (en.value ? e : zh);

const pts = computed(() =>
  props.reports
    .filter((r) => r.new_avg != null && (showEarly.value || cmp(r.version, "v4.4.3") >= 0))
    .sort((a, b) => cmp(a.version, b.version))
    .map((r) => ({ v: r.version, change: en.value ? r.en?.change : r.change, model: en.value ? r.en?.new_model : r.new_model,
                   LT: r.gaps[r.new_model].LT, EQ: r.gaps[r.new_model].EQ, GT: r.gaps[r.new_model].GT, avg: r.new_avg })));
// most common baseline mean among the reports (the final configuration)
const baseline = computed(() => {
  const c = {};
  for (const r of props.reports) if (r.base_avg != null) c[r.base_avg.toFixed(1)] = (c[r.base_avg.toFixed(1)] ?? 0) + 1;
  const best = Object.entries(c).sort((a, b) => b[1] - a[1])[0];
  return best ? Number(best[0]) : null;
});

const H = 300, M = { t: 16, r: 64, b: 34, l: 44 };
const yMax = computed(() => Math.ceil(Math.max(12, ...pts.value.flatMap((p) => [p.LT, p.EQ, p.GT, p.avg])) / 2) * 2);
const x = (i) => M.l + (pts.value.length <= 1 ? (W.value - M.l - M.r) / 2 : (i * (W.value - M.l - M.r)) / (pts.value.length - 1));
const y = (v) => M.t + (1 - v / yMax.value) * (H - M.t - M.b);
const ticks = computed(() => { const t = []; for (let v = 0; v <= yMax.value; v += yMax.value > 14 ? 4 : 2) t.push(v); return t; });
const path = (k) => pts.value.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p[k]).toFixed(1)}`).join("");
const colW = computed(() => (pts.value.length > 1 ? (W.value - M.l - M.r) / (pts.value.length - 1) : W.value));
const label = (k) => (k === "avg" ? L("三场景平均", "3-scenario mean") : k);
// end labels: nudge apart so they do not overlap
const endLabels = computed(() => {
  if (!pts.value.length) return [];
  const last = pts.value[pts.value.length - 1];
  const ls = SERIES.map((s) => ({ ...s, v: last[s.key], y: y(last[s.key]) })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ls.length; i++) if (ls[i].y - ls[i - 1].y < 13) ls[i].y = ls[i - 1].y + 13;
  return ls;
});
const go = (v) => (window.location.hash = `#/versions/${v}`);
</script>

<template>
  <div class="gap-trend" data-no-i18n>
    <div class="trend-head">
      <div class="legend">
        <span v-for="s in SERIES" :key="s.key" class="legend-item">
          <span class="legend-line" :style="{ background: s.color, height: s.width + 'px' }"></span>{{ label(s.key) }}
        </span>
        <span class="legend-item"><span class="legend-line legend-line--target"></span>{{ L("目标 5%", "target 5%") }}</span>
        <span v-if="baseline" class="legend-item"><span class="legend-line legend-line--base"></span>{{ L(`定稿基线 ${baseline}%`, `final baseline ${baseline}%`) }}</span>
      </div>
      <label class="small dim"><input v-model="showEarly" type="checkbox" /> {{ L("显示 MLP 时期（v4.4.3 之前）", "show the MLP era (before v4.4.3)") }}</label>
    </div>
    <div ref="wrap" class="trend-wrap">
      <svg v-if="pts.length" :width="W" :height="H" role="img"
        :aria-label="L('各版本测试集相对 gap 走势', 'Test relative gap by version')" @mouseleave="hover = null">
        <g class="grid-lines">
          <g v-for="t in ticks" :key="t">
            <line :x1="M.l" :x2="W - M.r" :y1="y(t)" :y2="y(t)" />
            <text :x="M.l - 8" :y="y(t) + 4" text-anchor="end">{{ t }}%</text>
          </g>
        </g>
        <line class="ref-target" :x1="M.l" :x2="W - M.r" :y1="y(TARGET)" :y2="y(TARGET)" />
        <line v-if="baseline" class="ref-base" :x1="M.l" :x2="W - M.r" :y1="y(baseline)" :y2="y(baseline)" />
        <text v-for="(p, i) in pts" :key="'x' + p.v" class="x-label" :x="x(i)" :y="H - 12" text-anchor="middle">{{ p.v.replace(/^v4\.4\./, "4.4.") }}</text>
        <line v-if="hover !== null" class="crosshair" :x1="x(hover)" :x2="x(hover)" :y1="M.t" :y2="H - M.b" />
        <path v-for="s in SERIES" :key="'p' + s.key" :d="path(s.key)" fill="none" :stroke="s.color" :stroke-width="s.width"
          stroke-linejoin="round" stroke-linecap="round" />
        <g v-for="s in SERIES" :key="'m' + s.key">
          <circle v-for="(p, i) in pts" :key="p.v" :cx="x(i)" :cy="y(p[s.key])" :r="hover === i ? 5 : 4"
            :fill="s.color" stroke="var(--panel)" stroke-width="2" />
        </g>
        <text v-for="e in endLabels" :key="'e' + e.key" class="end-label" :x="W - M.r + 8" :y="e.y + 4">{{ label(e.key) }} {{ e.v.toFixed(1) }}%</text>
        <rect v-for="(p, i) in pts" :key="'h' + p.v" class="hit" :x="x(i) - colW / 2" :y="M.t" :width="colW" :height="H - M.t - M.b"
          @mouseenter="hover = i" @click="go(p.v)" />
      </svg>
      <div v-else class="empty">{{ L("还没有可画的版本", "No versions to plot yet") }}</div>
      <div v-if="hover !== null && pts[hover]" class="trend-tip"
        :style="{ left: Math.min(x(hover) + 12, W - 300) + 'px', top: '8px' }">
        <div class="tip-title">{{ pts[hover].v }} <span class="dim">· {{ pts[hover].model || L("新变体", "new variant") }}</span></div>
        <div class="tip-change">{{ pts[hover].change || L("（见报告）", "(see the report — Chinese only)") }}</div>
        <table>
          <tr v-for="s in SERIES" :key="s.key"><td><span class="tip-dot" :style="{ background: s.color }"></span>{{ label(s.key) }}</td>
            <td class="num">{{ pts[hover][s.key].toFixed(1) }}%</td></tr>
        </table>
        <div class="dim small">{{ L("点击查看该版本报告", "click to open the report") }}</div>
      </div>
    </div>
    <details class="trend-table">
      <summary class="small">{{ L("数据表", "Data table") }}</summary>
      <table class="grid">
        <thead><tr><th>{{ L("版本", "Version") }}</th><th class="num">LT</th><th class="num">EQ</th><th class="num">GT</th><th class="num">{{ label("avg") }}</th></tr></thead>
        <tbody><tr v-for="p in [...pts].reverse()" :key="p.v"><td><a :href="`#/versions/${p.v}`">{{ p.v }}</a></td>
          <td class="num">{{ p.LT.toFixed(1) }}%</td><td class="num">{{ p.EQ.toFixed(1) }}%</td><td class="num">{{ p.GT.toFixed(1) }}%</td><td class="num"><b>{{ p.avg.toFixed(1) }}%</b></td></tr></tbody>
      </table>
    </details>
  </div>
</template>

<style scoped>
.trend-head { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 6px; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: var(--text-dim); }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.legend-line { display: inline-block; width: 18px; border-radius: 2px; }
.legend-line--target { height: 0; border-top: 2px dashed var(--ok); }
.legend-line--base { height: 0; border-top: 2px dotted var(--text-faint); }
.trend-wrap { position: relative; width: 100%; overflow: hidden; }
.grid-lines line { stroke: var(--border); stroke-width: 1; }
.grid-lines text, .x-label { font-size: 11px; fill: var(--text-faint); }
.ref-target { stroke: var(--ok); stroke-width: 1.5; stroke-dasharray: 6 4; }
.ref-base { stroke: var(--text-faint); stroke-width: 1.5; stroke-dasharray: 2 3; }
.crosshair { stroke: var(--border-strong); stroke-width: 1; }
.end-label { font-size: 11px; fill: var(--text); }
.hit { fill: transparent; cursor: pointer; }
.trend-tip { position: absolute; width: 280px; background: var(--panel); border: 1px solid var(--border-strong); border-radius: 8px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, .08); padding: 8px 10px; font-size: 12px; pointer-events: none; }
.tip-title { font-weight: 600; margin-bottom: 2px; }
.tip-change { color: var(--text-dim); margin-bottom: 6px; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.trend-tip table { width: 100%; border-collapse: collapse; }
.trend-tip td { padding: 1px 0; } .trend-tip td.num { text-align: right; font-variant-numeric: tabular-nums; }
.tip-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.trend-table { margin-top: 8px; }
</style>
