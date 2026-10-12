<script setup>
// Live SM-clock curve of one GPU (last `window` seconds), from /api/gpu_history samples.
// One series, one axis: the clock. Base / max clock are dashed reference lines; a status band under
// the plot marks the samples where the driver reported a slowdown reason. Hover: crosshair + tooltip.
import { computed, ref } from "vue";
import { langRef as curLang } from "../i18n.js";

const props = defineProps({
  samples: { type: Array, default: () => [] },   // [{t, clock, max, util, temp, power, thr}]
  window: { type: Number, default: 1800 },
  base: { type: Number, default: 1480 },         // GTX 1080 Ti base clock
});
const T = (zh, en) => (curLang.value === "en" ? en : zh);

const W = 360, H = 120, L = 38, R = 8, TOP = 8, PLOT_B = 92, BAND_Y = 98, BAND_H = 5;
const now = computed(() => (props.samples.length ? props.samples[props.samples.length - 1].t : Date.now() / 1000));
const maxClock = computed(() => props.samples.find((s) => s.max)?.max ?? 1911);
const yLo = computed(() => {
  const lo = Math.min(props.base, ...props.samples.map((s) => s.clock ?? props.base));
  return Math.floor((lo - 120) / 100) * 100;
});
const yHi = computed(() => Math.ceil((Math.max(maxClock.value, ...props.samples.map((s) => s.clock ?? 0)) + 40) / 100) * 100);
const x = (t) => L + ((t - (now.value - props.window)) / props.window) * (W - L - R);
const y = (v) => PLOT_B - ((v - yLo.value) / (yHi.value - yLo.value)) * (PLOT_B - TOP);

// break the line where samples are missing (panel restart, nvidia-smi hiccup)
const path = computed(() => {
  let d = "", prev = null;
  for (const s of props.samples) {
    if (s.clock == null) { prev = null; continue; }
    d += `${prev && s.t - prev.t < 10 ? "L" : "M"}${x(s.t).toFixed(1)},${y(s.clock).toFixed(1)}`;
    prev = s;
  }
  return d;
});
const throttled = computed(() => {
  const out = [];
  const ss = props.samples;
  for (let k = 0; k < ss.length; k++) {
    if (!ss[k].thr?.length) continue;
    const t1 = k + 1 < ss.length ? ss[k + 1].t : ss[k].t + 2;
    out.push({ x: x(ss[k].t), w: Math.max(1.5, x(t1) - x(ss[k].t)) });
  }
  return out;
});
const ticks = computed(() => {
  const step = (yHi.value - yLo.value) > 600 ? 200 : 100;
  const out = [];
  for (let v = Math.ceil(yLo.value / step) * step; v <= yHi.value; v += step) out.push(v);
  return out;
});
const last = computed(() => props.samples[props.samples.length - 1] ?? null);

const hover = ref(null);
const svgEl = ref(null);
function onMove(ev) {
  if (!props.samples.length || !svgEl.value) return;
  const r = svgEl.value.getBoundingClientRect();
  const px = ((ev.clientX - r.left) / r.width) * W;
  const t = now.value - props.window + ((px - L) / (W - L - R)) * props.window;
  let best = props.samples[0];
  for (const s of props.samples) if (Math.abs(s.t - t) < Math.abs(best.t - t)) best = s;
  hover.value = { s: best, left: (x(best.t) / W) * 100 };
}
const clock = (t) => new Date(t * 1000).toLocaleTimeString(curLang.value === "en" ? "en-GB" : "zh-CN", { hour12: false });
const thrText = (s) => (s.thr?.length ? s.thr.join(", ") : T("无", "none"));
</script>

<template>
  <div class="gcc">
    <div class="gcc-head">
      <span>{{ T("核心频率", "SM clock") }} · {{ T(`最近 ${window / 60} 分钟`, `last ${window / 60} min`) }}</span>
      <b v-if="last">{{ last.clock ?? "—" }} MHz</b>
    </div>
    <div class="gcc-plot" @mousemove="onMove" @mouseleave="hover = null">
      <svg ref="svgEl" :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="none" role="img"
        :aria-label="T('核心频率随时间变化', 'SM clock over time')">
        <g class="gcc-grid">
          <line v-for="v in ticks" :key="v" :x1="L" :x2="W - R" :y1="y(v)" :y2="y(v)" />
        </g>
        <g class="gcc-axis">
          <text v-for="v in ticks" :key="'t' + v" :x="L - 4" :y="y(v) + 3" text-anchor="end">{{ v }}</text>
          <text :x="L" :y="H - 2">−{{ window / 60 }} min</text>
          <text :x="W - R" :y="H - 2" text-anchor="end">{{ T("现在", "now") }}</text>
        </g>
        <g class="gcc-ref">
          <line :x1="L" :x2="W - R" :y1="y(maxClock)" :y2="y(maxClock)" />
          <text :x="W - R - 2" :y="y(maxClock) - 3" text-anchor="end">{{ T("最高", "max") }} {{ maxClock }}</text>
          <line :x1="L" :x2="W - R" :y1="y(base)" :y2="y(base)" />
          <text :x="W - R - 2" :y="y(base) - 3" text-anchor="end">{{ T("基础", "base") }} {{ base }}</text>
        </g>
        <rect :x="L" :y="BAND_Y" :width="W - L - R" :height="BAND_H" class="gcc-band-bg" />
        <rect v-for="(b, k) in throttled" :key="k" :x="b.x" :y="BAND_Y" :width="b.w" :height="BAND_H" class="gcc-band" />
        <path :d="path" class="gcc-line" vector-effect="non-scaling-stroke" />
        <line v-if="hover" :x1="x(hover.s.t)" :x2="x(hover.s.t)" :y1="TOP" :y2="PLOT_B" class="gcc-cross" vector-effect="non-scaling-stroke" />
      </svg>
      <div v-if="hover" class="gcc-tip" :style="{ left: Math.min(Math.max(hover.left, 18), 82) + '%' }">
        <div class="dim">{{ clock(hover.s.t) }}</div>
        <div>{{ T("频率", "clock") }} <b>{{ hover.s.clock ?? "—" }} MHz</b></div>
        <div>{{ T("利用率", "util") }} {{ hover.s.util ?? "—" }}% · {{ hover.s.temp ?? "—" }} °C · {{ hover.s.power?.toFixed(0) ?? "—" }} W</div>
        <div :class="hover.s.thr?.length ? 'gcc-warn' : ''">{{ T("降频", "throttle") }}：{{ thrText(hover.s) }}</div>
      </div>
    </div>
    <div class="gcc-legend dim small">
      <span class="gcc-key gcc-key--line"></span>{{ T("核心频率", "SM clock") }}
      <span class="gcc-key gcc-key--ref"></span>{{ T("基础 / 最高频率", "base / max clock") }}
      <span class="gcc-key gcc-key--band"></span>{{ T("降频（驱动报告）", "slowdown (driver-reported)") }}
    </div>
  </div>
</template>

<style scoped>
.gcc { margin-top: 10px; padding-top: 8px; border-top: 1px dashed var(--border); }
.gcc-head { display: flex; justify-content: space-between; font-size: 12px; color: var(--text-dim); margin-bottom: 4px; }
.gcc-head b { color: var(--text); font-variant-numeric: tabular-nums; }
.gcc-plot { position: relative; }
.gcc-plot svg { display: block; width: 100%; height: 120px; overflow: visible; }
.gcc-grid line { stroke: var(--border); stroke-width: 1; vector-effect: non-scaling-stroke; }
.gcc-axis text, .gcc-ref text { font-size: 9px; fill: var(--text-faint); font-variant-numeric: tabular-nums; }
.gcc-ref line { stroke: var(--text-faint); stroke-dasharray: 3 3; stroke-width: 1; vector-effect: non-scaling-stroke; }
.gcc-line { fill: none; stroke: var(--run); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.gcc-cross { stroke: var(--text-dim); stroke-width: 1; }
.gcc-band-bg { fill: var(--off-soft); }
.gcc-band { fill: var(--warn); }
.gcc-tip {
  position: absolute; top: 0; transform: translateX(-50%); pointer-events: none; z-index: 2;
  background: var(--panel); border: 1px solid var(--border-strong); border-radius: 6px; box-shadow: var(--shadow-card);
  padding: 6px 8px; font-size: 12px; line-height: 1.5; white-space: nowrap; color: var(--text);
}
.gcc-warn { color: var(--warn); font-weight: 600; }
.gcc-legend { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-top: 2px; }
.gcc-key { display: inline-block; width: 14px; margin-left: 6px; }
.gcc-key:first-child { margin-left: 0; }
.gcc-key--line { height: 2px; background: var(--run); }
.gcc-key--ref { height: 0; border-top: 1px dashed var(--text-faint); }
.gcc-key--band { height: 5px; background: var(--warn); }
</style>
