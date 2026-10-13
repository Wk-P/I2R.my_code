<script setup>
// One GPU: current state, then the last `window` seconds of its SM clock (one line; base / max clock as dashed
// reference lines; samples with a driver-reported slowdown reason shaded) and of its temperature and power.
// Same reading as the classic panel's clock chart: the y axis spans only the clock range in use.
import { computed } from "vue";
import { t, lang } from "../i18n.js";
import Chart from "./Chart.vue";

const props = defineProps({
  gpu: { type: Object, required: true },             // current state (/api/v2 system gpus[])
  samples: { type: Array, default: () => [] },      // [{t, clock, max, util, temp, power, thr}] of this GPU
  window: { type: Number, default: 1800 },
  base: { type: Number, default: 1480 },            // GTX 1080 Ti base clock
});
const THR = { 2: ["应用时钟设置", "applications clocks"], 4: ["软件功耗墙", "SW power cap"], 8: ["硬件降频", "HW slowdown"],
  16: ["同步加速", "sync boost"], 32: ["软件热降频（温度目标）", "SW thermal slowdown"], 64: ["硬件热降频", "HW thermal slowdown"],
  128: ["硬件电源制动", "HW power brake"], 256: ["显示时钟设置", "display clock"] };
const thrText = (bits) => (bits?.length ? bits.map((b) => t(...(THR[b] ?? [String(b), String(b)]))).join("、") : t("无", "none"));

const now = computed(() => (props.samples.length ? props.samples[props.samples.length - 1].t : Date.now() / 1000));
const maxClock = computed(() => props.samples.find((s) => s.max)?.max ?? props.gpu.clock_sm_max ?? 1911);
const xAxis = computed(() => ({ type: "time", min: (now.value - props.window) * 1000, max: now.value * 1000, splitNumber: 6,
  axisLabel: { formatter: "{HH}:{mm}", fontSize: 11, hideOverlap: true }, splitLine: { show: false } }));
// a gap of more than 10 s (restart, nvidia-smi hiccup) breaks the line
const pts = (key) => {
  const out = [];
  let prev = null;
  for (const s of props.samples) {
    if (prev && s.t - prev.t > 10) out.push([(prev.t + 1) * 1000, null]);
    out.push([s.t * 1000, s[key] ?? null]);
    prev = s;
  }
  return out;
};
// slowdown spans: consecutive throttled samples merged into one shaded band
const bands = computed(() => {
  const out = [];
  const ss = props.samples;
  for (let k = 0; k < ss.length; k++) {
    if (!ss[k].thr?.length) continue;
    const t0 = ss[k].t, t1 = k + 1 < ss.length ? ss[k + 1].t : ss[k].t + 2;
    const last = out[out.length - 1];
    if (last && last[1].xAxis >= t0 * 1000 - 1) last[1].xAxis = t1 * 1000;
    else out.push([{ xAxis: t0 * 1000 }, { xAxis: t1 * 1000 }]);
  }
  return out;
});
const tipAt = (ms) => props.samples.reduce((b, s) => (Math.abs(s.t * 1000 - ms) < Math.abs(b.t * 1000 - ms) ? s : b), props.samples[0]);
const time = (ms) => new Date(ms).toTimeString().slice(0, 8);

const clockOpt = computed(() => {
  lang.value;
  const vals = props.samples.map((s) => s.clock).filter((v) => v != null);
  const lo = Math.floor((Math.min(props.base, ...vals) - 120) / 100) * 100;
  const hi = Math.ceil((Math.max(maxClock.value, ...vals) + 40) / 100) * 100;
  return {
    grid: { left: 46, right: 12, top: 12, bottom: 26 },
    tooltip: { trigger: "axis", confine: true, axisPointer: { type: "line" }, formatter: (ps) => {
      const s = tipAt(ps[0].value[0]);
      if (!s) return "";
      return `${time(s.t * 1000)}<br>${t("频率", "clock")} <b>${s.clock ?? "—"} MHz</b><br>`
        + `${t("利用率", "util")} ${s.util ?? "—"}% · ${s.temp ?? "—"} °C · ${s.power?.toFixed(0) ?? "—"} W<br>`
        + `<span style="color:${s.thr?.length ? "#b45309" : "inherit"}">${t("降频", "slowdown")}：${thrText(s.thr)}</span>`;
    } },
    xAxis: xAxis.value,
    yAxis: { type: "value", min: lo, max: hi, interval: hi - lo > 600 ? 200 : 100, axisLabel: { fontSize: 11 } },
    series: [{
      type: "line", name: t("核心频率", "SM clock"), color: "#4f46e5", showSymbol: false, lineStyle: { width: 2 }, data: pts("clock"),
      markLine: { silent: true, symbol: "none", lineStyle: { type: "dashed", color: "#94a3b8", width: 1 },
        label: { position: "insideEndTop", fontSize: 10, color: "#64748b" },
        data: [{ yAxis: maxClock.value, label: { formatter: `${t("最高", "max")} ${maxClock.value}` } },
               { yAxis: props.base, label: { formatter: `${t("基础", "base")} ${props.base}` } }] },
      markArea: { silent: true, itemStyle: { color: "rgba(217, 119, 6, 0.18)" }, data: bands.value },
    }],
  };
});
// utilisation, temperature and power: one small chart each, with fixed readable ranges
const small = (key, name, color, yAxis, unit) => ({
  grid: { left: 40, right: 8, top: 10, bottom: 24 },
  tooltip: { trigger: "axis", confine: true, valueFormatter: (v) => (v == null ? "—" : `${Math.round(v)} ${unit}`) },
  xAxis: { ...xAxis.value, splitNumber: 1, axisLabel: { formatter: "{HH}:{mm}", fontSize: 10, hideOverlap: true, showMinLabel: true, showMaxLabel: true } },
  yAxis: { type: "value", axisLabel: { fontSize: 10 }, ...yAxis },
  series: [{ type: "line", name, color, showSymbol: false, lineStyle: { width: 1.8 }, data: pts(key) }],
});
const utilOpt = computed(() => { lang.value; return small("util", t("利用率", "util"), "#4f46e5", { min: 0, max: 100, interval: 50 }, "%"); });
const tempOpt = computed(() => {
  lang.value;
  const temps = props.samples.map((s) => s.temp).filter((v) => v != null);
  const hi = Math.max(90, Math.ceil((Math.max(...temps, 0) + 5) / 10) * 10);
  return small("temp", t("温度", "temp"), "#dc2626", { min: Math.min(hi - 40, Math.floor((Math.min(...temps, hi) - 5) / 10) * 10), max: hi, interval: 20 }, "°C");
});
const powerOpt = computed(() => {
  lang.value;
  const hi = Math.ceil((props.gpu.power_limit || 250) / 50) * 50;
  return small("power", t("功耗", "power"), "#0891b2", { min: 0, max: hi, interval: hi / 2 }, "W");
});
const g = computed(() => props.gpu);
const miniH = window.matchMedia?.("(max-width: 700px)").matches ? 80 : 110;   // phones: the three stack, keep them short
</script>

<template>
  <div class="card gpu">
    <div class="card-h">
      <h2>GPU {{ g.index }}</h2><span class="faint small">{{ g.name?.replace("NVIDIA GeForce ", "") }}</span>
      <span class="spacer" />
      <b class="mono">{{ g.clock_sm?.toFixed(0) ?? "—" }} MHz</b>
    </div>
    <div class="card-b">
      <div class="stats">
        <div><span>{{ t("利用率", "Util") }}</span><b>{{ g.util?.toFixed(0) ?? "—" }}%</b></div>
        <div><span>{{ t("显存", "Memory") }}</span><b>{{ ((g.mem_used || 0) / 1024).toFixed(1) }}/{{ ((g.mem_total || 0) / 1024).toFixed(0) }} GB</b></div>
        <div><span>{{ t("温度", "Temp") }}</span><b>{{ g.temp?.toFixed(0) ?? "—" }} °C</b></div>
        <div><span>{{ t("功耗", "Power") }}</span><b>{{ g.power?.toFixed(0) ?? "—" }}/{{ g.power_limit?.toFixed(0) ?? "—" }} W</b></div>
        <div><span>{{ t("进程", "Procs") }}</span><b>{{ g.procs }}</b></div>
      </div>
      <div class="thr" :class="{ on: g.throttle?.length }">{{ t("降频", "Slowdown") }}：{{ g.throttle?.length ? g.throttle.map((x) => t(x.zh, x.en)).join("、") : t("无", "none") }}</div>
      <h3>{{ t("核心频率（MHz）", "SM clock (MHz)") }} · {{ t(`最近 ${window / 60} 分钟`, `last ${window / 60} min`) }}</h3>
      <Chart :option="clockOpt" :height="190" />
      <div class="legend small faint">
        <span><i class="k line" />{{ t("核心频率", "SM clock") }}</span>
        <span><i class="k ref" />{{ t("基础 / 最高频率", "base / max clock") }}</span>
        <span><i class="k band" />{{ t("降频（驱动报告）", "slowdown (driver-reported)") }}</span>
      </div>
      <div class="minis">
        <div><h3>{{ t("利用率 %", "Utilisation %") }}</h3><Chart :option="utilOpt" :height="miniH" /></div>
        <div><h3>{{ t("温度 °C", "Temperature °C") }}</h3><Chart :option="tempOpt" :height="miniH" /></div>
        <div><h3>{{ t("功耗 W", "Power W") }}</h3><Chart :option="powerOpt" :height="miniH" /></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.stats { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 6px; margin-bottom: 8px; }
.stats div { display: flex; flex-direction: column; }
.stats span { font-size: 11.5px; color: var(--dim); }
.stats b { font-size: 14px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.thr { font-size: 12px; color: var(--dim); margin-bottom: 8px; }
.thr.on { color: var(--warn); font-weight: 600; }
h3 { margin: 4px 0 0; }
.minis { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; }
.k { display: inline-block; width: 14px; margin-right: 5px; vertical-align: middle; }
.k.line { height: 2px; background: #4f46e5; }
.k.ref { border-top: 1px dashed #94a3b8; }
.k.band { height: 8px; background: rgba(217, 119, 6, 0.35); }
@media (max-width: 1300px) { .stats { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 700px) { .minis { grid-template-columns: 1fr; gap: 0; } .minis h3 { font-size: 12px; } }
</style>
