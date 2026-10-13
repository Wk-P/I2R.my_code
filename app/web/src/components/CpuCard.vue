<script setup>
// The CPUs: current state, then the last `window` seconds of utilisation, mean frequency (max as a dashed line),
// package temperatures (one line per socket) and package / DRAM power (RAPL; root-only by default -- shown with the
// command that makes it readable). Same look as GpuCard.
import { computed } from "vue";
import { t, lang } from "../i18n.js";
import Chart from "./Chart.vue";

const props = defineProps({
  sys: { type: Object, required: true },             // /api/v2 system (cpu = latest sample)
  samples: { type: Array, default: () => [] },      // [{t, cpu: {util, freq, freq_max, temps: [{name, c}], power: [{name, w}]}}]
  window: { type: Number, default: 1800 },
});
const COLORS = ["#4f46e5", "#d97706", "#059669", "#dc2626"];
const now = computed(() => (props.samples.length ? props.samples[props.samples.length - 1].t : Date.now() / 1000));
const cur = computed(() => props.sys.cpu || {});
const xAxis = computed(() => ({ type: "time", min: (now.value - props.window) * 1000, max: now.value * 1000, splitNumber: 4,
  axisLabel: { formatter: "{HH}:{mm}", fontSize: 10, hideOverlap: true }, splitLine: { show: false } }));
const pts = (get) => {
  const out = [];
  let prev = null;
  for (const s of props.samples) {
    if (prev && s.t - prev.t > 10) out.push([(prev.t + 1) * 1000, null]);
    out.push([s.t * 1000, s.cpu ? get(s.cpu) ?? null : null]);
    prev = s;
  }
  return out;
};
const names = (key) => [...new Set(props.samples.flatMap((s) => (s.cpu?.[key] || []).map((x) => x.name)))];
const short = (n) => n.replace("Package id ", t("CPU ", "CPU ")).replace(/^package-/, "CPU ").replace(/^dram\s*/, "DRAM ");
const chart = (series, yAxis, unit) => ({
  grid: { left: 44, right: 10, top: series.length > 1 ? 24 : 10, bottom: 24 },
  legend: series.length > 1 ? { top: 0, right: 0, itemWidth: 12, textStyle: { fontSize: 10.5 } } : undefined,
  tooltip: { trigger: "axis", confine: true, valueFormatter: (v) => (v == null ? "—" : `${Math.round(v * 10) / 10} ${unit}`) },
  xAxis: xAxis.value, yAxis: { type: "value", axisLabel: { fontSize: 10 }, ...yAxis }, series,
});
const line = (name, color, data, extra = {}) => ({ type: "line", name, color, showSymbol: false, lineStyle: { width: 1.8 }, data, ...extra });
const utilOpt = computed(() => { lang.value; return chart([line(t("利用率", "util"), "#4f46e5", pts((c) => c.util))], { min: 0, max: 100, interval: 50 }, "%"); });
const freqOpt = computed(() => {
  lang.value;
  const max = props.samples.find((s) => s.cpu?.freq_max)?.cpu.freq_max ?? cur.value.freq_max;
  const s = line(t("平均频率", "mean clock"), "#0891b2", pts((c) => c.freq));
  if (max) s.markLine = { silent: true, symbol: "none", lineStyle: { type: "dashed", color: "#94a3b8" },
    label: { formatter: `${t("最高", "max")} ${max}`, position: "insideEndTop", fontSize: 10, color: "#64748b" }, data: [{ yAxis: max }] };
  const vals = props.samples.map((x) => x.cpu?.freq).filter((v) => v != null);
  const lo = Math.floor((Math.min(...vals, max || 3000) - 300) / 500) * 500;
  return chart([s], { min: Math.max(0, lo), max: Math.ceil(((max || 3000) + 100) / 500) * 500, interval: 500 }, "MHz");
});
const tempOpt = computed(() => {
  lang.value;
  const ns = names("temps");
  const all = props.samples.flatMap((s) => (s.cpu?.temps || []).map((x) => x.c));
  const hi = Math.max(90, Math.ceil((Math.max(...all, 0) + 5) / 10) * 10);
  return chart(ns.map((n, k) => line(short(n), COLORS[k % 4], pts((c) => c.temps?.find((x) => x.name === n)?.c))),
    { min: Math.min(hi - 60, Math.floor((Math.min(...all, hi) - 5) / 10) * 10), max: hi, interval: 20 }, "°C");
});
const powerOpt = computed(() => {
  lang.value;
  const ns = names("power");
  return chart(ns.map((n, k) => line(short(n), COLORS[k % 4], pts((c) => c.power?.find((x) => x.name === n)?.w))), { min: 0 }, "W");
});
const readable = computed(() => cur.value.power_readable);
const miniH = window.matchMedia?.("(max-width: 700px)").matches ? 90 : 120;
// as root, once: a udev rule makes the counters owned (read-only) by the panel's user, also after a reboot
const CMD = `echo 'SUBSYSTEM=="powercap", ACTION=="add", RUN+="/bin/chown soar009 /sys%p/energy_uj"' > /etc/udev/rules.d/90-rapl-soar009.rules
udevadm control --reload-rules && udevadm trigger --subsystem-match=powercap --action=add`;
</script>

<template>
  <div class="card cpu">
    <div class="card-h">
      <h2>CPU</h2><span class="faint small">{{ sys.cpu_model }} · {{ sys.cpu_count }} {{ t("线程", "threads") }}</span>
      <span class="spacer" />
      <b class="mono">{{ cur.freq ?? "—" }} MHz</b>
    </div>
    <div class="card-b">
      <div class="stats">
        <div><span>{{ t("利用率", "Util") }}</span><b>{{ cur.util ?? "—" }}%</b></div>
        <div><span>{{ t("负载 1 分钟", "Load 1 min") }}</span><b>{{ sys.load_avg?.["1m"]?.toFixed(1) ?? "—" }}</b></div>
        <div v-for="x in cur.temps || []" :key="x.name"><span>{{ short(x.name) }} {{ t("温度", "temp") }}</span><b>{{ x.c.toFixed(0) }} °C</b></div>
        <div v-for="x in cur.power || []" :key="x.name"><span>{{ short(x.name) }} {{ t("功耗", "power") }}</span><b>{{ x.w.toFixed(0) }} W</b></div>
      </div>
      <div class="minis">
        <div><h3>{{ t("利用率 %", "Utilisation %") }}</h3><Chart :option="utilOpt" :height="miniH" /></div>
        <div><h3>{{ t("平均频率 MHz", "Mean clock MHz") }}</h3><Chart :option="freqOpt" :height="miniH" /></div>
        <div><h3>{{ t("封装温度 °C", "Package temperature °C") }}</h3><Chart :option="tempOpt" :height="miniH" /></div>
        <div>
          <h3>{{ t("功耗 W（RAPL）", "Power W (RAPL)") }}</h3>
          <Chart v-if="readable" :option="powerOpt" :height="miniH" />
          <div v-else class="noperm small">
            {{ t("功耗计数器只有 root 可读。用 root 执行一次（只授权给 soar009、只读，重启后仍有效）：", "The power counters are root-only. Run once as root (read-only, for soar009 only, kept after a reboot):") }}
            <code>{{ CMD }}</code>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.stats { display: flex; flex-wrap: wrap; gap: 6px 22px; margin-bottom: 8px; }
.stats div { display: flex; flex-direction: column; }
.stats span { font-size: 11.5px; color: var(--dim); }
.stats b { font-size: 14px; font-variant-numeric: tabular-nums; white-space: nowrap; }
h3 { margin: 4px 0 0; }
.minis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
.minis > * { min-width: 0; }
.noperm { margin-top: 8px; padding: 8px 10px; background: var(--warn-soft); color: var(--warn); border-radius: 8px; line-height: 1.6; }
.noperm code { display: block; margin-top: 4px; word-break: break-all; white-space: pre-wrap; color: var(--text); background: var(--panel); padding: 4px 6px; border-radius: 4px; }
@media (max-width: 1100px) { .minis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 700px) { .minis { grid-template-columns: 1fr; } }
</style>
