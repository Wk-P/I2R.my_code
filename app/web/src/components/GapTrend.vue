<script setup>
// Test relative gap by version (/api/v2/version_trend): LT / EQ / GT and the three-scenario mean of each
// version's new variant; 5% target and the final baseline as reference lines; the anchors (version_trend.json)
// marked on the mean line; dashed vertical lines where a phase starts (Transformer, 50k instances, 5M / 10M / 20M).
// The MLP era (before v4.4.3) is hidden unless ticked. Same numbers as the classic panel's chart.
import { ref, computed, onMounted } from "vue";
import { t, tl, lang } from "../i18n.js";
import { getVersionTrend } from "../api.js";
import Chart from "./Chart.vue";

const d = ref(null);
const showMlp = ref(false);
onMounted(async () => { d.value = await getVersionTrend(); });

const vkey = (v) => v.replace(/^v/, "").split(".").map(Number);
const vcmp = (a, b) => { const x = vkey(a), y = vkey(b); for (let i = 0; i < Math.max(x.length, y.length); i++) { const k = (x[i] ?? 0) - (y[i] ?? 0); if (k) return k; } return 0; };
const pts = computed(() => (d.value?.points || []).filter((p) => showMlp.value || !d.value.mlp_era_before || vcmp(p.version, d.value.mlp_era_before) >= 0));
const SERIES = [
  { key: "LT", color: "#2a78d6", width: 2 }, { key: "EQ", color: "#eb6834", width: 2 },
  { key: "GT", color: "#1baf7a", width: 2 }, { key: "avg", color: "#171c2b", width: 3 },
];
const name = (k) => (k === "avg" ? t("三场景平均", "3-scenario mean") : k);
const narrow = window.matchMedia?.("(max-width: 700px)").matches;
const short = (v) => v.replace(/^v/, "");

const option = computed(() => {
  lang.value;
  if (!pts.value.length) return {};
  const xs = pts.value.map((p) => short(p.version));
  const yMax = Math.ceil(Math.max(12, ...pts.value.flatMap((p) => [p.LT, p.EQ, p.GT, p.avg])) / 2) * 2;
  const phases = (d.value.phases || []).filter((ph) => xs.includes(short(ph.from)));
  // end labels: nudge apart (in value units) when two series end too close, as the classic chart does
  const plotH = (narrow ? 300 : 340) - 40 - (narrow ? 46 : 34), gapV = (14 / plotH) * yMax;
  const last = pts.value[pts.value.length - 1];
  const ends = SERIES.map((s) => ({ key: s.key, v: last[s.key], y: last[s.key] })).sort((a, b) => b.v - a.v);
  for (let i = 1; i < ends.length; i++) if (ends[i - 1].y - ends[i].y < gapV) ends[i].y = ends[i - 1].y - gapV;
  const off = Object.fromEntries(ends.map((e) => [e.key, ((e.v - e.y) / yMax) * plotH]));
  const series = SERIES.map((s) => ({
    type: "line", name: name(s.key), color: s.color, data: pts.value.map((p) => p[s.key]), symbol: "circle", symbolSize: s.key === "avg" ? 8 : 7,
    lineStyle: { width: s.width }, itemStyle: { borderColor: "#fff", borderWidth: 1.5 }, z: s.key === "avg" ? 4 : 3,
    endLabel: narrow ? undefined : { show: true, formatter: (p) => `${name(s.key)} ${p.value?.toFixed(1)}%`, color: s.color, fontSize: 11, distance: 6, offset: [0, off[s.key]] },
    emphasis: { focus: "series" },
  }));
  const avg = series[3];
  // anchors: a diamond with its label on the mean line
  avg.markPoint = { symbol: "diamond", symbolSize: 16, itemStyle: { color: "#fff", borderColor: "#d97706", borderWidth: 2.5 },
    label: { show: !narrow, position: "top", distance: 8, color: "#b45309", fontSize: 11, fontWeight: 600, formatter: (p) => p.name },
    data: pts.value.filter((p) => p.anchor).map((p) => ({ name: tl(p.anchor.short ?? p.anchor), coord: [short(p.version), p.avg] })) };
  // reference lines and phase starts
  series.push({ type: "line", data: [], silent: true,
    markLine: { symbol: "none", silent: true, data: [
      { yAxis: d.value.target, lineStyle: { type: "dashed", color: "#15803d", width: 1.5 }, label: { show: !narrow, formatter: t(`目标 ${d.value.target}%`, `target ${d.value.target}%`), position: "insideStartTop", color: "#15803d", fontSize: 11 } },
      ...(d.value.baseline != null ? [{ yAxis: d.value.baseline, lineStyle: { type: "dotted", color: "#94a3b8", width: 1.5 }, label: { show: !narrow, formatter: t(`定稿基线 ${d.value.baseline}%`, `final baseline ${d.value.baseline}%`), position: "insideStartTop", color: "#64748b", fontSize: 11 } }] : []),
      ...phases.map((ph) => ({ xAxis: short(ph.from), lineStyle: { type: [4, 4], color: "#a5b4fc", width: 1 },
        label: { show: !narrow, formatter: tl(ph), position: "end", color: "#4f46e5", fontSize: 10.5 } })),
    ] } });
  return {
    grid: { left: 40, right: narrow ? 12 : 128, top: narrow ? 14 : 40, bottom: narrow ? 46 : 34 },
    legend: { show: false },
    tooltip: { trigger: "axis", confine: true, formatter: (ps) => {
      const p = pts.value[ps[0].dataIndex];
      if (!p) return "";
      const rows = SERIES.map((s) => `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${s.color};margin-right:6px"></span>${name(s.key)}<b style="float:right;margin-left:16px">${p[s.key].toFixed(1)}%</b>`);
      return `<div style="max-width:300px;white-space:normal"><b>${p.version}</b> · ${tl(p.model)}${p.anchor ? ` <span style="color:#b45309">（${tl(p.anchor)}）</span>` : ""}`
        + `<div style="color:#5b6478;font-size:12px;margin:4px 0">${tl(p.change) || ""}</div>${rows.join("<br>")}</div>`;
    } },
    xAxis: { type: "category", data: xs, boundaryGap: false, axisLabel: { fontSize: 11, rotate: narrow ? 45 : 0, interval: 0 }, splitLine: { show: false } },
    yAxis: { type: "value", min: 0, max: yMax, interval: yMax > 14 ? 4 : 2, axisLabel: { formatter: "{value}%" } },
    series,
  };
});
</script>

<template>
  <div class="card">
    <div class="card-h">
      <h2>{{ t("各版本测试集相对 gap", "Test relative gap by version") }}</h2>
      <span class="spacer" />
      <label class="check small"><input v-model="showMlp" type="checkbox" />{{ t("显示 MLP 时期（v4.4.3 之前）", "show the MLP era (before v4.4.3)") }}</label>
    </div>
    <div class="card-b">
      <div class="lg small">
        <span v-for="s in SERIES" :key="s.key"><i :style="{ background: s.color, height: s.width + 'px' }" />{{ name(s.key) }}</span>
        <span><i class="dash" style="border-color: #15803d" />{{ t("目标", "target") }} {{ d?.target }}%</span>
        <span v-if="d?.baseline != null"><i class="dot" />{{ t("定稿基线", "final baseline") }} {{ d.baseline }}%</span>
        <span><b class="dia">◇</b>{{ t("锚点", "anchor") }}</span>
        <span><i class="ph" />{{ t("阶段开始", "phase starts") }}<template v-if="narrow">：{{ (d?.phases || []).map((p) => `${tl(p)} ${p.from.replace(/^v/, "")}`).join("，") }}</template></span>
        <span v-if="narrow"><b class="dia">◇</b>{{ (d?.points || []).filter((p) => p.anchor).map((p) => tl(p.anchor)).join("，") }}</span>
      </div>
      <div v-if="!d" class="empty">{{ t("加载中…", "Loading…") }}</div>
      <Chart v-else :option="option" :height="narrow ? 300 : 340" />
      <details class="tbl-d">
        <summary class="small">{{ t("数据表", "Data table") }}</summary>
        <div class="table-wrap">
          <table class="tbl">
            <thead><tr><th>{{ t("版本", "Version") }}</th><th class="hide-sm">{{ t("新变体", "New variant") }}</th><th class="num">LT</th><th class="num">EQ</th><th class="num">GT</th><th class="num">{{ name("avg") }}</th><th class="hide-sm">{{ t("报告", "Report") }}</th></tr></thead>
            <tbody>
              <tr v-for="p in [...pts].reverse()" :key="p.version">
                <td><b>{{ p.version }}</b><span v-if="p.anchor" class="tag" style="margin-left: 6px; color: #b45309; border-color: #f5c27a">{{ tl(p.anchor) }}</span></td>
                <td class="small hide-sm">{{ tl(p.model) }}</td>
                <td class="num">{{ p.LT.toFixed(1) }}%</td><td class="num">{{ p.EQ.toFixed(1) }}%</td><td class="num">{{ p.GT.toFixed(1) }}%</td>
                <td class="num"><b>{{ p.avg.toFixed(1) }}%</b></td>
                <td class="small mono faint hide-sm">paper_contents/{{ p.version }}/{{ p.report }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-if="d?.skipped?.length" class="small faint" style="margin-top: 6px">
          {{ t("未画出：", "Not plotted: ") }}{{ d.skipped.map((s) => `${s.version}（${s.reason}）`).join("；") }}
        </div>
      </details>
    </div>
    <div class="card-note">{{ t("每个版本取该版本新变体的测试集相对 gap（各版本报告的汇总表）；纳入哪些版本、锚点与阶段写在 app/backend/v2/version_trend.json。", "Each version: its new variant's test relative gap (summary table of the version's report); which versions, the anchors and the phases are listed in app/backend/v2/version_trend.json.") }}</div>
  </div>
</template>

<style scoped>
.lg { display: flex; gap: 14px; flex-wrap: wrap; color: var(--dim); margin-bottom: 4px; align-items: center; }
.lg i { display: inline-block; width: 16px; margin-right: 5px; vertical-align: middle; }
.lg i.dash { height: 0; border-top: 2px dashed; }
.lg i.dot { height: 0; border-top: 2px dotted #94a3b8; }
.lg i.ph { height: 12px; width: 0; border-left: 1.5px dashed #a5b4fc; margin: 0 8px 0 4px; }
.lg .dia { color: #d97706; margin-right: 4px; }
.tbl-d { margin-top: 8px; }
.tbl-d summary { cursor: pointer; color: var(--dim); }
</style>
