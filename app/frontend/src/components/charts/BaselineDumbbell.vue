<script setup>
// Each version's change against its own baseline, per scenario (small multiples LT / EQ / GT / mean):
// grey dot = baseline, coloured dot = the version's new variant, joined by a line; left = better.
import { ref, computed } from "vue";
import { langRef as curLang } from "../../i18n.js";
import { SCEN_COLOR, TARGET, useWidth, niceMax, fmt1, vcmp } from "./common.js";

const props = defineProps({ reports: { type: Array, default: () => [] } });
const L = (zh, en) => (curLang.value === "en" ? en : zh);
const SC = ["LT", "EQ", "GT", "AVG"];
const showEarly = ref(false);
const mean = (g) => { const v = ["LT", "EQ", "GT"].map((s) => g?.[s]).filter((x) => x != null); return v.length ? v.reduce((a, b) => a + b) / v.length : null; };
const rows = computed(() => props.reports
  .filter((r) => r.new_model && r.base_model && r.new_model !== r.base_model && r.gaps?.[r.new_model] && r.gaps?.[r.base_model])
  .filter((r) => showEarly.value || vcmp(r.version, "v4.4.3") >= 0)
  .sort((a, b) => vcmp(b.version, a.version))
  .map((r) => {
    const n = r.gaps[r.new_model], b = r.gaps[r.base_model];
    return { v: r.version, change: curLang.value === "en" ? r.en?.change : r.change,
             newName: curLang.value === "en" ? r.en?.new_model ?? r.new_model : r.new_model,
             baseName: curLang.value === "en" ? r.en?.base_model ?? r.base_model : r.base_model,
             n: { ...n, AVG: mean(n) }, b: { ...b, AVG: mean(b) } };
  }));

const { el, width } = useWidth(320);
const cols = computed(() => Math.max(1, Math.min(4, Math.floor((width.value + 12) / 220))));
const LBL = 58;                                            // version label column (first panel of a row only)
const PW = computed(() => Math.floor((width.value - LBL - 12 * (cols.value - 1)) / cols.value));
const ROW = 22, M = { t: 6, r: 18, b: 24, l: 16 };
const H = computed(() => M.t + rows.value.length * ROW + M.b);
const xMax = computed(() => niceMax(Math.max(8, ...rows.value.flatMap((r) => SC.flatMap((s) => [r.n[s] ?? 0, r.b[s] ?? 0]))), 4));
const xs = (v) => M.l + (v / xMax.value) * (PW.value - M.l - M.r);
const ticks = computed(() => { const t = []; const st = xMax.value > 16 ? 4 : 2; for (let v = 0; v <= xMax.value; v += st) t.push(v); return t; });
const yr = (i) => M.t + i * ROW + ROW / 2;
const hover = ref(null);
const scenName = (s) => (s === "AVG" ? L("三场景平均", "3-scenario mean") : s);
</script>

<template>
  <div class="db" data-no-i18n>
    <div class="db-head">
      <span class="legend">
        <span class="legend-item"><span class="dot dot--base"></span>{{ L("基线", "baseline") }}</span>
        <span class="legend-item"><span class="dot" :style="{ background: SCEN_COLOR.LT }"></span>{{ L("新变体（颜色 = 场景）", "new variant (colour = scenario)") }}</span>
        <span class="legend-item"><span class="sw sw--target"></span>{{ L("目标 5%", "goal 5%") }}</span>
        <span class="legend-item dim">{{ L("越靠左越好", "further left = better") }}</span>
      </span>
      <label class="small dim"><input v-model="showEarly" type="checkbox" /> {{ L("显示 MLP 时期（v4.4.3 之前）", "show the MLP era (before v4.4.3)") }}</label>
    </div>
    <div ref="el" class="db-grid" :style="{ gridTemplateColumns: `${LBL}px repeat(${cols}, ${PW}px)` }" @mouseleave="hover = null">
      <template v-for="(s, k) in SC" :key="s">
        <div v-if="k % cols === 0" class="vlabels" :style="{ height: H + 'px' }">
          <div class="ph"></div>
          <div v-for="(r, i) in rows" :key="r.v" class="vlabel" :class="{ on: hover === i }" :style="{ top: (yr(i) + 18) + 'px' }">
            <a :href="`#/versions/${r.v}`">{{ r.v }}</a></div>
        </div>
        <div class="db-panel">
          <div class="panel-title">{{ scenName(s) }}</div>
          <svg :width="PW" :height="H" role="img" :aria-label="`${scenName(s)}: ${L('基线与新变体', 'baseline and new variant')}`">
            <g class="ax">
              <g v-for="t in ticks" :key="t"><line :x1="xs(t)" :x2="xs(t)" :y1="M.t" :y2="H - M.b" />
                <text :x="xs(t)" :y="H - 8" text-anchor="middle">{{ t }}%</text></g>
            </g>
            <line class="target" :x1="xs(TARGET)" :x2="xs(TARGET)" :y1="M.t" :y2="H - M.b" />
            <g v-for="(r, i) in rows" :key="r.v" @mouseenter="hover = i">
              <rect class="hit" :x="0" :y="yr(i) - ROW / 2" :width="PW" :height="ROW" :class="{ on: hover === i }" />
              <template v-if="r.n[s] != null && r.b[s] != null">
                <line class="conn" :x1="xs(r.b[s])" :x2="xs(r.n[s])" :y1="yr(i)" :y2="yr(i)" />
                <circle :cx="xs(r.b[s])" :cy="yr(i)" r="4" class="dot-base" />
                <circle :cx="xs(r.n[s])" :cy="yr(i)" r="5" :fill="SCEN_COLOR[s]" class="dot-new" />
              </template>
            </g>
          </svg>
        </div>
      </template>
    </div>
    <div v-if="hover !== null && rows[hover]" class="tip">
      <div class="tip-head"><a :href="`#/versions/${rows[hover].v}`">{{ rows[hover].v }}</a></div>
      <div class="tip-change dim">{{ rows[hover].change || "" }}</div>
      <table><thead><tr><th></th><th class="num">{{ L("基线", "Baseline") }}</th><th class="num">{{ L("新变体", "New") }}</th><th class="num">Δ</th></tr></thead>
        <tbody><tr v-for="s in SC" :key="s"><td>{{ scenName(s) }}</td><td class="num">{{ fmt1(rows[hover].b[s]) }}</td><td class="num"><b>{{ fmt1(rows[hover].n[s]) }}</b></td>
          <td class="num" :class="(rows[hover].n[s] - rows[hover].b[s]) < 0 ? 'better' : 'worse'">{{ (rows[hover].n[s] - rows[hover].b[s] > 0 ? "+" : "") + (rows[hover].n[s] - rows[hover].b[s]).toFixed(1) }}</td></tr></tbody></table>
      <div class="small dim">{{ L("基线", "baseline") }}: {{ rows[hover].baseName }} · {{ L("新变体", "new") }}: {{ rows[hover].newName }}</div>
    </div>
    <div v-if="!rows.length" class="empty">{{ L("没有同时带基线和新变体的版本", "No version has both a baseline and a new variant") }}</div>
  </div>
</template>

<style scoped>
.db-head { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 6px; }
.legend { display: inline-flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: var(--text-dim); }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; }
.dot--base { background: var(--text-faint); }
.sw--target { display: inline-block; width: 14px; border-top: 2px dashed var(--ok); }
.db-grid { display: grid; column-gap: 12px; row-gap: 16px; width: 100%; }
.vlabels { position: relative; }
.ph { height: 18px; }
.vlabel { position: absolute; right: 4px; transform: translateY(-50%); font-size: 11px; font-variant-numeric: tabular-nums; }
.vlabel a { color: var(--text-dim); }
.vlabel.on a { color: var(--primary); font-weight: 600; }
.panel-title { font-size: 12px; font-weight: 650; color: var(--text-dim); height: 18px; }
.ax line { stroke: var(--border); stroke-width: 1; }
.ax text { font-size: 10px; fill: var(--text-faint); }
.target { stroke: var(--ok); stroke-width: 1.5; stroke-dasharray: 6 4; }
.hit { fill: transparent; }
.hit.on { fill: var(--primary-soft); }
.conn { stroke: var(--border-strong); stroke-width: 2; }
.dot-base { fill: var(--text-faint); stroke: var(--panel); stroke-width: 2; }
.dot-new { stroke: var(--panel); stroke-width: 2; }
.tip { margin-top: 8px; background: var(--panel); border: 1px solid var(--border-strong); border-radius: 8px; padding: 8px 10px; font-size: 12px; max-width: 560px; }
.tip-head { font-weight: 600; }
.tip-change { margin: 2px 0 6px; }
.tip table { border-collapse: collapse; margin-bottom: 4px; }
.tip td, .tip th { padding: 1px 10px 1px 0; text-align: left; font-weight: 500; }
.tip .num { text-align: right; font-variant-numeric: tabular-nums; }
.better { color: var(--ok); } .worse { color: var(--bad); }
</style>
