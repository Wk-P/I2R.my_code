<script setup>
// Two batches side by side (/compare?a=<batch>&b=<batch>): the main runs of both, grouped by scenario x
// RL_algorithm x strategy (+ params that differ inside a batch), each cell the mean over the finished seeds.
// A bar chart of the test relative gap per scenario, then the table with the change B − A and the seeds.
import { ref, computed, watch, onMounted } from "vue";
import { useRoute, useRouter } from "vue-router";
import { t, tl } from "../i18n.js";
import { getBatches, getBatch } from "../api.js";
import { pct } from "../format.js";
import { SCENARIOS, RL_ALGORITHMS, STRATEGIES, scenName } from "../labels.js";
import Chart from "../components/Chart.vue";

const route = useRoute();
const router = useRouter();
const list = ref([]);
const A = ref(null), B = ref(null);
const nameA = computed(() => route.query.a || ""), nameB = computed(() => route.query.b || "");
const setQ = (k, v) => router.replace({ query: { ...route.query, [k]: v } });

onMounted(async () => { list.value = (await getBatches()).batches.filter((b) => b.training); });
watch(nameA, async (n) => { A.value = n ? await getBatch(n) : null; }, { immediate: true });
watch(nameB, async (n) => { B.value = n ? await getBatch(n) : null; }, { immediate: true });

const mean = (xs) => { const v = xs.filter((x) => x != null && !Number.isNaN(x)); return v.length ? v.reduce((a, c) => a + c, 0) / v.length : null; };
// params of a run that differ from its batch's (as on the batch page), minus steps / seed
const diffParams = (b, p) => Object.entries(p || {}).filter(([k, x]) => !["steps", "seed"].includes(k) && x != null && x !== ""
  && JSON.stringify(x) !== JSON.stringify(b.params?.[k])).map(([k, x]) => `${k}=${x}`).join(" · ");
const modelKey = (b, r) => [r.RL_algorithm, r.strategy, diffParams(b, r.params)].join("|");
function groups(b) {
  const out = new Map();
  for (const r of (b?.runs || []).filter((r) => r.role === "main" && r.status === "done" && r.metrics)) {
    const k = `${r.scenario}|${modelKey(b, r)}`;
    if (!out.has(k)) out.set(k, []);
    out.get(k).push(r);
  }
  const cell = new Map();
  for (const [k, rs] of out) {
    const m = (f) => mean(rs.map((r) => r.metrics[f]));
    cell.set(k, { n: rs.length, gap: m("rel_gap"), cap: m("cap_viol"), priv: m("priv_viol"), exit: m("exit"),
      seeds: rs.map((r) => r.metrics.rel_gap).sort((x, y) => x - y) });
  }
  return cell;
}
const gA = computed(() => groups(A.value)), gB = computed(() => groups(B.value));
// models present in both batches (by RL_algorithm x strategy; params may differ between batches, so match on those)
const models = computed(() => {
  const ms = (b, g) => [...new Set([...g.keys()].map((k) => k.split("|").slice(1).join("|")))];
  const a = ms(A.value, gA.value), b = ms(B.value, gB.value);
  const base = (k) => k.split("|").slice(0, 2).join("|");
  const pairs = [];
  for (const ka of a) for (const kb of b) if (base(ka) === base(kb)) pairs.push({ ka, kb, rl: ka.split("|")[0], st: ka.split("|")[1],
    pa: ka.split("|")[2], pb: kb.split("|")[2] });
  const ri = (x) => RL_ALGORITHMS.indexOf(x), si = (x) => STRATEGIES.indexOf(x);
  return pairs.sort((x, y) => ri(x.rl) - ri(y.rl) || si(x.st) - si(y.st));
});
const scens = computed(() => SCENARIOS.filter((s) => models.value.some((m) => gA.value.has(`${s}|${m.ka}`) || gB.value.has(`${s}|${m.kb}`))));
const rows = computed(() => models.value.flatMap((m) => scens.value.map((s) => ({ ...m, s, a: gA.value.get(`${s}|${m.ka}`), b: gB.value.get(`${s}|${m.kb}`) }))));
const avg = (m, side) => mean(scens.value.map((s) => (side === "a" ? gA.value : gB.value).get(`${s}|${side === "a" ? m.ka : m.kb}`)?.gap));
const label = (m) => `${m.rl} · ${m.st}`;
const verA = computed(() => (A.value?.version ? `v${A.value.version}` : nameA.value));
const verB = computed(() => (B.value?.version ? `v${B.value.version}` : nameB.value));

const option = computed(() => {
  const cats = models.value.flatMap((m) => [...scens.value.map((s) => s.toUpperCase()), t("平均", "Mean")].map((s) => `${label(m)}\n${s}`));
  const val = (side) => models.value.flatMap((m) => [...scens.value.map((s) => (side === "a" ? gA.value : gB.value).get(`${s}|${side === "a" ? m.ka : m.kb}`)?.gap ?? null), avg(m, side)]
    .map((v) => (v == null ? null : +(100 * v).toFixed(2))));
  const bar = (name, data, color) => ({ name, type: "bar", data, barGap: "10%", itemStyle: { color },
    label: { show: true, position: "top", fontSize: 10, formatter: (p) => (p.value == null ? "" : `${p.value.toFixed(1)}`) } });
  return {
    tooltip: { trigger: "axis", valueFormatter: (v) => (v == null ? "—" : `${v.toFixed(2)}%`) },
    legend: { top: 0 },
    grid: { left: 48, right: 16, top: 36, bottom: 56 },
    xAxis: { type: "category", data: cats, axisLabel: { interval: 0, fontSize: 11 } },
    yAxis: { type: "value", name: t("相对 gap (%)", "Relative gap (%)") },
    series: [bar(verA.value, val("a"), "#94a3b8"),
      { ...bar(verB.value, val("b"), "#4f46e5"), markLine: { symbol: "none", silent: true, lineStyle: { color: "#dc2626", type: "dashed" },
        label: { formatter: t("目标 5%", "target 5%"), position: "insideEndTop" }, data: [{ yAxis: 5 }] } }],
  };
});
const chartWidth = computed(() => Math.max(600, models.value.length * (scens.value.length + 1) * 90));
const delta = (a, b) => (a == null || b == null ? null : b - a);
const tone = (d) => (d == null ? "" : d < -0.002 ? "good" : d > 0.002 ? "bad" : "");
const sgn = (d) => (d == null ? "—" : `${d > 0 ? "+" : d < 0 ? "−" : "±"}${(100 * Math.abs(d)).toFixed(2)} pp`);
</script>

<template>
  <div class="page">
    <div class="head"><div>
      <h1>{{ t("批次对比", "Compare batches") }}</h1>
      <div class="head-sub">{{ t("两个批次的正式 run，按场景 × RL_algorithm × strategy 分组，各格为已完成种子的均值；相对 gap =（ILP AR − AR）/ ILP AR，只在无违约的测试实例上计算。",
        "Main runs of two batches, grouped by scenario × RL_algorithm × strategy; each cell averages the finished seeds. Relative gap = (ILP AR − AR) / ILP AR over the violation-free test instances.") }}</div>
    </div></div>

    <div class="card"><div class="card-b row" style="gap: 12px">
      <label class="small dim">A（{{ t("基线", "baseline") }}）
        <select :value="nameA" @change="setQ('a', $event.target.value)"><option value="">—</option><option v-for="b in list" :key="b.name" :value="b.name">{{ b.name }} · {{ tl(b.title) }}</option></select></label>
      <label class="small dim">B（{{ t("新版本", "new") }}）
        <select :value="nameB" @change="setQ('b', $event.target.value)"><option value="">—</option><option v-for="b in list" :key="b.name" :value="b.name">{{ b.name }} · {{ tl(b.title) }}</option></select></label>
      <router-link v-if="A" class="tag accent" :to="`/batches/${nameA}`">A · {{ nameA }}</router-link>
      <router-link v-if="B" class="tag accent" :to="`/batches/${nameB}`">B · {{ nameB }}</router-link>
    </div></div>

    <div v-if="!A || !B" class="card empty">{{ t("请选择两个批次。", "Pick two batches.") }}</div>
    <div v-else-if="!models.length" class="card empty">{{ t("两个批次没有共同的 RL_algorithm × strategy。", "The two batches share no RL_algorithm × strategy.") }}</div>
    <template v-else>
      <div class="card">
        <div class="card-h"><h2>{{ t("测试集相对 gap", "Test relative gap") }}</h2><span class="faint small">{{ verA }} vs {{ verB }} · {{ t("越低越好，虚线为 5% 目标", "lower is better; dashed line = 5% target") }}</span></div>
        <div class="card-b table-wrap"><div :style="{ minWidth: chartWidth + 'px' }"><Chart :option="option" :height="340" /></div></div>
      </div>

      <div class="card">
        <div class="card-h"><h2>{{ t("结果列表", "Results") }}</h2><span class="faint small">{{ t("Δ = B − A，单位为百分点；绿色为变好", "Δ = B − A in percentage points; green = better") }}</span></div>
        <div class="card-b table-wrap">
          <table class="tbl">
            <thead><tr>
              <th>{{ t("模型", "Model") }}</th><th>{{ t("场景", "Scen.") }}</th>
              <th class="num">{{ verA }} gap</th><th class="num">{{ verB }} gap</th><th class="num">Δ gap</th>
              <th class="num">{{ t("容量违约", "Cap. viol.") }} A / B</th><th class="num">{{ t("隐私违约", "Priv. viol.") }} A / B</th>
              <th class="num">EXIT A / B</th><th class="num">Δ EXIT</th><th>{{ t("各种子 gap", "Per-seed gap") }} A → B</th>
            </tr></thead>
            <tbody>
              <template v-for="m in models" :key="m.ka + m.kb">
                <tr v-for="r in rows.filter((x) => x.ka === m.ka && x.kb === m.kb)" :key="r.s">
                  <td><b>{{ label(m) }}</b><div v-if="m.pa || m.pb" class="small faint">{{ m.pa || "—" }} → {{ m.pb || "—" }}</div></td>
                  <td>{{ scenName(r.s) }}</td>
                  <td class="num">{{ pct(r.a?.gap, 2) }}</td><td class="num"><b>{{ pct(r.b?.gap, 2) }}</b></td>
                  <td class="num" :class="tone(delta(r.a?.gap, r.b?.gap))">{{ sgn(delta(r.a?.gap, r.b?.gap)) }}</td>
                  <td class="num">{{ pct(r.a?.cap) }} / {{ pct(r.b?.cap) }}</td><td class="num">{{ pct(r.a?.priv) }} / {{ pct(r.b?.priv) }}</td>
                  <td class="num">{{ m.st === "maskable" ? `${pct(r.a?.exit, 2)} / ${pct(r.b?.exit, 2)}` : "" }}</td>
                  <td class="num" :class="m.st === 'maskable' ? tone(delta(r.a?.exit, r.b?.exit)) : ''">{{ m.st === "maskable" ? sgn(delta(r.a?.exit, r.b?.exit)) : "" }}</td>
                  <td class="small mono">{{ (r.a?.seeds || []).map((x) => pct(x)).join(" ") || "—" }} → {{ (r.b?.seeds || []).map((x) => pct(x)).join(" ") || "—" }}</td>
                </tr>
                <tr class="avg">
                  <td><b>{{ label(m) }}</b></td><td>{{ t("三场景平均", "Mean of scenarios") }}</td>
                  <td class="num">{{ pct(avg(m, "a"), 2) }}</td><td class="num"><b>{{ pct(avg(m, "b"), 2) }}</b></td>
                  <td class="num" :class="tone(delta(avg(m, 'a'), avg(m, 'b')))">{{ sgn(delta(avg(m, "a"), avg(m, "b"))) }}</td>
                  <td colspan="5"></td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>
        <div class="card-note">{{ t("只列两个批次都有的 RL_algorithm × strategy；数据来自各 run 的 results.json。", "Only RL_algorithm × strategy present in both batches; from each run's results.json.") }}</div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.good { color: var(--ok); font-weight: 650; }
.bad { color: var(--bad); font-weight: 650; }
tr.avg td { border-top: 1px solid var(--border); background: var(--accent-soft); }
select { margin-left: 6px; max-width: 360px; }
</style>
