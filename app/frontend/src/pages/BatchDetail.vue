<script setup>
import { ref, computed, watch, onUnmounted } from "vue";
import { getBatch } from "../api.js";
import { SCENARIOS, algoLabel, algoIndex, learnerLabel, mechLabel, learnerOf, statusLabel, statusClass, zeroRepairViol } from "../labels.js";
import { pct, fmt, pm, shortTime, secToHuman, steps } from "../format.js";

const props = defineProps({ name: { type: String, required: true } });

const data = ref(null);
const missing = ref(false);
const scen = ref("all");
let timer = null;

async function load() {
  const d = await getBatch(props.name);
  (d?.rows || []).forEach(zeroRepairViol);
  missing.value = !d;
  data.value = d;
}
watch(() => props.name, () => {
  data.value = null;
  scen.value = "all";
  clearInterval(timer);
  load();
  timer = setInterval(load, 10000);
}, { immediate: true });
onUnmounted(() => clearInterval(timer));

const rows = computed(() => {
  const rs = (data.value?.rows || []).filter((r) => scen.value === "all" || r.scenario === scen.value);
  return [...rs].sort((a, b) =>
    SCENARIOS.indexOf(a.scenario) - SCENARIOS.indexOf(b.scenario) ||
    algoIndex(a.algo) - algoIndex(b.algo) || a.variant.localeCompare(b.variant));
});
const hasVariant = computed(() => (data.value?.rows || []).some((r) => r.variant));
const scenCount = (s) => (data.value?.rows || []).filter((r) => r.scenario === s).length;

// first row of each scenario group gets a divider
const firstOfLearner = (i) =>
  firstOfScen(i) || learnerOf(rows.value[i - 1].algo) !== learnerOf(rows.value[i].algo);
const firstOfScen = (i) => i === 0 || rows.value[i - 1].scenario !== rows.value[i].scenario;

const gap = (r) =>
  r.ilp_ar_mean == null || r.test_ar_mean_mean == null ? null : r.ilp_ar_mean - r.test_ar_mean_mean;

const seedTitle = (s) =>
  `种子 ${s.seed ?? "—"} · ${statusLabel(s.status)}${s.progress_pct != null ? ` ${s.progress_pct}%` : ""}${s.exp_id ? " · " + s.exp_id : ""}`;
const runLink = (r, s) =>
  s.status === "done" ? `#/run/${data.value.branch}/${r.scenario}/${r.algo}/${s.exp_id}` : null;
const resultsLink = computed(() => `#/results?branch=${encodeURIComponent(data.value?.branch ?? "")}&version=${encodeURIComponent(data.value?.version ?? "未标注版本")}&batch=${encodeURIComponent(props.name)}&view=pivot`);
</script>

<template>
  <div class="crumbs"><a href="#/batches">批次管理</a> / {{ name }}</div>
  <div v-if="missing" class="empty">找不到批次 {{ name }}</div>
  <div v-else-if="!data" class="empty">加载中…</div>
  <template v-else>
    <div class="page-head page-head--row">
      <div>
        <h1>{{ name }} <span class="badge badge--lg" :class="statusClass(data.status)">{{ statusLabel(data.status) }}</span></h1>
        <div class="page-sub">版本 {{ data.version ?? "—" }} · 分支 {{ data.branch }} · {{ steps(data.steps) }} 步 · 开始于 {{ shortTime(data.started_at) }}</div>
      </div>
      <a class="btn" :href="resultsLink">在实验结果中对比 →</a>
    </div>

    <div class="kpis">
      <div class="kpi"><div class="kpi-label">总进度</div><div class="kpi-value">{{ data.overall_pct }}%</div>
        <div class="bar bar--kpi"><div class="bar-fill" :style="{ width: data.overall_pct + '%' }"></div></div></div>
      <div class="kpi"><div class="kpi-label">已完成</div><div class="kpi-value">{{ data.done }}<span class="kpi-unit">/ {{ data.total_runs }}</span></div></div>
      <div class="kpi"><div class="kpi-label">运行 / 排队 / 中断{{ data.skipped ? " / 未运行" : "" }}</div><div class="kpi-value kpi-value--sm">{{ data.running }} / {{ data.queued }} / {{ data.stopped }}{{ data.skipped ? " / " + data.skipped : "" }}</div></div>
      <div class="kpi"><div class="kpi-label">已用时</div><div class="kpi-value kpi-value--sm">{{ secToHuman(data.elapsed_seconds) }}</div>
        <div class="kpi-foot">最后更新 {{ shortTime(data.last_updated) }}</div></div>
    </div>

    <div class="tabs">
      <button class="tab" :class="{ active: scen === 'all' }" @click="scen = 'all'">全部场景 <span class="tab-count">{{ data.rows.length }}</span></button>
      <button v-for="s in SCENARIOS" :key="s" v-show="scenCount(s)" class="tab" :class="{ active: scen === s }" @click="scen = s">
        {{ s.toUpperCase() }} <span class="tab-count">{{ scenCount(s) }}</span>
      </button>
    </div>

    <section class="panel">
      <table class="grid">
        <thead>
          <tr>
            <th>场景</th><th>模型</th><th>约束处理</th><th v-if="hasVariant">奖励模式</th><th>种子</th>
            <th class="num" title="success_rate = 测试实例中 M 个服务全部合法放置且无违规的比例">success_rate</th><th class="num" title="AR = average resource utilization（平均资源利用率，优化目标）">AR</th><th class="num">ILP AR</th><th class="num" title="AR gap = ILP AR − AR">AR gap</th>
            <th class="num">容量违规</th><th class="num">冲突违规</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(r, i) in rows" :key="r.scenario + r.algo + r.variant" :class="{ 'row-sep': firstOfScen(i) }">
            <td><span v-if="firstOfScen(i)" class="scen" :class="`scen--${r.scenario}`">{{ r.scenario.toUpperCase() }}</span></td>
            <td><b v-if="firstOfLearner(i)">{{ learnerLabel(r.algo) }}</b></td>
            <td :class="{ dim: mechLabel(r.algo) === '无约束' }">{{ mechLabel(r.algo) }}</td>
            <td v-if="hasVariant"><span v-if="r.variant" class="tag">{{ r.variant }}</span></td>
            <td>
              <span class="seeds">
                <component :is="runLink(r, s) ? 'a' : 'span'" v-for="s in r.seeds" :key="`${s.seed}-${s.exp_id}`" :href="runLink(r, s)"
                  class="seed" :class="`seed--${s.status}`" :title="seedTitle(s)">
                  {{ s.status === "running" && s.progress_pct != null ? Math.round(s.progress_pct) : s.seed ?? "·" }}
                </component>
              </span>
              <span class="dim small">{{ r.n_done }}/{{ r.n_total }}</span>
            </td>
            <td class="num">{{ pm(r.test_success_rate_mean, r.test_success_rate_std, pct) }}</td>
            <td class="num">{{ pm(r.test_ar_mean_mean, r.test_ar_mean_std) }}</td>
            <td class="num dim">{{ fmt(r.ilp_ar_mean) }}</td>
            <td class="num">{{ fmt(gap(r)) }}</td>
              <td class="num" :class="{ warn: r.test_cap_viol_rate_mean }">{{ pct(r.test_cap_viol_rate_mean) }}</td>
              <td class="num" :class="{ warn: r.test_conflict_viol_rate_mean }">{{ pct(r.test_conflict_viol_rate_mean) }}</td>
          </tr>
        </tbody>
      </table>
      <div class="legend">
        <span><i class="seed seed--done"></i>已完成（可点击查看）</span>
        <span><i class="seed seed--running"></i>运行中（数字为训练进度 %）</span>
        <span><i class="seed seed--queued"></i>排队中</span>
        <span v-if="data.skipped"><i class="seed seed--skipped"></i>未运行（批次已结束）</span>
        <span><i class="seed seed--stopped"></i>已中断</span>
        <span>指标为已完成种子的均值 ± 标准差；AR = average resource utilization；AR gap = ILP AR − AR；success_rate = M 个服务全部合法放置且无违规的测试实例比例</span>
      </div>
    </section>
  </template>
</template>
