<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from "vue";
import { store, trainingProcs, spaceLabel } from "../store.js";
import { getBatches, getExperiments, getReports } from "../api.js";
import GapTrend from "../components/GapTrend.vue";
import { algoLabel, statusLabel, statusClass, reporting, reportText, reportPct } from "../labels.js";
import { langRef as curLang } from "../i18n.js";
import { pct, fmt, shortTime } from "../format.js";

const batches = ref([]);
const exps = ref([]);
const reports = ref([]);
let timer = null;
const go = (h) => (window.location.hash = h);

async function load() {
  if (!store.viewBranch) return;
  batches.value = (await getBatches(true, store.viewBranch)).batches;
  exps.value = await getExperiments(store.viewBranch);
  reports.value = (await getReports()).reports;
}
onMounted(() => { timer = setInterval(load, 15000); });
watch(() => store.viewBranch, load, { immediate: true });
onUnmounted(() => clearInterval(timer));

const runningBatches = computed(() => batches.value.filter((b) => b.status === "running" || reporting(b)));
const recentBatches = computed(() => batches.value.filter((b) => b.status !== "running" && !reporting(b)).slice(0, 5));
const recentRuns = computed(() =>
  [...exps.value].sort((a, b) => (b.created_at ?? "").localeCompare(a.created_at ?? "")).slice(0, 10));
// best three-scenario mean so far (new variant of each version) and its distance to the 5% target
const bestRep = computed(() => reports.value.filter((r) => r.new_avg != null)
  .reduce((a, r) => (!a || r.new_avg < a.new_avg ? r : a), null));
const latestRep = computed(() => reports.value.find((r) => r.new_avg != null) ?? null);
const en = computed(() => curLang.value === "en");
const loadPct = computed(() => {
  const s = store.system;
  return s?.load_avg?.["1m"] ? Math.round((100 * s.load_avg["1m"]) / s.cpu_count) : null;
});
</script>

<template>
  <div class="page-head">
    <h1>总览</h1>
    <div class="page-sub">{{ spaceLabel(store.viewBranch) }} 的批次与最新结果；训练任务、CPU 为本机全局状态</div>
  </div>

  <div class="kpis">
    <a class="kpi" href="#/monitor">
      <div class="kpi-label">训练中任务</div>
      <div class="kpi-value">{{ trainingProcs().length }}</div>
      <div class="kpi-foot">查看训练监控 →</div>
    </a>
    <a class="kpi" href="#/versions">
      <div class="kpi-label">进行中的批次</div>
      <div class="kpi-value">{{ runningBatches.length }}</div>
      <div class="kpi-foot">训练或生成报告中 · 实验版本 →</div>
    </a>
    <a v-if="bestRep" class="kpi" :href="`#/versions/${bestRep.version}`" data-no-i18n>
      <div class="kpi-label">{{ en ? "Best 3-scenario mean gap" : "目前最好的三场景平均 gap" }}</div>
      <div class="kpi-value">{{ bestRep.new_avg.toFixed(1) }}%</div>
      <div class="kpi-foot">{{ bestRep.version }} · {{ en ? `${(bestRep.new_avg - 5).toFixed(1)}pp above the 5% target` : `距目标 5% 还差 ${(bestRep.new_avg - 5).toFixed(1)}pp` }} →</div>
    </a>
    <a v-if="latestRep" class="kpi" :href="`#/versions/${latestRep.version}`" data-no-i18n>
      <div class="kpi-label">{{ en ? "Latest version" : "最新版本" }} · {{ latestRep.version }}</div>
      <div class="kpi-value kpi-value--sm">LT {{ latestRep.gaps[latestRep.new_model].LT?.toFixed(1) }}% · EQ {{ latestRep.gaps[latestRep.new_model].EQ?.toFixed(1) }}% · GT {{ latestRep.gaps[latestRep.new_model].GT?.toFixed(1) }}%</div>
      <div class="kpi-foot">{{ en ? (latestRep.en?.new_model ?? "new variant") : latestRep.new_model }}</div>
    </a>
    <div class="kpi">
      <div class="kpi-label">CPU 负载</div>
      <div class="kpi-value">{{ loadPct === null ? "—" : loadPct + "%" }}</div>
      <div class="kpi-foot">{{ store.system?.cpu_count ?? "—" }} 核</div>
    </div>
  </div>

  <section class="panel">
    <div class="panel-head"><h2>测试集相对 gap 走势</h2><a href="#/versions">全部版本 →</a></div>
    <GapTrend :reports="reports" />
  </section>

  <section class="panel">
    <div class="panel-head"><h2>进行中的批次</h2><a href="#/batches">全部批次 →</a></div>
    <div v-if="!runningBatches.length" class="empty">当前没有运行中的批次</div>
    <div class="table-scroll" v-else><table class="grid">
      <thead><tr><th>批次</th><th>版本</th><th class="w-progress">进度</th><th>运行 / 排队</th><th>开始时间</th></tr></thead>
      <tbody>
        <tr v-for="b in runningBatches" :key="b.batch_name" class="clickable" @click="go(`#/batches/${b.batch_name}`)">
          <td><a :href="`#/batches/${b.batch_name}`">{{ b.batch_name }}</a></td>
          <td>{{ b.version ?? "—" }}</td>
          <td>
            <template v-if="reporting(b)">
              <div class="bar"><div class="bar-fill bar-fill--report" :style="{ width: reportPct(b.report) + '%' }"></div></div>
              <span class="bar-text" data-no-i18n>{{ reportText(b.report, curLang) }}</span>
            </template>
            <template v-else>
              <div class="bar"><div class="bar-fill" :style="{ width: b.overall_pct + '%' }"></div></div>
              <span class="bar-text">{{ b.done }}/{{ b.total_runs }}（{{ b.overall_pct }}%）</span>
            </template>
          </td>
          <td>{{ b.running }} / {{ b.queued }}</td>
          <td>{{ shortTime(b.started_at) }}</td>
        </tr>
      </tbody>
    </table></div>
  </section>

  <div class="two-col">
    <section class="panel">
      <div class="panel-head"><h2>最近完成的运行</h2><a href="#/results">运行明细 →</a></div>
      <div class="table-scroll"><table class="grid">
        <thead><tr><th>时间</th><th>场景</th><th>算法</th><th>批次</th><th class="num" title="success_rate = 测试实例中 M 个服务全部合法放置且无违规的比例">success_rate</th><th class="num" title="AR = average resource utilization（平均资源利用率，优化目标）">AR</th></tr></thead>
        <tbody>
          <tr v-for="r in recentRuns" :key="r.scenario + r.algo + r.run">
            <td>{{ shortTime(r.created_at) }}</td>
            <td><span class="scen" :class="`scen--${r.scenario}`">{{ r.scenario.toUpperCase() }}</span></td>
            <td><a :href="`#/run/${store.viewBranch}/${r.scenario}/${r.algo}/${r.run}`">{{ algoLabel(r.algo) }}</a>
              <span v-if="r.variant" class="tag">{{ r.variant }}</span></td>
            <td class="dim">{{ r.batch ?? "—" }}</td>
            <td class="num">{{ pct(r.test_success_rate) }}</td>
            <td class="num">{{ fmt(r.test_ar_mean) }}</td>
          </tr>
        </tbody>
      </table></div>
    </section>

    <section class="panel">
      <div class="panel-head"><h2>最近结束的批次</h2><a href="#/batches">全部批次 →</a></div>
      <div class="table-scroll"><table class="grid">
        <thead><tr><th>批次</th><th>状态</th><th>完成</th><th>最后更新</th></tr></thead>
        <tbody>
          <tr v-for="b in recentBatches" :key="b.batch_name">
            <td><a :href="`#/batches/${b.batch_name}`">{{ b.batch_name }}</a></td>
            <td><span class="badge" :class="statusClass(b.status)">{{ statusLabel(b.status) }}</span></td>
            <td>{{ b.done }}/{{ b.total_runs }}</td>
            <td>{{ shortTime(b.last_updated) }}</td>
          </tr>
        </tbody>
      </table></div>
    </section>
  </div>
</template>
