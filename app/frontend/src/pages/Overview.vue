<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from "vue";
import { store, trainingProcs, stageOf, currentStage } from "../store.js";
import { getBatches, getReports, getQueue } from "../api.js";
import GapTrend from "../components/GapTrend.vue";
import { algoLabel, statusLabel, statusClass, reporting, reportText, reportPct } from "../labels.js";
import { langRef as curLang } from "../i18n.js";
import { pct, fmt, shortTime } from "../format.js";

const batches = ref([]);
const reports = ref([]);
let timer = null;
const go = (h) => (window.location.hash = h);

const queue = ref({ items: [], paused: false, runner: "unknown" });
async function load() {
  batches.value = (await getBatches(true)).batches;
  reports.value = (await getReports()).reports;
  queue.value = await getQueue();
}
onMounted(() => { load(); timer = setInterval(load, 15000); });
onUnmounted(() => clearInterval(timer));

// every space: a running batch shows wherever its data lives; the finished list keeps to the current stage's formal batches
const runningBatches = computed(() => batches.value.filter((b) => b.status === "running" || reporting(b)));
const recentBatches = computed(() => batches.value.filter((b) => b.status !== "running" && !reporting(b) &&
  (b.purpose ?? "main") === "main" && stageOf(b.branch) === currentStage()).slice(0, 5));
// experiment queue: what is still to come (src/scripts/queue_runner.py)
const pendingQueue = computed(() => queue.value.items.filter((x) => ["queued", "waiting", "running"].includes(x.status)));
const Q = { queued: ["排队中", "badge--off"], waiting: ["等待前置任务", "badge--warn"], running: ["运行中", "badge--run"] };
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
    <div class="page-sub">正在进行的批次、实验队列与最新结果</div>
  </div>

  <div class="kpis">
    <a class="kpi" href="#/monitor">
      <div class="kpi-label">训练中任务</div>
      <div class="kpi-value">{{ trainingProcs().length }}</div>
      <div class="kpi-foot">进程监控 →</div>
    </a>
    <a class="kpi" href="#/batches">
      <div class="kpi-label">进行中的批次</div>
      <div class="kpi-value">{{ runningBatches.length }}</div>
      <div class="kpi-foot">训练或生成报告中 · 批次 →</div>
    </a>
    <a class="kpi" href="#/monitor">
      <div class="kpi-label">实验队列</div>
      <div class="kpi-value">{{ pendingQueue.length }}<span v-if="queue.paused" class="kpi-unit">已暂停</span></div>
      <div class="kpi-foot">待执行的项目 →</div>
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
  </div>

  <section v-if="pendingQueue.length || queue.paused" class="panel">
    <div class="panel-head"><h2>实验队列</h2><a href="#/monitor">进程监控 →</a></div>
    <div v-if="queue.paused" class="notice">队列已暂停：排队的项目不会自动启动（<code>queue_runner.py unpause</code> 恢复）</div>
    <div v-if="!pendingQueue.length" class="empty">队列中没有待执行的项目</div>
    <div class="table-scroll" v-else><table class="grid">
      <thead><tr><th>项目</th><th>状态</th><th>前置条件</th><th>开始</th></tr></thead>
      <tbody>
        <tr v-for="it in pendingQueue" :key="it.id" :title="(it.cmd || []).join(' ')">
          <td><b>{{ it.id }}</b></td>
          <td><span class="badge" :class="Q[it.status][1]">{{ Q[it.status][0] }}</span></td>
          <td class="mono small">{{ it.wait_for || "—" }}</td>
          <td>{{ it.started || "—" }}</td>
        </tr>
      </tbody>
    </table></div>
  </section>

  <section class="panel">
    <div class="panel-head"><h2>测试集相对 gap 走势</h2><a href="#/versions">版本报告 →</a></div>
    <GapTrend :reports="reports" />
  </section>

  <section class="panel">
    <div class="panel-head"><h2>进行中的批次</h2><a href="#/batches">批次 →</a></div>
    <div v-if="!runningBatches.length" class="empty">当前没有运行中的批次</div>
    <div class="table-scroll" v-else><table class="grid">
      <thead><tr><th>批次</th><th>版本</th><th class="w-progress">进度</th><th>运行 / 排队</th><th>开始时间</th></tr></thead>
      <tbody>
        <tr v-for="b in runningBatches" :key="b.batch_name" class="clickable" @click="go(`#/batches/${b.batch_name}`)">
          <td><a :href="`#/batches/${b.batch_name}`">{{ b.batch_name }}</a>
            <span v-if="b.purpose && b.purpose !== 'main'" class="tag tag--wait">{{ b.purpose === "smoke" ? "冒烟测试" : "测试" }}</span></td>
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

  <section class="panel">
      <div class="panel-head"><h2>最近结束的批次</h2><a href="#/batches">批次 →</a></div>
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
</template>
