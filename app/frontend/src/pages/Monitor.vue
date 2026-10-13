<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue";
import { store } from "../store.js";
import { getBatches, getBatch, getQueue } from "../api.js";
import { algoLabel, algoIndex, scenarioIndex, variantLabel, netLabel, reporting, reportText, reportPct } from "../labels.js";
import { langRef as curLang } from "../i18n.js";
import { secToClock } from "../format.js";
import HardwarePanel from "../components/HardwarePanel.vue";

const procs = computed(() => store.system?.processes || []);
const training = computed(() =>
  procs.value
    .filter((p) => p.scenario && p.algo)
    .sort((a, b) =>
      (a.batch ?? "").localeCompare(b.batch ?? "") ||
      scenarioIndex(a.scenario) - scenarioIndex(b.scenario) ||
      algoIndex(a.algo) - algoIndex(b.algo) ||
      (a.variant ?? "").localeCompare(b.variant ?? "")));
const others = computed(() => procs.value.filter((p) => !(p.scenario && p.algo)));
// other processes grouped by command line (a regret diagnostic = 1 main + 48 solver workers)
const othersGrouped = computed(() => {
  const m = new Map();
  for (const p of others.value) {
    const g = m.get(p.cmd) ?? { label: p.label, cmd: p.cmd, n: 0, cpu: 0, elapsed: 0, pid: p.pid };
    g.n += 1; g.cpu += p.cpu_percent || 0; g.elapsed = Math.max(g.elapsed, p.elapsed_seconds || 0); g.pid = Math.min(g.pid, p.pid);
    m.set(p.cmd, g);
  }
  return [...m.values()].sort((a, b) => b.cpu - a.cpu);
});
// other processes worth showing: not the panel's own services, not shell wrappers
const SKIP = /app\.backend\.(main|monitor)|uvicorn app\.backend|^(\/bin\/)?(ba)?sh -c |queue_runner\.py run/;
const othersShown = computed(() => othersGrouped.value.filter((g) => !SKIP.test(g.cmd)));
const loadPct = computed(() => {
  const s = store.system;
  return s?.load_avg?.["1m"] ? Math.round((100 * s.load_avg["1m"]) / (s.cpu_count || 1)) : null;
});
const diagCpu = computed(() => others.value.filter((p) => p.cmd.includes("diag_regret.py")).reduce((s, p) => s + (p.cpu_percent || 0), 0));
// batches whose training is over but whose report (regret diagnostics) is still being built
const reportBatches = ref([]);
// serial experiment queue (read-only)
const queue = ref({ items: [], paused: false, runner: "unknown" });
const QSTATUS = { queued: ["排队中", "queued", "badge--off"], waiting: ["等待前置任务", "waiting for prerequisite", "badge--warn"],
  running: ["运行中", "running", "badge--run"], done: ["完成", "done", "badge--ok"], failed: ["失败", "failed", "badge--bad"],
  interrupted: ["中断", "interrupted", "badge--bad"] };
const qLabel = (st) => (QSTATUS[st] ?? [st, st])[curLang.value === "en" ? 1 : 0];
const qClass = (st) => (QSTATUS[st] ?? [])[2] ?? "";
const qDur = (it) => {
  if (!it.started) return "—";
  const end = it.finished ? new Date(it.finished.replace(" ", "T")) : new Date();
  return secToClock(Math.max(0, (end - new Date(it.started.replace(" ", "T"))) / 1000));
};
// queued jobs of every running batch
const queued = ref([]);
let qTimer = null;
async function loadQueued() {
  const { batches } = await getBatches(false);
  reportBatches.value = batches.filter(reporting);
  queue.value = await getQueue();
  const out = [];
  for (const b of batches) {
    const d = await getBatch(b.batch_name);
    for (const r of d?.rows || [])
      for (const s of r.seeds)
        if (s.status === "queued") out.push({ batch: b.batch_name, scenario: r.scenario, algo: r.algo, variant: r.variant, seed: s.seed });
  }
  queued.value = out;
}
onMounted(() => { loadQueued(); qTimer = setInterval(loadQueued, 10000); });
onUnmounted(() => clearInterval(qTimer));
const queuedByBatch = computed(() => {
  const m = new Map();
  for (const q of queued.value) {
    if (!m.has(q.batch)) m.set(q.batch, []);
    m.get(q.batch).push(q);
  }
  return [...m.entries()];
});

const gpus = computed(() => store.system?.gpus || []);
const frozenN = computed(() => training.value.filter((p) => p.frozen).length);
const totalCpu = computed(() => training.value.reduce((s, p) => s + (p.cpu_percent || 0), 0));
</script>

<template>
  <div class="page-head">
    <h1>进程监控</h1>
    <div class="page-sub" data-no-i18n>{{ curLang === "en" ? "Processes of this project on this machine, refreshed every 5 s" : "本机正在运行的项目进程，每 5 秒刷新" }}
      · {{ curLang === "en" ? "queue" : "实验队列" }}：<template v-if="queue.items.length">{{ queue.items.filter((x) => ["queued", "waiting"].includes(x.status)).length }} {{ curLang === "en" ? "waiting" : "项等待" }}</template><template v-else>{{ curLang === "en" ? "empty" : "空" }}</template>
      · {{ curLang === "en" ? "runner" : "执行器" }} {{ queue.runner }}</div>
  </div>

  <div class="kpis" data-no-i18n>
    <div class="kpi"><div class="kpi-label">{{ curLang === "en" ? "Training jobs" : "训练任务" }}</div><div class="kpi-value">{{ training.length - frozenN }}</div>
      <div v-if="frozenN" class="kpi-foot">{{ curLang === "en" ? `running · ${frozenN} frozen (SIGSTOP, resumable)` : `运行中 · 另有 ${frozenN} 个冻结（SIGSTOP，可恢复）` }}</div></div>
    <div class="kpi"><div class="kpi-label">CPU</div><div class="kpi-value">{{ loadPct === null ? "—" : loadPct + "%" }}</div>
      <div class="kpi-foot">{{ curLang === "en" ? "training" : "训练" }} {{ (totalCpu / 100).toFixed(1) }}<template v-if="diagCpu > 0"> · {{ curLang === "en" ? "diagnostics" : "诊断" }} {{ (diagCpu / 100).toFixed(1) }}</template> / {{ store.system?.cpu_count ?? "—" }} {{ curLang === "en" ? "cores" : "核" }}</div></div>
    <div v-if="gpus.length" class="kpi"><div class="kpi-label">GPU</div>
      <div class="kpi-value kpi-value--sm">{{ gpus.map((g) => (g.util ?? "—") + "%").join(" · ") }}</div>
      <div class="kpi-foot">{{ gpus.map((g) => (g.mem_used != null ? (g.mem_used / 1024).toFixed(1) : "—") + " GB").join(" · ") }}</div></div>
  </div>

  <HardwarePanel />

  <section class="panel">
    <div class="panel-head"><h2>训练任务</h2></div>
    <div v-if="!training.length" class="empty">当前没有训练任务在运行</div>
    <div class="table-scroll" v-else><table class="grid">
      <thead>
        <tr><th>批次</th><th>场景</th><th>算法</th><th>网络 · 设备</th><th>种子</th>
          <th class="w-progress">训练进度</th><th class="num" title="最近约 30 万步的速度（由日志反推）；悬停单元格可看自启动以来的平均速度">步/秒</th><th class="num">已运行</th></tr>
      </thead>
      <tbody>
        <tr v-for="p in training" :key="p.pid" :title="`exp_id ${p.exp_id ?? '—'} · PID ${p.pid}`">
          <td><a v-if="p.batch" :href="`#/batches/${p.batch}`">{{ p.batch }}</a><span v-else-if="p.paper_version?.endsWith('-check')" class="dim" data-no-i18n>v{{ p.paper_version.replace(/-check$/, "") }} {{ curLang === "en" ? "check (test)" : "验证测试" }}</span><span v-else class="dim">手动启动</span></td>
          <td><span class="scen" :class="`scen--${p.scenario}`">{{ p.scenario.toUpperCase() }}</span></td>
          <td>{{ algoLabel(p.algo) }}<span v-if="p.variant" class="tag">{{ variantLabel(p.variant) }}</span></td>
          <td>{{ netLabel(p.net) }} · <span :class="{ dim: p.device === 'CPU' }">{{ p.device ?? "—" }}</span></td>
          <td>{{ p.seed ?? "—" }}<span v-if="p.frozen" class="badge badge--warn" style="margin-left:6px">冻结中</span></td>
          <td>
            <div class="bar"><div class="bar-fill" :style="{ width: (p.progress_pct ?? 0) + '%' }"></div></div>
            <span class="bar-text">{{ p.progress_pct ?? 0 }}%</span>
          </td>
          <td class="num" :title="p.steps_per_sec_avg != null ? `自启动以来平均 ${p.steps_per_sec_avg.toLocaleString()} 步/秒` : ''">{{ p.frozen ? "—" : p.steps_per_sec?.toLocaleString() ?? "—" }}</td>
          <td class="num">{{ secToClock(p.elapsed_seconds) }}</td>
        </tr>
      </tbody>
    </table></div>
  </section>

  <section v-if="reportBatches.length" class="panel">
    <div class="panel-head"><h2>生成报告（训练结束后）</h2></div>
    <div class="table-scroll"><table class="grid">
      <thead><tr><th>批次</th><th class="w-progress">进度</th><th>阶段</th></tr></thead>
      <tbody>
        <tr v-for="b in reportBatches" :key="b.batch_name">
          <td><a :href="`#/batches/${b.batch_name}`">{{ b.batch_name }}</a></td>
          <td><div class="bar"><div class="bar-fill bar-fill--report" :style="{ width: reportPct(b.report) + '%' }"></div></div>
            <span class="bar-text">{{ reportPct(b.report).toFixed(0) }}%</span></td>
          <td data-no-i18n>{{ reportText(b.report, curLang) }}</td>
        </tr>
      </tbody>
    </table></div>
  </section>

  <section v-if="queue.items.length" class="panel" data-no-i18n>
    <div class="panel-head"><h2>{{ curLang === "en" ? "Experiment queue" : "实验队列" }}</h2>
      <span class="dim small">{{ curLang === "en" ? "serial, managed with src/scripts/queue_runner.py" : "串行执行，由 src/scripts/queue_runner.py 管理" }}<span v-if="queue.paused"> · <b>{{ curLang === "en" ? "paused" : "已暂停" }}</b></span></span>
    </div>
    <div class="table-scroll">
      <table class="grid">
        <thead><tr><th>#</th><th>{{ curLang === "en" ? "Item" : "任务" }}</th><th>{{ curLang === "en" ? "Status" : "状态" }}</th>
          <th>{{ curLang === "en" ? "Waits for" : "前置条件" }}</th><th>{{ curLang === "en" ? "Started" : "开始" }}</th>
          <th class="num">{{ curLang === "en" ? "Duration" : "用时" }}</th></tr></thead>
        <tbody>
          <tr v-for="(it, i) in queue.items" :key="it.id" :title="(it.cmd || []).join(' ')">
            <td class="num dim">{{ i + 1 }}</td>
            <td><a :href="`#/versions/${it.id}`"><b>{{ it.id }}</b></a></td>
            <td><span class="badge" :class="qClass(it.status)">{{ qLabel(it.status) }}</span><span v-if="it.exit" class="dim small"> exit {{ it.exit }}</span></td>
            <td class="mono small">{{ it.wait_for || "—" }}</td>
            <td class="nowrap">{{ it.started || "—" }}</td>
            <td class="num">{{ qDur(it) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>

  <section v-if="othersShown.length" class="panel">
    <details>
      <summary class="panel-head" style="cursor:pointer"><h2 style="display:inline">其他项目进程</h2> <span class="tab-count">{{ othersShown.length }}</span></summary>
      <div class="table-scroll"><table class="grid">
        <thead><tr><th>进程</th><th class="num">进程数</th><th class="num" title="同一命令下所有进程的 CPU 之和，按单核计：100% = 占满 1 个核（整机共 56 核）">CPU（单核合计）</th><th class="num">已运行</th></tr></thead>
        <tbody>
          <tr v-for="g in othersShown" :key="g.cmd" :title="`${g.cmd} · PID ${g.pid}`">
            <td>{{ g.label }}</td>
            <td class="num">{{ g.n }}</td>
            <td class="num">{{ g.cpu.toFixed(0) }}%</td>
            <td class="num">{{ secToClock(g.elapsed) }}</td>
          </tr>
        </tbody>
      </table></div>
    </details>
  </section>
</template>
