<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue";
import { store } from "../store.js";
import { getBatches, getBatch } from "../api.js";
import { algoLabel, algoIndex, scenarioIndex, variantLabel, netLabel, reporting, reportText, reportPct } from "../labels.js";
import { langRef as curLang } from "../i18n.js";
import { secToClock } from "../format.js";

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
const diagCpu = computed(() => others.value.filter((p) => p.cmd.includes("diag_regret.py")).reduce((s, p) => s + (p.cpu_percent || 0), 0));
// batches whose training is over but whose report (regret diagnostics) is still being built
const reportBatches = ref([]);
// queued jobs of every running batch
const queued = ref([]);
let qTimer = null;
async function loadQueued() {
  const { batches } = await getBatches(false);
  reportBatches.value = batches.filter(reporting);
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
const totalCpu = computed(() => training.value.reduce((s, p) => s + (p.cpu_percent || 0), 0));
</script>

<template>
  <div class="page-head">
    <h1>训练监控</h1>
    <div class="page-sub">本机正在运行的项目进程，每 5 秒刷新</div>
  </div>

  <div class="kpis">
    <div class="kpi"><div class="kpi-label">训练任务</div><div class="kpi-value">{{ training.length }}</div></div>
    <div class="kpi"><div class="kpi-label">训练占用 CPU</div><div class="kpi-value">{{ (totalCpu / 100).toFixed(1) }}</div><div class="kpi-foot">核（共 {{ store.system?.cpu_count ?? "—" }}）</div></div>
    <div v-if="diagCpu > 0" class="kpi"><div class="kpi-label">报告诊断占用 CPU</div><div class="kpi-value">{{ (diagCpu / 100).toFixed(1) }}</div><div class="kpi-foot">核（共 {{ store.system?.cpu_count ?? "—" }}）</div></div>
    <div class="kpi"><div class="kpi-label">系统负载 1/5/15 分钟</div>
      <div class="kpi-value kpi-value--sm">{{ store.system ? ["1m", "5m", "15m"].map((k) => store.system.load_avg[k]?.toFixed(1)).join(" / ") : "—" }}</div></div>
    <div v-for="g in gpus" :key="g.index" class="kpi" :title="g.name">
      <div class="kpi-label">GPU {{ g.index }} 利用率</div>
      <div class="kpi-value">{{ g.util ?? "—" }}%</div>
      <div class="kpi-foot">显存 {{ g.mem_used != null ? (g.mem_used / 1024).toFixed(1) : "—" }} / {{ g.mem_total != null ? (g.mem_total / 1024).toFixed(0) : "—" }} GB · {{ g.temp ?? "—" }}°C</div>
    </div>
  </div>

  <section class="panel">
    <div class="panel-head"><h2>训练任务</h2></div>
    <div v-if="!training.length" class="empty">当前没有训练任务在运行</div>
    <table v-else class="grid">
      <thead>
        <tr><th>批次</th><th>场景</th><th>算法</th><th title="奖励模式（早期试跑）或策略网络（v4.4.3）">变体</th><th>网络</th><th>设备</th><th>种子</th><th>exp_id</th>
          <th class="w-progress">训练进度</th><th class="num" title="日志里最近一次 [train] 记录的平均速度（自启动以来）">步/秒</th><th class="num" title="单个进程占用的 CPU，按单核计：100% = 占满 1 个核（整机共 56 核）。GPU 训练的进程同样会占满 1 个核：环境推进、动作掩码与 rollout 循环都在 CPU 上">CPU（单核）</th><th class="num">已运行</th><th class="num">PID</th></tr>
      </thead>
      <tbody>
        <tr v-for="p in training" :key="p.pid">
          <td><a v-if="p.batch" :href="`#/batches/${p.batch}`">{{ p.batch }}</a><span v-else class="dim">手动启动</span></td>
          <td><span class="scen" :class="`scen--${p.scenario}`">{{ p.scenario.toUpperCase() }}</span></td>
          <td>{{ algoLabel(p.algo) }}</td>
          <td><span v-if="p.variant" class="tag">{{ variantLabel(p.variant) }}</span><span v-else class="dim">—</span></td>
          <td>{{ netLabel(p.net) }}</td>
          <td :class="{ dim: p.device === 'CPU' }">{{ p.device ?? "—" }}</td>
          <td>{{ p.seed ?? "—" }}</td>
          <td class="mono dim">{{ p.exp_id ?? "—" }}</td>
          <td>
            <div class="bar"><div class="bar-fill" :style="{ width: (p.progress_pct ?? 0) + '%' }"></div></div>
            <span class="bar-text">{{ p.progress_pct ?? 0 }}%</span>
          </td>
          <td class="num">{{ p.steps_per_sec?.toLocaleString() ?? "—" }}</td>
          <td class="num">{{ p.cpu_percent.toFixed(0) }}%</td>
          <td class="num">{{ secToClock(p.elapsed_seconds) }}</td>
          <td class="num mono dim">{{ p.pid }}</td>
        </tr>
      </tbody>
    </table>
    <div class="note">训练进度来自各任务日志中最近一次 [train] 记录（旧实现每 20 万步、paper_rl 每 10 万步写一次），刚启动或处于 ILP / 评估阶段时显示 0%。</div>
  </section>

  <section class="panel">
    <div class="panel-head"><h2>排队中 <span class="tab-count">{{ queued.length }}</span></h2></div>
    <div v-if="!queued.length" class="empty">没有排队中的任务</div>
    <template v-for="[batch, items] in queuedByBatch" :key="batch">
      <div class="queue-head"><a :href="`#/batches/${batch}`">{{ batch }}</a> · {{ items.length }} 个任务，调度器按空闲 CPU 依次启动</div>
      <div class="queue-chips">
        <span v-for="(q, k) in items" :key="k" class="queue-chip">
          <span class="scen" :class="`scen--${q.scenario}`">{{ q.scenario.toUpperCase() }}</span>
          {{ algoLabel(q.algo) }}<span v-if="q.variant" class="tag">{{ variantLabel(q.variant) }}</span><span v-if="q.seed != null" class="dim small"> s{{ q.seed }}</span>
        </span>
      </div>
    </template>
  </section>

  <section v-if="reportBatches.length" class="panel">
    <div class="panel-head"><h2>生成报告（训练结束后）</h2></div>
    <table class="grid">
      <thead><tr><th>批次</th><th class="w-progress">进度</th><th>阶段</th></tr></thead>
      <tbody>
        <tr v-for="b in reportBatches" :key="b.batch_name">
          <td><a :href="`#/batches/${b.batch_name}`">{{ b.batch_name }}</a></td>
          <td><div class="bar"><div class="bar-fill bar-fill--report" :style="{ width: reportPct(b.report) + '%' }"></div></div>
            <span class="bar-text">{{ reportPct(b.report).toFixed(0) }}%</span></td>
          <td data-no-i18n>{{ reportText(b.report, curLang) }}</td>
        </tr>
      </tbody>
    </table>
  </section>

  <section class="panel">
    <div class="panel-head"><h2>其他项目进程</h2></div>
    <table class="grid">
      <thead><tr><th>进程</th><th class="num">进程数</th><th class="num" title="同一命令下所有进程的 CPU 之和，按单核计：100% = 占满 1 个核（整机共 56 核）">CPU（单核合计）</th><th class="num">已运行</th><th class="num">PID</th></tr></thead>
      <tbody>
        <tr v-for="g in othersGrouped" :key="g.cmd">
          <td :title="g.cmd">{{ g.label }}</td>
          <td class="num">{{ g.n }}</td>
          <td class="num">{{ g.cpu.toFixed(0) }}%</td>
          <td class="num">{{ secToClock(g.elapsed) }}</td>
          <td class="num mono dim">{{ g.pid }}</td>
        </tr>
      </tbody>
    </table>
  </section>
</template>
