<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue";
import { store } from "../store.js";
import { getBatches, getBatch } from "../api.js";
import { algoLabel, algoIndex, scenarioIndex } from "../labels.js";
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
// queued jobs of every running batch
const queued = ref([]);
let qTimer = null;
async function loadQueued() {
  const { batches } = await getBatches(false);
  const out = [];
  for (const b of batches) {
    const d = await getBatch(b.batch_name);
    for (const r of d?.rows || [])
      for (const s of r.seeds)
        if (s.status === "queued") out.push({ batch: b.batch_name, scenario: r.scenario, algo: r.algo, variant: r.variant, seed: s.seed });
  }
  queued.value = out;
}
onMounted(() => { loadQueued(); qTimer = setInterval(loadQueued, 15000); });
onUnmounted(() => clearInterval(qTimer));
const queuedByBatch = computed(() => {
  const m = new Map();
  for (const q of queued.value) {
    if (!m.has(q.batch)) m.set(q.batch, []);
    m.get(q.batch).push(q);
  }
  return [...m.entries()];
});

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
    <div class="kpi"><div class="kpi-label">系统负载 1/5/15 分钟</div>
      <div class="kpi-value kpi-value--sm">{{ store.system ? ["1m", "5m", "15m"].map((k) => store.system.load_avg[k]?.toFixed(1)).join(" / ") : "—" }}</div></div>
  </div>

  <section class="panel">
    <div class="panel-head"><h2>训练任务</h2></div>
    <div v-if="!training.length" class="empty">当前没有训练任务在运行</div>
    <table v-else class="grid">
      <thead>
        <tr><th>批次</th><th>场景</th><th>算法</th><th>奖励模式</th><th>种子</th><th>exp_id</th>
          <th class="w-progress">训练进度</th><th class="num">CPU</th><th class="num">已运行</th><th class="num">PID</th></tr>
      </thead>
      <tbody>
        <tr v-for="p in training" :key="p.pid">
          <td><a v-if="p.batch" :href="`#/batches/${p.batch}`">{{ p.batch }}</a><span v-else class="dim">手动启动</span></td>
          <td><span class="scen" :class="`scen--${p.scenario}`">{{ p.scenario.toUpperCase() }}</span></td>
          <td>{{ algoLabel(p.algo) }}</td>
          <td><span v-if="p.variant" class="tag">{{ p.variant }}</span><span v-else class="dim">—</span></td>
          <td>{{ p.seed ?? "—" }}</td>
          <td class="mono dim">{{ p.exp_id ?? "—" }}</td>
          <td>
            <div class="bar"><div class="bar-fill" :style="{ width: (p.progress_pct ?? 0) + '%' }"></div></div>
            <span class="bar-text">{{ p.progress_pct ?? 0 }}%</span>
          </td>
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
          {{ algoLabel(q.algo) }}<span v-if="q.variant" class="tag">{{ q.variant }}</span><span v-if="q.seed != null" class="dim small"> s{{ q.seed }}</span>
        </span>
      </div>
    </template>
  </section>

  <section class="panel">
    <div class="panel-head"><h2>其他项目进程</h2></div>
    <table class="grid">
      <thead><tr><th>进程</th><th class="num">CPU</th><th class="num">已运行</th><th class="num">PID</th></tr></thead>
      <tbody>
        <tr v-for="p in others" :key="p.pid">
          <td :title="p.cmd">{{ p.label }}</td>
          <td class="num">{{ p.cpu_percent.toFixed(0) }}%</td>
          <td class="num">{{ secToClock(p.elapsed_seconds) }}</td>
          <td class="num mono dim">{{ p.pid }}</td>
        </tr>
      </tbody>
    </table>
  </section>
</template>
