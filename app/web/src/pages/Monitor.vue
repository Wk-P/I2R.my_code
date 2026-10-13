<script setup>
// Live monitor: running batches (progress, throughput, ETA), every running run (step / total, steps/s, lambda,
// success of the last 500 episodes, ETA), training processes outside the v2 batches, report stages, GPUs, the
// machine and the other project processes. Refreshes every 5 s.
import { ref, computed, onMounted, onUnmounted } from "vue";
import { t, tl } from "../i18n.js";
import { getLive, getGpuHistory } from "../api.js";
import { dur, when, bytes, steps } from "../format.js";
import Status from "../components/Status.vue";
import Progress from "../components/Progress.vue";
import LiveRuns from "../components/LiveRuns.vue";
import GpuCard from "../components/GpuCard.vue";
import Th from "../components/Th.vue";
import { useSort } from "../sort.js";

const d = ref(null);
const hist = ref([]);
const updated = ref(null);
let timer;
async function load() {
  const x = await getLive();
  if (x) { d.value = x; updated.value = new Date(); }
  const since = hist.value.length ? hist.value[hist.value.length - 1].t : 0;
  const h = await getGpuHistory(since);
  hist.value = [...hist.value, ...(h?.samples || [])].filter((s) => s.t > Date.now() / 1000 - 1800);
}
onMounted(() => { load(); timer = setInterval(load, 5000); });
onUnmounted(() => clearInterval(timer));

const sys = computed(() => d.value?.system || {});
const others = computed(() => (sys.value.processes || []).filter((p) => p.kind !== "train"));
const otherSort = useSort(others, { label: (p) => p.label, pid: (p) => p.pid, cpu: (p) => p.cpu_percent, el: (p) => p.elapsed_seconds, cmd: (p) => p.cmd });
const trainSort = useSort(computed(() => d.value?.other_training || []), {
  pid: (p) => p.pid, scen: (p) => p.scenario, dir: (p) => p.algo, id: (p) => p.exp_id, steps: (p) => p.live?.pct,
  sps: (p) => p.live?.steps_per_sec, dev: (p) => p.device, el: (p) => p.elapsed_seconds });
const at = (sec) => (sec == null ? "—" : when(Date.now() / 1000 + sec));
// one GPU's samples: [{t, clock, max, util, temp, power, thr}]
const gpuSamples = (i) => hist.value.map((x) => ({ t: x.t, ...(x.gpus.find((g) => g.i === i) || {}) }));
</script>

<template>
  <div class="page">
    <div class="head">
      <div>
        <h1>{{ t("实时监控", "Live monitor") }}</h1>
        <div class="head-sub">{{ t("每个运行中 run 的步数、速度、λ 与预计剩余时间；GPU 与机器。每 5 秒刷新。", "Steps, speed, λ and ETA of every running run; GPUs and the machine. Refreshes every 5 s.") }}</div>
      </div>
      <span class="faint small">{{ t("更新于", "updated") }} {{ updated ? updated.toTimeString().slice(0, 8) : "—" }}</span>
    </div>
    <div v-if="!d" class="empty">{{ t("加载中…", "Loading…") }}</div>
    <template v-else>
      <div v-if="!d.batches.length" class="card empty">{{ t("当前没有批次在训练。", "No batch is training right now.") }}</div>
      <div v-for="b in d.batches" :key="b.name" class="card">
        <div class="card-h">
          <h2><router-link :to="`/batches/${b.name}`">{{ b.name }}</router-link></h2><Status s="running" />
          <span class="dim small">{{ tl(b.title) }}</span>
        </div>
        <div class="card-b">
          <div class="row" style="gap: 16px; align-items: center">
            <div style="flex: 1; min-width: 260px"><Progress :b="b" legend /></div>
            <div class="kpi-v">{{ b.pct.toFixed(1) }}%</div>
          </div>
          <div class="kpis" style="margin-top: 12px">
            <div class="kpi"><div class="kpi-l">{{ t("已运行", "Elapsed") }}</div><div class="kpi-v" style="font-size: 18px">{{ dur(b.elapsed) }}</div><div class="kpi-s">{{ t("开始", "started") }} {{ when(b.started_at) }}</div></div>
            <div class="kpi"><div class="kpi-l">{{ t("合计速度", "Throughput") }}</div><div class="kpi-v" style="font-size: 18px">{{ b.steps_per_sec?.toLocaleString() ?? "—" }}</div><div class="kpi-s">{{ t("步/秒（运行中的 run 之和）", "steps/s, running runs") }}</div></div>
            <div class="kpi"><div class="kpi-l">{{ t("当前这批 run 结束", "Running runs done") }}</div><div class="kpi-v" style="font-size: 18px">{{ b.eta ? dur(b.eta.running_done_seconds) : "—" }}</div><div class="kpi-s">{{ t("约", "at") }} {{ at(b.eta?.running_done_seconds) }}</div></div>
            <div class="kpi"><div class="kpi-l">{{ t("预计训练全部结束", "Training done (est.)") }}</div><div class="kpi-v" style="font-size: 18px">{{ b.eta ? dur(b.eta.seconds) : "—" }}</div><div class="kpi-s">{{ t("约", "at") }} {{ at(b.eta?.seconds) }} · {{ t("不含报告", "report not included") }}</div></div>
            <div class="kpi"><div class="kpi-l">{{ t("每个 run", "Per run") }}</div><div class="kpi-v" style="font-size: 18px">{{ steps(b.params?.steps) }}</div><div class="kpi-s">{{ t("步", "steps") }} · {{ t("种子", "seeds") }} {{ b.seeds?.join(", ") }}</div></div>
          </div>
        </div>
        <div class="card-note">{{ t("预计时间：在跑的 run 按各自最近速度；排队的 run 按同一 RL_algorithm 在跑 run 的平均速度、依次占用空出的位置。只是粗估。", "Estimates: running runs at their recent speed; queued runs take freed slots in order at the mean speed of the running runs of the same RL_algorithm. A rough estimate.") }}</div>
      </div>

      <div v-if="d.runs.length" class="card">
        <div class="card-h"><h2>{{ t("运行中的 run", "Running runs") }}</h2><span class="faint small">{{ d.runs.length }} · {{ t("数据来自各 run 的训练日志", "from each run's training log") }}</span></div>
        <div class="card-b"><LiveRuns :runs="d.runs" :batch="d.batches.length > 1" :params="d.batches[0]?.params" :collapse="6" /></div>
      </div>

      <div v-if="d.reporting.length" class="card">
        <div class="card-h"><h2>{{ t("生成报告", "Writing reports") }}</h2></div>
        <div class="card-b">
          <div v-for="b in d.reporting" :key="b.name" class="row"><router-link :to="`/batches/${b.name}`">{{ b.name }}</router-link><span class="mono small dim">{{ b.report.step }}</span></div>
        </div>
      </div>

      <div v-if="d.other_training.length" class="card">
        <div class="card-h"><h2>{{ t("其他训练进程", "Other training processes") }}</h2><span class="faint small">{{ t("不属于新面板的批次（例如手动启动或旧脚本）", "not in a batch of this panel (started by hand or by an old script)") }}</span></div>
        <div class="card-b table-wrap">
          <table class="tbl">
            <thead><tr><Th :s="trainSort" k="pid">PID</Th><Th :s="trainSort" k="scen">{{ t("场景", "Scen.") }}</Th><Th :s="trainSort" k="dir">{{ t("目录", "Dir") }}</Th><Th :s="trainSort" k="id">exp_id</Th><Th :s="trainSort" k="steps">{{ t("步数", "Steps") }}</Th><Th :s="trainSort" k="sps" num>{{ t("步/秒", "Steps/s") }}</Th><Th :s="trainSort" k="dev">{{ t("设备", "Device") }}</Th><Th :s="trainSort" k="el">{{ t("已运行", "Elapsed") }}</Th></tr></thead>
            <tbody>
              <tr v-for="p in trainSort.sorted.value" :key="p.pid">
                <td class="mono">{{ p.pid }}</td><td>{{ p.scenario?.toUpperCase() }}</td><td class="mono">{{ p.algo }}</td><td class="mono">{{ p.exp_id }}</td>
                <td class="mono small">{{ p.live ? `${steps(p.live.step)} / ${steps(p.live.total)}` : "—" }}</td><td class="num">{{ p.live?.steps_per_sec ?? "—" }}</td>
                <td>{{ p.device }}</td><td class="small">{{ dur(p.elapsed_seconds) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="section-title"><h2>{{ t("机器", "Machine") }}</h2><span class="faint small">{{ sys.hostname }} · {{ sys.cpu_model }}</span></div>
      <div class="kpis">
        <div class="kpi"><div class="kpi-l">{{ t("CPU 负载 1 / 5 / 15 分钟", "CPU load 1 / 5 / 15 min") }}</div><div class="kpi-v" style="font-size: 18px">{{ sys.load_avg?.["1m"]?.toFixed(1) }} / {{ sys.load_avg?.["5m"]?.toFixed(1) }} / {{ sys.load_avg?.["15m"]?.toFixed(1) }}</div><div class="kpi-s">{{ sys.cpu_count }} {{ t("线程", "threads") }}</div></div>
        <div class="kpi"><div class="kpi-l">{{ t("内存可用", "Memory available") }}</div><div class="kpi-v" style="font-size: 18px">{{ bytes(sys.memory?.available) }}</div><div class="kpi-s">/ {{ bytes(sys.memory?.total) }}</div></div>
        <div class="kpi"><div class="kpi-l">{{ t("训练进程", "Training processes") }}</div><div class="kpi-v">{{ (sys.processes || []).filter((p) => p.kind === "train").length }}</div><div class="kpi-s">MPS {{ (sys.processes || []).some((p) => p.mps) ? t("开", "on") : t("关", "off") }}</div></div>
      </div>
      <div class="gpu-grid">
        <GpuCard v-for="g in sys.gpus" :key="g.index" :gpu="g" :samples="gpuSamples(g.index)" />
      </div>
      <div class="card">
        <div class="card-h"><h2>{{ t("其他项目进程", "Other project processes") }}</h2></div>
        <div class="card-b table-wrap">
          <table class="tbl">
            <thead><tr><Th :s="otherSort" k="label">{{ t("进程", "Process") }}</Th><Th :s="otherSort" k="pid" class="hide-sm">PID</Th><Th :s="otherSort" k="cpu" num>CPU</Th><Th :s="otherSort" k="el">{{ t("已运行", "Elapsed") }}</Th><Th :s="otherSort" k="cmd" class="hide-sm">{{ t("命令", "Command") }}</Th></tr></thead>
            <tbody>
              <tr v-for="p in otherSort.sorted.value" :key="p.pid">
                <td><b>{{ p.label }}</b><span v-if="p.kind === 'driver'" class="tag accent" style="margin-left: 6px">{{ t("驱动", "driver") }}</span><span v-if="p.frozen" class="badge warn" style="margin-left: 6px">{{ t("已冻结", "frozen") }}</span></td>
                <td class="mono hide-sm">{{ p.pid }}</td><td class="num">{{ p.cpu_percent?.toFixed(0) }}%</td><td class="small">{{ dur(p.elapsed_seconds) }}</td>
                <td class="mono faint small hide-sm" style="max-width: 560px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap" :title="p.cmd">{{ p.cmd }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </template>

  </div>
</template>

<style scoped>
.gpu-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(380px, 100%), 1fr)); gap: 16px; }
</style>
