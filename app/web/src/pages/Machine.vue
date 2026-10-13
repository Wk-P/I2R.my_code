<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue";
import { t, lang } from "../i18n.js";
import { getSystem, getGpuHistory } from "../api.js";
import { dur, bytes } from "../format.js";
import Chart from "../components/Chart.vue";

const s = ref(null);
const hist = ref([]);
let timer;
async function load() {
  s.value = await getSystem();
  const since = hist.value.length ? hist.value[hist.value.length - 1].t : 0;
  const h = await getGpuHistory(since);
  hist.value = [...hist.value, ...(h?.samples || [])].filter((x) => x.t > Date.now() / 1000 - 1800);
}
onMounted(() => { load(); timer = setInterval(load, 5000); });
onUnmounted(() => clearInterval(timer));

const COLORS = ["#4f46e5", "#d97706", "#059669", "#dc2626"];
const gpuOpt = (key, name, unit) => {
  lang.value;
  const ids = [...new Set(hist.value.flatMap((x) => x.gpus.map((g) => g.i)))];
  return { grid: { left: 44, right: 14, top: 30, bottom: 28 }, legend: { top: 0, right: 0, textStyle: { fontSize: 11 } },
    tooltip: { trigger: "axis", valueFormatter: (v) => `${v}${unit}` },
    xAxis: { type: "time", axisLabel: { formatter: "{HH}:{mm}" } }, yAxis: { type: "value", name, min: 0 },
    series: ids.map((i, k) => ({ type: "line", name: `GPU ${i}`, color: COLORS[k % 4], showSymbol: false,
      data: hist.value.map((x) => [x.t * 1000, x.gpus.find((g) => g.i === i)?.[key] ?? null]) })) };
};
const procs = computed(() => (s.value?.processes || []).slice().sort((a, b) => (b.cpu_percent || 0) - (a.cpu_percent || 0)));
const mem = computed(() => s.value?.host?.memory);
</script>

<template>
  <div class="page">
    <div class="head"><div><h1>{{ t("机器", "Machine") }}</h1><div class="head-sub">{{ s?.host?.hostname }} · {{ s?.host?.cpu?.model }}</div></div></div>
    <div v-if="!s" class="empty">{{ t("加载中…", "Loading…") }}</div>
    <template v-else>
      <div class="kpis">
        <div class="kpi"><div class="kpi-l">{{ t("CPU 负载 1 / 5 / 15 分钟", "CPU load 1 / 5 / 15 min") }}</div><div class="kpi-v" style="font-size: 18px">{{ s.load_avg["1m"]?.toFixed(1) }} / {{ s.load_avg["5m"]?.toFixed(1) }} / {{ s.load_avg["15m"]?.toFixed(1) }}</div><div class="kpi-s">{{ s.cpu_count }} {{ t("线程", "threads") }}</div></div>
        <div v-if="mem" class="kpi"><div class="kpi-l">{{ t("内存可用", "Memory available") }}</div><div class="kpi-v" style="font-size: 18px">{{ bytes(mem.available) }}</div><div class="kpi-s">/ {{ bytes(mem.total) }}</div></div>
        <div class="kpi"><div class="kpi-l">{{ t("项目进程", "Project processes") }}</div><div class="kpi-v">{{ procs.length }}</div><div class="kpi-s">{{ procs.filter((p) => p.scenario).length }} {{ t("个训练", "training") }}</div></div>
        <div v-for="g in s.gpus" :key="g.index" class="kpi">
          <div class="kpi-l">GPU {{ g.index }} · {{ g.name?.replace("NVIDIA GeForce ", "") }}</div>
          <div class="kpi-v">{{ g.util?.toFixed(0) }}%</div>
          <div class="kpi-s">{{ ((g.mem_used || 0) / 1024).toFixed(1) }}/{{ ((g.mem_total || 0) / 1024).toFixed(0) }} GB · {{ g.temp?.toFixed(0) }}°C · {{ g.power?.toFixed(0) }} W · {{ g.procs }} {{ t("进程", "procs") }}</div>
        </div>
      </div>
      <div class="grid-2">
        <div class="card"><div class="card-h"><h2>{{ t("GPU 利用率（近 30 分钟）", "GPU utilisation (last 30 min)") }}</h2></div><div class="card-b"><Chart :option="gpuOpt('util', '%', '%')" :height="220" /></div></div>
        <div class="card"><div class="card-h"><h2>{{ t("GPU 温度", "GPU temperature") }}</h2></div><div class="card-b"><Chart :option="gpuOpt('temp', '°C', '°C')" :height="220" /></div></div>
      </div>
      <div class="card">
        <div class="card-h"><h2>{{ t("项目进程", "Project processes") }}</h2><span class="faint small">{{ t("每 5 秒刷新", "refreshes every 5 s") }}</span></div>
        <div class="card-b table-wrap">
          <table class="tbl">
            <thead><tr><th>{{ t("进程", "Process") }}</th><th>PID</th><th class="num">CPU</th><th>GPU</th><th>{{ t("运行时长", "Running for") }}</th><th>{{ t("命令", "Command") }}</th></tr></thead>
            <tbody>
              <tr v-for="p in procs" :key="p.pid">
                <td><b>{{ p.label }}</b><span v-if="p.frozen" class="badge warn" style="margin-left: 6px">{{ t("已冻结", "frozen") }}</span></td>
                <td class="mono">{{ p.pid }}</td><td class="num">{{ p.cpu_percent?.toFixed(0) }}%</td><td>{{ p.gpu ?? "" }}</td>
                <td class="small">{{ dur(p.elapsed_seconds) }}</td>
                <td class="mono faint small" style="max-width: 520px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap" :title="p.cmd">{{ p.cmd }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </template>
  </div>
</template>
