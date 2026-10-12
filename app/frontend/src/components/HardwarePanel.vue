<script setup>
// Machine hardware (read-only): CPU, memory, disk and every GPU, from /api/system (refreshed with it).
import { computed, onMounted, onUnmounted, reactive } from "vue";
import GpuClockChart from "./GpuClockChart.vue";
import { getGpuHistory } from "../api.js";
import { store } from "../store.js";
import { langRef as curLang } from "../i18n.js";

const T = (zh, en) => (curLang.value === "en" ? en : zh);

// live GPU clock history: the whole buffer once, then only new samples every 2 s
const hist = reactive({ window: 1800, samples: [] });
let histTimer = null;
async function pollHistory() {
  const since = hist.samples.length ? hist.samples[hist.samples.length - 1].t : 0;
  const r = await getGpuHistory(since);
  if (!r?.samples?.length) return;
  hist.window = r.window || hist.window;
  const cut = r.samples[r.samples.length - 1].t - hist.window;
  hist.samples = hist.samples.concat(r.samples).filter((s) => s.t >= cut);
}
onMounted(() => { pollHistory(); histTimer = setInterval(pollHistory, 2000); });
onUnmounted(() => clearInterval(histTimer));
const series = (i) => hist.samples.map((s) => ({ t: s.t, ...(s.gpus.find((g) => g.i === i) ?? {}) }));
const host = computed(() => store.system?.host || null);
const gpus = computed(() => store.system?.gpus || []);
const load = computed(() => store.system?.load_avg || {});
const procs = computed(() => store.system?.processes || []);
const trainingOn = (i, frozen) => procs.value.filter((p) => p.scenario && p.algo && p.gpu === i && !!p.frozen === frozen).length;

const GB = (b) => (b == null ? "—" : (b / 1024 ** 3).toFixed(b >= 100 * 1024 ** 3 ? 0 : 1) + " GB");
const TB = (b) => (b == null ? "—" : (b / 1024 ** 4).toFixed(2) + " TB");
const n0 = (v, u = "") => (v == null ? "—" : Math.round(v) + u);
const ratio = (a, b) => (a == null || !b ? 0 : Math.min(100, (100 * a) / b));
const uptime = (s) => {
  if (s == null) return "—";
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600);
  return curLang.value === "en" ? `${d} d ${h} h` : `${d} 天 ${h} 小时`;
};
// usage / temperature colouring: green < warn threshold <= amber < bad threshold <= red
const level = (v, warn, bad) => (v == null ? "" : v >= bad ? "hw-bad" : v >= warn ? "hw-warn" : "hw-ok");
const coreColor = (u) => {
  if (u == null) return "var(--off-soft)";
  const a = 0.12 + 0.88 * Math.min(1, u / 100);
  return `rgba(37, 99, 235, ${a.toFixed(2)})`;
};

const mem = computed(() => {
  const m = host.value?.memory;
  if (!m?.total) return null;
  const used = m.total - (m.available ?? 0);
  const swapUsed = m.swap_total != null && m.swap_free != null ? m.swap_total - m.swap_free : null;
  return { ...m, used, swapUsed };
});
</script>

<template>
  <section class="panel hw" data-no-i18n>
    <div class="panel-head">
      <h2>{{ T("硬件", "Hardware") }}</h2>
      <span v-if="host" class="dim small">{{ host.hostname }} · Linux {{ host.kernel }} · {{ T("已开机", "up") }} {{ uptime(host.uptime_seconds) }}
        <template v-if="host.nvidia_driver"> · NVIDIA {{ host.nvidia_driver }} · CUDA {{ host.cuda ?? "—" }}</template></span>
    </div>
    <div v-if="!host" class="empty">{{ T("硬件信息加载中…", "Loading hardware information…") }}</div>
    <template v-else>
      <div class="hw-grid">
        <!-- CPU -->
        <div class="hw-card">
          <div class="hw-title">CPU</div>
          <div class="hw-sub">{{ host.cpu.model ?? "—" }}</div>
          <div class="hw-sub">{{ host.cpu.sockets ?? "—" }} {{ T("路", "sockets") }} · {{ host.cpu.cores ?? "—" }} {{ T("核", "cores") }} · {{ host.cpu.threads ?? "—" }} {{ T("线程", "threads") }}</div>
          <div class="hw-row"><span>{{ T("使用率", "Usage") }}</span>
            <span class="bar hw-bar"><span class="bar-fill" :class="level(host.cpu.usage, 80, 95)" :style="{ width: (host.cpu.usage ?? 0) + '%' }"></span></span>
            <b>{{ host.cpu.usage == null ? "—" : host.cpu.usage.toFixed(0) + "%" }}</b></div>
          <div class="hw-row"><span>{{ T("负载 1/5/15 分钟", "Load 1/5/15 min") }}</span>
            <b>{{ ["1m", "5m", "15m"].map((k) => load[k]?.toFixed(1) ?? "—").join(" / ") }}</b></div>
          <div class="hw-row"><span>{{ T("平均频率", "Avg clock") }}</span><b>{{ n0(host.cpu.mhz_avg, " MHz") }} / {{ n0(host.cpu.mhz_max, " MHz") }}</b></div>
          <div v-for="t in host.cpu.temps" :key="t.name" class="hw-row">
            <span>{{ T("温度", "Temp") }} · {{ t.name.replace("Package id ", T("CPU ", "CPU ")) }}</span>
            <b :class="level(t.temp, (t.crit ?? 100) - 15, (t.crit ?? 100) - 5)">{{ n0(t.temp, " °C") }}</b>
            <span class="dim small">{{ T("临界", "crit") }} {{ n0(t.crit, " °C") }}</span></div>
          <div v-if="host.cpu.per_core?.length" class="hw-cores-wrap">
            <div class="dim small">{{ T("逐线程使用率（颜色越深越忙）", "Per-thread usage (darker = busier)") }}</div>
            <div class="hw-cores">
              <span v-for="(u, i) in host.cpu.per_core" :key="i" class="hw-core" :style="{ background: coreColor(u) }"
                :title="`cpu${i}: ${u == null ? '—' : u.toFixed(0) + '%'}`"></span>
            </div>
          </div>
        </div>

        <!-- memory + disk -->
        <div class="hw-card">
          <div class="hw-title">{{ T("内存", "Memory") }}</div>
          <template v-if="mem">
            <div class="hw-row"><span>{{ T("已用", "Used") }}</span>
              <span class="bar hw-bar"><span class="bar-fill" :class="level(ratio(mem.used, mem.total), 80, 92)" :style="{ width: ratio(mem.used, mem.total) + '%' }"></span></span>
              <b>{{ GB(mem.used) }} / {{ GB(mem.total) }}</b></div>
            <div class="hw-row"><span>{{ T("可用", "Available") }}</span><b>{{ GB(mem.available) }}</b></div>
            <div class="hw-row"><span>{{ T("缓存（可回收）", "Cache (reclaimable)") }}</span><b>{{ GB(mem.cached) }}</b></div>
            <div class="hw-row"><span>Swap</span>
              <span class="bar hw-bar"><span class="bar-fill" :class="level(ratio(mem.swapUsed, mem.swap_total), 50, 80)" :style="{ width: ratio(mem.swapUsed, mem.swap_total) + '%' }"></span></span>
              <b>{{ GB(mem.swapUsed) }} / {{ GB(mem.swap_total) }}</b></div>
          </template>
          <div class="hw-title hw-title--gap">{{ T("磁盘（项目所在分区）", "Disk (project partition)") }}</div>
          <template v-if="host.disk">
            <div class="hw-row"><span>{{ T("已用", "Used") }}</span>
              <span class="bar hw-bar"><span class="bar-fill" :class="level(ratio(host.disk.used, host.disk.total), 80, 92)" :style="{ width: ratio(host.disk.used, host.disk.total) + '%' }"></span></span>
              <b>{{ TB(host.disk.used) }} / {{ TB(host.disk.total) }}</b></div>
            <div class="hw-row"><span>{{ T("剩余", "Free") }}</span><b>{{ TB(host.disk.free) }}</b></div>
            <div class="hw-sub mono small">{{ host.disk.path }}</div>
          </template>
        </div>

        <!-- GPUs -->
        <div v-for="g in gpus" :key="g.index" class="hw-card">
          <div class="hw-title">GPU {{ g.index }} <span class="dim small">{{ g.name }}</span></div>
          <div class="hw-sub">{{ g.pstate }} · PCIe {{ g.pcie }} · {{ T("训练任务", "training jobs") }} {{ trainingOn(g.index, false) }}<template v-if="trainingOn(g.index, true)"> · {{ T("冻结", "frozen") }} {{ trainingOn(g.index, true) }}</template>
            · {{ T("计算进程", "compute procs") }} {{ g.procs }}</div>
          <div class="hw-row"><span>{{ T("利用率", "Utilisation") }}</span>
            <span class="bar hw-bar"><span class="bar-fill" :style="{ width: (g.util ?? 0) + '%' }"></span></span>
            <b>{{ n0(g.util, "%") }}</b></div>
          <div class="hw-row"><span>{{ T("显存", "Memory") }}</span>
            <span class="bar hw-bar"><span class="bar-fill" :class="level(ratio(g.mem_used, g.mem_total), 80, 92)" :style="{ width: ratio(g.mem_used, g.mem_total) + '%' }"></span></span>
            <b>{{ g.mem_used == null ? "—" : (g.mem_used / 1024).toFixed(1) }} / {{ g.mem_total == null ? "—" : (g.mem_total / 1024).toFixed(0) }} GB</b></div>
          <div class="hw-row"><span>{{ T("显存带宽占用", "Memory bandwidth") }}</span><b>{{ n0(g.mem_util, "%") }}</b></div>
          <div class="hw-row"><span>{{ T("功耗", "Power") }}</span>
            <span class="bar hw-bar"><span class="bar-fill" :style="{ width: ratio(g.power, g.power_limit) + '%' }"></span></span>
            <b>{{ n0(g.power, " W") }} / {{ n0(g.power_limit, " W") }}</b></div>
          <div class="hw-row"><span>{{ T("温度", "Temp") }}</span>
            <b :class="level(g.temp, g.temp_target ?? 84, g.temp_slowdown ?? 93)">{{ n0(g.temp, " °C") }}</b>
            <span class="dim small">{{ T("目标", "target") }} {{ n0(g.temp_target, "°C") }} · {{ T("硬降频", "HW slowdown") }} {{ n0(g.temp_slowdown, "°C") }} · {{ T("关机", "shutdown") }} {{ n0(g.temp_shutdown, "°C") }}</span></div>
          <div class="hw-row"><span>{{ T("风扇", "Fan") }}</span><b>{{ n0(g.fan, "%") }}</b></div>
          <div class="hw-row"><span>{{ T("核心频率", "SM clock") }}</span>
            <b :class="g.throttle?.length ? 'hw-warn' : ''" :title="g.throttle?.length ? '' : T('无降频；低负载时 GPU Boost 不会升到最高频率', 'not throttled; GPU Boost does not reach the max clock at low load')">{{ n0(g.clock_sm) }} / {{ n0(g.clock_sm_max, " MHz") }}</b>
            <span class="dim small">{{ T("显存", "mem") }} {{ n0(g.clock_mem, " MHz") }}</span></div>
          <div class="hw-row"><span>{{ T("降频原因", "Throttling") }}</span>
            <span v-if="!g.throttle?.length" class="badge badge--ok">{{ T("无", "none") }}</span>
            <span v-for="r in g.throttle" :key="r.en" class="badge badge--warn">{{ T(r.zh, r.en) }}</span></div>
          <GpuClockChart :samples="series(g.index)" :window="hist.window" />
        </div>
      </div>
    </template>
  </section>
</template>

<style scoped>
.hw-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }
.hw-card { border: 1px solid var(--border); border-radius: var(--radius); padding: 12px 14px; min-width: 0; }
.hw-title { font-weight: 650; margin-bottom: 2px; }
.hw-title--gap { margin-top: 14px; }
.hw-sub { font-size: 12px; color: var(--text-dim); margin-bottom: 6px; overflow-wrap: anywhere; }
.hw-row { display: flex; align-items: center; gap: 8px; font-size: 13px; padding: 3px 0; flex-wrap: wrap; }
.hw-row > span:first-child { color: var(--text-dim); min-width: 110px; }
.hw-bar { flex: 1; min-width: 80px; max-width: 160px; }
.hw-bar > .bar-fill { display: block; }
.bar-fill.hw-ok { background: var(--ok); }
.bar-fill.hw-warn { background: #e0a65a; }
.bar-fill.hw-bad { background: var(--bad); }
b.hw-warn { color: var(--warn); }
b.hw-bad { color: var(--bad); }
.hw-cores-wrap { margin-top: 8px; }
.hw-cores { display: grid; grid-template-columns: repeat(auto-fill, minmax(14px, 1fr)); gap: 3px; margin-top: 4px; }
.hw-core { height: 14px; border-radius: 3px; }
@media (max-width: 640px) { .hw-grid { grid-template-columns: 1fr; } .hw-row > span:first-child { min-width: 90px; } }
</style>
