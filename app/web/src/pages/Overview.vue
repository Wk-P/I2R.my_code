<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue";
import { useRouter } from "vue-router";
import { t, tl } from "../i18n.js";
import { getOverview } from "../api.js";
import { dur, when, ago, bytes, steps } from "../format.js";
import { purposeName } from "../labels.js";
import Status from "../components/Status.vue";
import Progress from "../components/Progress.vue";
import LiveRuns from "../components/LiveRuns.vue";
import GapTrend from "../components/GapTrend.vue";
import Th from "../components/Th.vue";
import { useSort } from "../sort.js";

const router = useRouter();
const o = ref(null);
let timer;
const load = async () => { o.value = await getOverview(); };
onMounted(() => { load(); timer = setInterval(load, 10000); });
onUnmounted(() => clearInterval(timer));

const queue = computed(() => o.value?.queue || { items: [] });
const pending = computed(() => queue.value.items.filter((i) => ["queued", "waiting", "running"].includes(i.status)));
const qStatus = (s) => ({ queued: t("排队中", "Queued"), running: t("运行中", "Running"), waiting: t("等待中", "Waiting"),
  done: t("已完成", "Done"), failed: t("失败", "Failed"), interrupted: t("已中断", "Interrupted") }[s] ?? s);
const qTone = { queued: "off", running: "run", waiting: "off", done: "ok", failed: "bad", interrupted: "warn" };
const m = computed(() => o.value?.machine || {});
const recentSort = useSort(computed(() => o.value?.recent || []), {
  name: (b) => b.name, title: (b) => tl(b.title), status: (b) => b.status, runs: (b) => b.counts.done, ended: (b) => b.updated_at });
const mem = computed(() => m.value.memory);

</script>

<template>
  <div class="page">
    <div class="head">
      <div>
        <h1>{{ t("总览", "Overview") }}</h1>
        <div class="head-sub">{{ t("正在跑什么、接下来跑什么、机器忙不忙。", "What is running, what runs next, and how busy the machine is.") }}</div>
      </div>
    </div>

    <GapTrend />
    <div v-if="!o" class="empty">{{ t("加载中…", "Loading…") }}</div>
    <template v-else>
      <div class="section-title"><h2>{{ t("正在进行", "In progress") }}</h2><span class="faint small">{{ t("每 10 秒刷新", "refreshes every 10 s") }}</span></div>
      <div v-if="!o.running.length && !o.reporting.length" class="card empty">{{ t("当前没有批次在训练。", "No batch is training right now.") }}</div>
      <div v-for="b in o.running" :key="b.name" class="card" style="cursor: pointer" @click="router.push(`/batches/${b.name}`)">
        <div class="card-h">
          <h2><router-link :to="`/batches/${b.name}`" @click.stop>{{ b.name }}</router-link></h2>
          <Status s="running" />
          <span v-if="b.purpose !== 'main'" class="tag">{{ purposeName(b.purpose) }}</span>
          <span class="spacer" />
          <span class="dim small">{{ t("已运行", "elapsed") }} {{ dur(b.elapsed) }} · {{ t("开始于", "started") }} {{ when(b.started_at) }}</span>
        </div>
        <div class="card-b">
          <div class="dim" style="margin-bottom: 10px">{{ tl(b.title) }}</div>
          <div class="row" style="gap: 16px; align-items: center">
            <div style="flex: 1; min-width: 240px"><Progress :b="b" legend /></div>
            <div class="kpi-v">{{ b.pct.toFixed(1) }}%</div>
          </div>
          <div class="row small dim" style="gap: 18px; margin-top: 10px">
            <span>{{ t("每个 run", "per run") }} <b>{{ steps(b.params?.steps) }}</b> {{ t("步", "steps") }}</span>
            <span>{{ t("合计", "throughput") }} <b>{{ b.steps_per_sec?.toLocaleString() ?? "—" }}</b> {{ t("步/秒", "steps/s") }}</span>
            <span v-if="b.eta">{{ t("预计训练结束", "training done") }} <b>{{ dur(b.eta.seconds) }}</b> {{ t("后", "from now") }}（{{ t("约", "at") }} {{ when(Date.now() / 1000 + b.eta.seconds) }}，{{ t("不含报告", "report not included") }}）</span>
          </div>
          <div style="margin-top: 12px" @click.stop>
            <LiveRuns :runs="o.running_runs.filter((r) => r.batch === b.name)" :params="b.params" :collapse="4" />
          </div>
        </div>
      </div>
      <div v-for="b in o.reporting" :key="b.name" class="card">
        <div class="card-h">
          <h2><router-link :to="`/batches/${b.name}`">{{ b.name }}</router-link></h2>
          <span class="badge run">{{ t("生成报告", "Writing report") }}</span>
          <span class="spacer" />
          <span class="dim small mono" v-if="b.report.step">{{ b.report.step }}</span>
        </div>
        <div class="card-b dim">{{ t("训练已全部结束，正在评估 checkpoint、计时并写报告。", "Training is over; evaluating checkpoints, timing and writing the report.") }}</div>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-h">
            <h2>{{ t("实验队列", "Experiment queue") }}</h2>
            <span class="badge" :class="queue.runner === 'active' ? 'ok' : 'warn'">{{ t("调度器", "runner") }} {{ queue.runner }}</span>
            <span v-if="queue.paused" class="badge warn">{{ t("已暂停", "Paused") }}</span>
          </div>
          <div class="card-b">
            <div v-if="queue.paused" class="notice" style="margin-bottom: 10px">
              {{ t("队列已暂停：排队的实验不会自动开始。恢复：", "The queue is paused; queued items will not start. Resume with:") }}
              <code>queue_runner.py unpause</code>
            </div>
            <div v-if="!pending.length" class="faint">{{ t("队列里没有待运行的实验。", "Nothing queued.") }}</div>
            <div v-else class="table-wrap"><table class="tbl">
              <thead><tr><th>{{ t("项目", "Item") }}</th><th>{{ t("状态", "Status") }}</th><th class="hide-sm">{{ t("等待", "Waits for") }}</th><th class="hide-sm">{{ t("加入", "Added") }}</th></tr></thead>
              <tbody>
                <tr v-for="i in pending" :key="i.id">
                  <td><b>{{ i.id }}</b><div class="faint mono small hide-sm" style="max-width: 360px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap" :title="i.cmd?.join?.(' ') ?? i.cmd">{{ i.cmd?.join?.(" ") ?? i.cmd }}</div></td>
                  <td><span class="badge" :class="qTone[i.status] || 'off'">{{ qStatus(i.status) }}</span></td>
                  <td class="mono small hide-sm">{{ i.wait_for || "—" }}</td>
                  <td class="small dim hide-sm">{{ i.added || "—" }}</td>
                </tr>
              </tbody>
            </table></div>
          </div>
        </div>

        <div class="card">
          <div class="card-h"><h2>{{ t("机器", "Machine") }}</h2><span class="spacer" /><router-link class="small" to="/monitor">{{ t("详细 →", "Details →") }}</router-link></div>
          <div class="card-b stack" style="gap: 12px">
            <div class="row" style="gap: 18px">
              <div><div class="kpi-l">{{ t("训练进程", "Training processes") }}</div><div class="kpi-v">{{ m.training_procs ?? "—" }}</div></div>
              <div><div class="kpi-l">{{ t("CPU 负载（1 分钟）", "CPU load (1 min)") }}</div><div class="kpi-v">{{ m.load_avg?.["1m"]?.toFixed(1) ?? "—" }}<span class="faint small"> / {{ m.cpu_count }}</span></div></div>
              <div v-if="mem"><div class="kpi-l">{{ t("内存可用", "Memory free") }}</div><div class="kpi-v">{{ bytes(mem.available) }}<span class="faint small"> / {{ bytes(mem.total) }}</span></div></div>
            </div>
            <div v-for="g in m.gpus" :key="g.index" class="row" style="gap: 10px">
              <span class="small" style="width: 54px">GPU {{ g.index }}</span>
              <div class="bar" style="flex: 1"><span class="s-running" :style="{ width: (g.util || 0) + '%' }" /></div>
              <span class="small num" style="width: 44px">{{ g.util?.toFixed(0) }}%</span>
              <span class="small faint" style="width: 150px">{{ ((g.mem_used || 0) / 1024).toFixed(1) }} / {{ ((g.mem_total || 0) / 1024).toFixed(0) }} GB · {{ g.temp?.toFixed(0) }}°C</span>
            </div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-h"><h2>{{ t("最近完成的正式批次", "Recently finished main batches") }}</h2><span class="spacer" /><router-link class="small" to="/batches">{{ t("全部批次 →", "All batches →") }}</router-link></div>
        <div class="card-b table-wrap">
          <table class="tbl">
            <thead><tr><Th :s="recentSort" k="name">{{ t("批次", "Batch") }}</Th><Th :s="recentSort" k="title" class="hide-sm">{{ t("内容", "What") }}</Th><Th :s="recentSort" k="status">{{ t("状态", "Status") }}</Th><Th :s="recentSort" k="runs" num>run</Th><Th :s="recentSort" k="ended">{{ t("结束", "Ended") }}</Th></tr></thead>
            <tbody>
              <tr v-if="!o.recent.length"><td colspan="5" class="faint">{{ t("当前还没有完成的正式批次。", "No main batch has finished yet.") }}</td></tr>
              <tr v-for="b in recentSort.sorted.value" :key="b.name" class="link" @click="router.push(`/batches/${b.name}`)">
                <td><router-link :to="`/batches/${b.name}`">{{ b.name }}</router-link></td>
                <td class="dim small hide-sm" style="max-width: 560px">{{ tl(b.title) || "—" }}</td>
                <td><Status :s="b.status" /></td>
                <td class="num">{{ b.counts.done }}/{{ b.total }}</td>
                <td class="small dim">{{ ago(b.updated_at) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </template>
  </div>
</template>
