<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from "vue";
import { store, spaceLabel } from "../store.js";
import { getBatches } from "../api.js";
import { statusLabel, statusClass } from "../labels.js";
import { shortTime, steps } from "../format.js";

const batches = ref([]);
const loading = ref(true);
const filter = ref("all");
let timer = null;
const go = (h) => (window.location.hash = h);

async function load() {
  if (!store.viewBranch) return;
  batches.value = (await getBatches(true, store.viewBranch)).batches;
  loading.value = false;
}
onMounted(() => { timer = setInterval(load, 10000); });
watch(() => store.viewBranch, () => { loading.value = true; load(); }, { immediate: true });
onUnmounted(() => clearInterval(timer));

const FILTERS = [["all", "全部"], ["running", "运行中"], ["finished", "已完成"], ["stopped", "已中断"], ["cancelled", "已作废"]];
const count = (k) => (k === "all" ? batches.value.length : batches.value.filter((b) => b.status === k).length);
const shown = computed(() => (filter.value === "all" ? batches.value : batches.value.filter((b) => b.status === filter.value)));
// 版本 -> 批次 (batches are already newest first, so groups come out newest first)
const groups = computed(() => {
  const m = new Map();
  for (const b of shown.value) {
    const v = b.version ?? "未标注版本";
    if (!m.has(v)) m.set(v, []);
    m.get(v).push(b);
  }
  return [...m.entries()].map(([version, items]) => ({
    version, items,
    done: items.reduce((s, b) => s + b.done, 0),
    total: items.reduce((s, b) => s + b.total_runs, 0),
  }));
});
</script>

<template>
  <div class="page-head">
    <h1>批次管理</h1>
    <div class="page-sub">{{ spaceLabel(store.viewBranch) }} 的训练批次与评估批次，按版本分组；点击查看各任务的状态与汇总结果</div>
  </div>

  <div class="tabs">
    <button v-for="[k, label] in FILTERS" :key="k" class="tab" :class="{ active: filter === k }" @click="filter = k">
      {{ label }} <span class="tab-count">{{ count(k) }}</span>
    </button>
  </div>

  <section class="panel">
    <div v-if="loading" class="empty">加载中…</div>
    <div v-else-if="!shown.length" class="empty">没有符合条件的批次</div>
    <table v-else class="grid">
      <thead>
        <tr><th>批次</th><th>类型</th><th>状态</th><th>版本</th><th>分支</th><th class="num">步数</th><th class="w-progress">进度</th>
          <th class="num">完成</th><th class="num">运行</th><th class="num">排队</th><th class="num">中断</th><th>开始</th><th>最后更新</th></tr>
      </thead>
      <tbody v-for="g in groups" :key="g.version">
        <tr class="group-row">
          <td colspan="13">
            <span class="group-title">{{ g.version }}</span>
            <span class="dim small">{{ g.items.length }} 个批次 · {{ g.done }}/{{ g.total }} 个任务完成</span>
            <a class="small" :href="`#/results?branch=${encodeURIComponent(g.items[0].branch ?? '')}&version=${encodeURIComponent(g.version)}&view=pivot`" @click.stop>查看该版本结果 →</a>
          </td>
        </tr>
        <tr v-for="b in g.items" :key="b.batch_name" class="clickable" @click="go(`#/batches/${b.batch_name}`)">
          <td class="indent"><a :href="`#/batches/${b.batch_name}`">{{ b.batch_name }}</a></td>
          <td><span class="tag">{{ b.kind === "eval" ? "评估" : "训练" }}</span></td>
          <td><span class="badge" :class="statusClass(b.status)">{{ statusLabel(b.status) }}</span></td>
          <td>{{ b.version ?? "—" }}</td>
          <td class="dim">{{ b.branch ?? "—" }}</td>
          <td class="num">{{ b.kind === "eval" ? "—" : steps(b.steps) }}</td>
          <td>
            <div class="bar"><div class="bar-fill" :class="`bar-fill--${b.status}`" :style="{ width: b.overall_pct + '%' }"></div></div>
            <span class="bar-text">{{ b.overall_pct }}%</span>
          </td>
          <td class="num">{{ b.done }}/{{ b.total_runs }}</td>
          <td class="num">{{ b.running || "" }}</td>
          <td class="num">{{ b.queued || "" }}</td>
          <td class="num">{{ b.stopped || "" }}</td>
          <td>{{ shortTime(b.started_at) }}</td>
          <td>{{ shortTime(b.last_updated) }}</td>
        </tr>
      </tbody>
    </table>
  </section>
</template>
