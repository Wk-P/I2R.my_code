<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from "vue";
import { store, stages, stageOf, currentStage } from "../store.js";
import { getBatches } from "../api.js";
import { statusLabel, statusClass, reporting, reportText, reportPct } from "../labels.js";
import { langRef as curLang } from "../i18n.js";
import { shortTime, steps } from "../format.js";

const batches = ref([]);
const loading = ref(true);
const filter = ref("all");
// docs/data_spec.md: smoke / test batches are hidden unless asked for
const showTests = ref(false);
const isTest = (b) => b.purpose && b.purpose !== "main";
let timer = null;
const go = (h) => (window.location.hash = h);

// every data space at once; the stage filter (default: the current stage) replaces the old global space switch.
// ?version=<v> (from a batch page's breadcrumb) narrows to one version.
const stage = ref(null);
const onlyVersion = ref(new URLSearchParams(window.location.hash.split("?")[1] || "").get("version") || "");
async function load() {
  batches.value = (await getBatches(true)).batches;
  loading.value = false;
}
onMounted(() => { load(); timer = setInterval(load, 10000); });
onUnmounted(() => clearInterval(timer));
watch(() => store.resultBranches.length, () => { if (stage.value == null) stage.value = currentStage(); }, { immediate: true });
const inStage = (b) => stage.value === "all" || stage.value == null || stageOf(b.branch) === stage.value;

const FILTERS = [["all", "全部"], ["running", "运行中"], ["finished", "已完成"], ["stopped", "已中断"], ["cancelled", "已作废"]];
const scoped = computed(() => batches.value.filter((b) => inStage(b) && (!onlyVersion.value || b.version === onlyVersion.value)));
const visible = computed(() => scoped.value.filter((b) => showTests.value || !isTest(b)));
const nTests = computed(() => scoped.value.filter(isTest).length);
const count = (k) => (k === "all" ? visible.value.length : visible.value.filter((b) => b.status === k).length);
const shown = computed(() => (filter.value === "all" ? visible.value : visible.value.filter((b) => b.status === filter.value)));
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
    <h1>批次</h1>
    <div class="page-sub">每次启动的训练 / 评估为一个批次，按版本分组；点击批次查看结果矩阵与各 run 的状态</div>
  </div>

  <div class="filter-row">
    <label class="stage-select">阶段
      <select v-model="stage">
        <option v-for="s in stages()" :key="s.stage" :value="s.stage">{{ s.label }}{{ s.current ? "（当前）" : "" }}</option>
        <option value="all">全部阶段</option>
      </select>
    </label>
    <span v-if="onlyVersion" class="tag tag--param">版本 {{ onlyVersion }} <a href="#/batches" @click="onlyVersion = ''">×</a></span>
  </div>
  <div class="tabs">
    <button v-for="[k, label] in FILTERS" :key="k" class="tab" :class="{ active: filter === k }" @click="filter = k">
      {{ label }} <span class="tab-count">{{ count(k) }}</span>
    </button>
    <label v-if="nTests" class="tab tab--check"><input type="checkbox" v-model="showTests" /> 显示测试批次（smoke 等 {{ nTests }} 个）</label>
  </div>

  <section class="panel">
    <div v-if="loading" class="empty">加载中…</div>
    <div v-else-if="!shown.length" class="empty">没有符合条件的批次</div>
    <div class="table-scroll" v-else><table class="grid">
      <thead>
        <tr><th>批次</th><th>类型</th><th>状态</th><th>版本</th><th class="num">步数</th><th class="w-progress">进度</th>
          <th class="num">完成</th><th class="num">运行</th><th class="num">排队</th><th class="num">中断</th><th>开始</th><th>最后更新</th></tr>
      </thead>
      <tbody v-for="g in groups" :key="g.version">
        <tr class="group-row">
          <td colspan="12">
            <span class="group-title">{{ g.version }}</span>
            <span class="dim small">{{ g.items.length }} 个批次 · {{ g.done }}/{{ g.total }} 个任务完成</span>
            <a v-if="g.version !== '未标注版本'" class="small" :href="`#/versions/v${String(g.version).replace(/^v/, '')}`" @click.stop>版本报告 →</a>
            <a class="small" :href="`#/results?branch=${encodeURIComponent(g.items[0].branch ?? '')}&version=${encodeURIComponent(g.version)}&view=pivot`" @click.stop>结果对比 →</a>
          </td>
        </tr>
        <tr v-for="b in g.items" :key="b.batch_name" class="clickable" @click="go(`#/batches/${b.batch_name}`)">
          <td class="indent"><a :href="`#/batches/${b.batch_name}`">{{ b.batch_name }}</a></td>
          <td><span class="tag">{{ b.kind === "eval" ? "评估" : "训练" }}</span>
            <span v-if="isTest(b)" class="tag tag--wait">{{ b.purpose === "smoke" ? "冒烟测试" : "测试" }}</span></td>
          <td><span class="badge" :class="statusClass(b.status)">{{ statusLabel(b.status) }}</span>
            <div v-if="reporting(b)" class="small report-note" data-no-i18n>{{ reportText(b.report, curLang) }}</div></td>
          <td>{{ b.version ?? "—" }}</td>
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
    </table></div>
  </section>
</template>
