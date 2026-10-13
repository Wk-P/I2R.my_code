<script setup>
import { ref, computed, onMounted, watch } from "vue";
import { useRouter, useRoute } from "vue-router";
import { t, tl } from "../i18n.js";
import { getBatches } from "../api.js";
import { dur, when, steps } from "../format.js";
import { purposeName } from "../labels.js";
import Status from "../components/Status.vue";
import Progress from "../components/Progress.vue";
import Th from "../components/Th.vue";
import { useSort } from "../sort.js";

const router = useRouter();
const route = useRoute();
const tests = ref(route.query.tests === "1");
const q = ref("");
const list = ref(null);
const load = async () => { list.value = null; list.value = (await getBatches(tests.value)).batches; };
onMounted(load);
watch(tests, () => { router.replace({ query: { tests: tests.value ? "1" : undefined } }); load(); });

const shown = computed(() => {
  const s = q.value.trim().toLowerCase();
  return (list.value || []).filter((b) => !s || b.name.toLowerCase().includes(s) || tl(b.title).toLowerCase().includes(s));
});
// group by version family (v5.0.x, v4.5.x, ...)
const STATI = { running: 0, stopped: 1, cancelled: 2, finished: 3 };
const sort = useSort(shown, {
  name: (b) => b.name, title: (b) => tl(b.title), status: (b) => STATI[b.status], steps: (b) => b.params?.steps, pct: (b) => b.pct,
  counts: (b) => b.counts.running, started: (b) => b.started_at, took: (b) => b.elapsed,
});
const groups = computed(() => {
  const out = new Map();
  for (const b of sort.sorted.value) {
    const v = b.version ? `v${b.version.split(".").slice(0, 2).join(".")}` : t("其他", "Other");
    if (!out.has(v)) out.set(v, []);
    out.get(v).push(b);
  }
  return [...out.entries()];
});
</script>

<template>
  <div class="page">
    <div class="head">
      <div>
        <h1>{{ t("批次", "Batches") }}</h1>
        <div class="head-sub">{{ t("一个批次 = 一次驱动脚本的运行（例如 v5.0.4）。点进去看结果矩阵、图表和报告。", "A batch is one run of a driver script (e.g. v5.0.4). Open one for its result matrix, charts and report.") }}</div>
      </div>
    </div>
    <div class="row">
      <label class="check"><input v-model="tests" type="checkbox" />{{ t("显示 smoke / 测试批次", "Show smoke / test batches") }}</label>
      <span class="spacer" />
      <input v-model="q" type="search" :placeholder="t('搜索批次名或内容', 'Search name or title')" style="width: 240px" />
    </div>

    <div v-if="!list" class="empty">{{ t("加载中…", "Loading…") }}</div>
    <div v-else-if="!shown.length" class="card empty">
      {{ t("没有符合条件的批次。", "No batch matches.") }}<br />
      <span class="small">{{ t("新面板只显示按新数据规范（manifest schema 2，v5.0.3 起）写出的批次；更早的批次请到", "The new panel only shows batches written to the new data spec (manifest schema 2, v5.0.3 on); older batches are in the") }}
        <a href="/classic/#/batches">{{ t("旧版面板", "classic panel") }}</a>{{ t("。", ".") }}</span>
    </div>
    <div v-for="[v, bs] in groups" :key="v" class="card">
      <div class="card-h"><h2>{{ v }}</h2><span class="faint small">{{ bs.length }} {{ t("个批次", "batches") }}</span></div>
      <div class="card-b table-wrap">
        <table class="tbl">
          <thead>
            <tr>
              <Th :s="sort" k="name" style="width: 150px">{{ t("批次", "Batch") }}</Th><Th :s="sort" k="title" class="hide-sm">{{ t("内容", "What") }}</Th><Th :s="sort" k="status">{{ t("状态", "Status") }}</Th>
              <Th :s="sort" k="steps" class="hide-sm">{{ t("每个 run", "Per run") }}</Th><Th :s="sort" k="pct" style="width: 200px">{{ t("进度", "Progress") }}</Th><Th :s="sort" k="counts" class="hide-sm">{{ t("运行 / 排队 / 中断", "Run / queued / stopped") }}</Th><Th :s="sort" k="started" class="hide-sm">{{ t("开始", "Started") }}</Th><Th :s="sort" k="took" class="hide-sm">{{ t("用时", "Took") }}</Th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="b in bs" :key="b.name" class="link" @click="router.push(`/batches/${b.name}`)">
              <td>
                <router-link :to="`/batches/${b.name}`"><b>{{ b.name }}</b></router-link>
                <div class="show-sm faint small">{{ tl(b.title) }}</div>
                <span v-if="b.purpose !== 'main'" class="tag" style="margin-left: 6px">{{ purposeName(b.purpose) }}</span>
                              </td>
              <td class="dim small hide-sm" style="max-width: 520px">{{ tl(b.title) || "—" }}</td>
              <td><Status :s="b.status" /></td>
              <td class="small mono hide-sm">{{ steps(b.params?.steps) }}</td>
              <td><div class="row" style="flex-wrap: nowrap"><div style="flex: 1"><Progress :b="b" /></div><span class="small num" style="width: 64px">{{ b.counts.done }}/{{ b.total }}</span></div></td>
              <td class="small hide-sm">{{ b.counts.running }} / {{ b.counts.queued + b.counts.waiting }} / {{ b.counts.stopped }}</td>
              <td class="small dim hide-sm">{{ when(b.started_at) }}</td>
              <td class="small dim hide-sm">{{ b.elapsed != null ? dur(b.elapsed) : "—" }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>
