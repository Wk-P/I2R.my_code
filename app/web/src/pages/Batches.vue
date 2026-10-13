<script setup>
import { ref, computed, onMounted, watch } from "vue";
import { useRouter, useRoute } from "vue-router";
import { t, tl } from "../i18n.js";
import { getBatches, getMeta } from "../api.js";
import { dur, when } from "../format.js";
import { purposeName } from "../labels.js";
import Status from "../components/Status.vue";
import Progress from "../components/Progress.vue";

const router = useRouter();
const route = useRoute();
const meta = ref({ stages: [] });
const stage = ref(route.query.stage || "current");
const tests = ref(route.query.tests === "1");
const q = ref("");
const list = ref(null);
const load = async () => { list.value = null; list.value = (await getBatches(stage.value, tests.value)).batches; };
onMounted(async () => { meta.value = await getMeta(); load(); });
watch([stage, tests], () => { router.replace({ query: { stage: stage.value, tests: tests.value ? "1" : undefined } }); load(); });

const shown = computed(() => {
  const s = q.value.trim().toLowerCase();
  return (list.value || []).filter((b) => !s || b.name.toLowerCase().includes(s) || tl(b.title).toLowerCase().includes(s));
});
// group by version family (v5.0.x, v4.5.x, ...)
const groups = computed(() => {
  const out = new Map();
  for (const b of shown.value) {
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
      <select v-model="stage">
        <option value="current">{{ t("当前阶段", "Current stage") }}</option>
        <option v-for="s in meta.stages.filter((x) => !x.current)" :key="s.stage" :value="String(s.stage)">{{ s.label }}</option>
        <option value="all">{{ t("全部阶段", "All stages") }}</option>
      </select>
      <label class="check"><input v-model="tests" type="checkbox" />{{ t("显示 smoke / 测试批次", "Show smoke / test batches") }}</label>
      <span class="spacer" />
      <input v-model="q" type="search" :placeholder="t('搜索批次名或内容', 'Search name or title')" style="width: 240px" />
    </div>

    <div v-if="!list" class="empty">{{ t("加载中…", "Loading…") }}</div>
    <div v-else-if="!shown.length" class="card empty">{{ t("没有符合条件的批次。", "No batch matches.") }}</div>
    <div v-for="[v, bs] in groups" :key="v" class="card">
      <div class="card-h"><h2>{{ v }}</h2><span class="faint small">{{ bs.length }} {{ t("个批次", "batches") }}</span></div>
      <div class="card-b table-wrap">
        <table class="tbl">
          <thead>
            <tr>
              <th style="width: 150px">{{ t("批次", "Batch") }}</th><th>{{ t("内容", "What") }}</th><th>{{ t("状态", "Status") }}</th>
              <th style="width: 200px">{{ t("进度", "Progress") }}</th><th>{{ t("开始", "Started") }}</th><th>{{ t("用时", "Took") }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="b in bs" :key="b.name" class="link" @click="router.push(`/batches/${b.name}`)">
              <td>
                <router-link :to="`/batches/${b.name}`"><b>{{ b.name }}</b></router-link>
                <span v-if="b.purpose !== 'main'" class="tag" style="margin-left: 6px">{{ purposeName(b.purpose) }}</span>
                <span v-if="b.kind === 'eval'" class="tag" style="margin-left: 6px">{{ t("评估", "eval") }}</span>
              </td>
              <td class="dim small" style="max-width: 520px">{{ tl(b.title) || b.description || "—" }}</td>
              <td><Status :s="b.status" /></td>
              <td><div class="row" style="flex-wrap: nowrap"><div style="flex: 1"><Progress :b="b" /></div><span class="small num" style="width: 64px">{{ b.counts.done }}/{{ b.total }}</span></div></td>
              <td class="small dim">{{ when(b.started_at) }}</td>
              <td class="small dim">{{ b.elapsed != null ? dur(b.elapsed) : "—" }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>
