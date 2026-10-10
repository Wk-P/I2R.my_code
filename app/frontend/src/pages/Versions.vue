<script setup>
// Experiment versions: one row per version with a report (paper_contents/v*/), the test gap of its new
// variant per scenario, the three-scenario mean and the change against its baseline; versions still
// training / building their report appear on top. Older git tags (version/VERSION.md) below.
import { ref, computed, onMounted, onUnmounted } from "vue";
import { getReports, getBatches, getTags } from "../api.js";
import { reporting, reportText, reportPct } from "../labels.js";
import { langRef as curLang } from "../i18n.js";

const reports = ref([]);
const batches = ref([]);
const tags = ref([]);
const loading = ref(true);
const q = ref("");
let timer = null;
async function load() {
  const [r, b] = await Promise.all([getReports(), getBatches(true)]);
  reports.value = r.reports;
  batches.value = b.batches;
  loading.value = false;
}
onMounted(async () => { await load(); tags.value = await getTags(); timer = setInterval(load, 20000); });
onUnmounted(() => clearInterval(timer));

const en = computed(() => curLang.value === "en");
const L = (zh, e) => (en.value ? e : zh);
const vnorm = (v) => (v?.startsWith("v") ? v : "v" + v);
// batches of each version that are still training or building their report
const live = computed(() => {
  const m = {};
  for (const b of batches.value) if (b.version && (b.status === "running" || reporting(b))) (m[vnorm(b.version)] ||= []).push(b);
  return m;
});
const best = computed(() => Math.min(...reports.value.filter((r) => r.new_avg != null).map((r) => r.new_avg)));
const rows = computed(() => {
  const have = new Set(reports.value.map((r) => r.version));
  const pending = Object.keys(live.value).filter((v) => !have.has(v)).map((v) => ({ version: v, pending: true }));
  const s = q.value.trim().toLowerCase();
  return [...pending, ...reports.value].filter((r) => !s || (r.version + " " + (r.change ?? "") + " " + (r.en?.change ?? "")).toLowerCase().includes(s));
});
const change = (r) => (en.value ? r.en?.change ?? null : r.change);
const model = (r) => (en.value ? r.en?.new_model ?? null : r.new_model);
const gap = (r, sc) => r.gaps?.[r.new_model]?.[sc];
const delta = (r) => (r.new_avg != null && r.base_avg != null ? r.new_avg - r.base_avg : null);
const dclass = (d) => (d == null ? "" : d <= -0.5 ? "delta-good" : d >= 0.5 ? "delta-bad" : "delta-flat");
const f1 = (v) => (v == null ? "—" : v.toFixed(1) + "%");
const liveText = (v) => {
  const b = live.value[v]?.[0];
  if (!b) return null;
  return reporting(b) ? { pct: reportPct(b.report), text: reportText(b.report, curLang.value) }
    : { pct: b.overall_pct, text: L(`训练中 ${b.done}/${b.total_runs}`, `training ${b.done}/${b.total_runs}`), training: true };
};
const go = (h) => (window.location.hash = h);
</script>

<template>
  <div class="page-head">
    <h1>实验版本</h1>
    <div class="page-sub">每个版本一行：改了什么、新变体在三个场景的测试集相对 gap 和相对基线的变化；点击查看完整报告。目标：三场景 gap 都降到 5% 以下。</div>
  </div>

  <section class="panel">
    <div class="panel-head">
      <input v-model="q" class="search" placeholder="搜索版本号或改动…" />
      <span class="target-chip" data-no-i18n>{{ L("相对 gap = (ILP AR − AR) / ILP AR；变化 = 新变体三场景平均 − 基线三场景平均", "relative gap = (ILP AR − AR) / ILP AR; change = new-variant mean − baseline mean") }}</span>
    </div>
    <div v-if="loading" class="empty">加载中…</div>
    <div v-else class="table-scroll">
      <table class="grid">
        <thead><tr><th>版本</th><th>改动</th><th>状态</th><th class="num">LT</th><th class="num">EQ</th><th class="num">GT</th>
          <th class="num">三场景平均</th><th class="num">相对基线</th></tr></thead>
        <tbody>
          <tr v-for="r in rows" :key="r.version" class="clickable" @click="go(`#/versions/${r.version}`)">
            <td class="mono nowrap"><a :href="`#/versions/${r.version}`">{{ r.version }}</a></td>
            <td class="change-cell" data-no-i18n>
              <template v-if="r.pending">{{ L("报告尚未生成", "report not written yet") }}</template>
              <template v-else>
                <div v-if="model(r)"><b>{{ model(r) }}</b></div>
                <div>{{ change(r) ?? L("（见报告）", "(Chinese report only)") }}</div>
              </template>
            </td>
            <td class="nowrap" data-no-i18n>
              <template v-if="liveText(r.version)">
                <div class="bar"><div class="bar-fill" :class="{ 'bar-fill--report': !liveText(r.version).training }" :style="{ width: liveText(r.version).pct + '%' }"></div></div>
                <div class="small dim">{{ liveText(r.version).text }}</div>
              </template>
              <span v-else class="badge badge--ok">{{ L("已完成", "done") }}</span>
            </td>
            <td class="num gap-cell">{{ f1(gap(r, "LT")) }}</td>
            <td class="num gap-cell">{{ f1(gap(r, "EQ")) }}</td>
            <td class="num gap-cell">{{ f1(gap(r, "GT")) }}</td>
            <td class="num gap-cell"><b>{{ f1(r.new_avg) }}</b>
              <span v-if="r.new_avg != null && r.new_avg === best" class="badge badge--ok" style="margin-left:6px">{{ L("最好", "best") }}</span></td>
            <td class="num gap-cell" :class="dclass(delta(r))" data-no-i18n>{{ delta(r) == null ? "—" : (delta(r) > 0 ? "+" : "") + delta(r).toFixed(1) + "pp" }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>

  <section class="panel">
    <details>
      <summary class="panel-head" style="cursor:pointer"><h2 style="display:inline">全部 git 版本</h2>
        <span class="dim small" style="margin-left:8px">{{ tags.length }} 个 tag，摘要来自 version/VERSION.md</span></summary>
      <div class="table-scroll">
        <table class="grid">
          <thead><tr><th>版本</th><th>日期</th><th>摘要</th><th>文档</th></tr></thead>
          <tbody>
            <tr v-for="t in tags" :key="t.tag" class="clickable" @click="go(`#/versions/${t.tag}`)">
              <td class="mono nowrap"><a :href="`#/versions/${t.tag}`">{{ t.tag }}</a></td>
              <td class="nowrap">{{ t.date }}</td>
              <td class="summary">{{ t.summary || "—" }}</td>
              <td><span v-if="t.has_doc" class="badge badge--ok">有</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </details>
  </section>
</template>
