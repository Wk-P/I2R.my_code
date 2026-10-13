<script setup>
// One experiment version: headline cards (test gap of the new variant vs the baseline, per scenario and
// mean), the report rendered in the current language (any other markdown file of the version on a
// click), the batches of the version, and -- for versions without a report -- the git tag's doc.
import { ref, computed, watch } from "vue";
import { getReport, getBatches, getTags, getTagDoc } from "../api.js";
import { langRef } from "../i18n.js";
import { renderMarkdown } from "../markdown.js";
import { statusLabel, statusClass, reporting, reportText, reportPct } from "../labels.js";
import { shortTime } from "../format.js";

const props = defineProps({ tag: { type: String, required: true } });
const rep = ref(null);
const file = ref(null);
const batches = ref([]);
const tagMeta = ref(null);
const tagDoc = ref(null);
const loading = ref(true);
const en = computed(() => langRef.value === "en");
const L = (zh, e) => (en.value ? e : zh);

watch(() => props.tag, () => { file.value = null; }, { immediate: true });
watch([() => props.tag, langRef, file], async ([tag, lang, f]) => {
  loading.value = true;
  rep.value = await getReport(tag, f, lang);
  const all = (await getBatches(true)).batches;
  batches.value = all.filter((b) => b.version && ("v" + b.version.replace(/^v/, "")) === tag);
  if (!rep.value) {
    tagMeta.value = (await getTags()).find((t) => t.tag === tag) ?? null;
    tagDoc.value = tagMeta.value?.has_doc ? await getTagDoc(tag, lang) : null;
  }
  loading.value = false;
}, { immediate: true });

const parsed = computed(() => rep.value?.parsed ?? null);
const cards = computed(() => {
  const p = parsed.value;
  if (!p?.new_model) return [];
  const nv = p.gaps[p.new_model], bv = p.base_model !== p.new_model ? p.gaps[p.base_model] : null;
  const c = ["LT", "EQ", "GT"].map((s) => ({ k: s, v: nv[s], b: bv?.[s] }));
  c.push({ k: "avg", v: p.new_avg, b: p.base_avg });
  return c;
});
const html = computed(() => renderMarkdown(rep.value?.content ?? ""));
const title = computed(() => (rep.value?.content.match(/^# (.*)$/m) || [])[1] ?? props.tag);
const fname = (f) => ({ "report.md": L("主报告", "main report"), "seeds_report.md": L("多种子报告", "multi-seed report"),
  "pilot_report.md": L("试跑报告", "pilot report") }[f] ?? f.replace(/\.md$/, ""));
const f1 = (v) => (v == null ? "—" : v.toFixed(1) + "%");
const go = (h) => (window.location.hash = h);
const dclass = (d) => (d == null ? "" : d <= -0.5 ? "delta-good" : d >= 0.5 ? "delta-bad" : "delta-flat");
</script>

<template>
  <div class="crumbs"><a href="#/versions">版本报告</a> / {{ tag }}</div>
  <div v-if="loading && !rep" class="empty">加载中…</div>

  <template v-else-if="rep">
    <div class="page-head">
      <h1 data-no-i18n>{{ title }}</h1>
      <div class="page-sub" data-no-i18n>{{ L("新变体", "New variant") }}：<b>{{ parsed?.new_model ?? "—" }}</b>
        · {{ L("基线", "baseline") }}：{{ parsed?.base_model ?? "—" }}</div>
    </div>

    <div v-if="cards.length" class="kpis" data-no-i18n>
      <div v-for="c in cards" :key="c.k" class="kpi">
        <div class="kpi-label">{{ c.k === "avg" ? L("三场景平均 · 测试集相对 gap", "3-scenario mean · test relative gap") : c.k + L(" · 测试集相对 gap", " · test relative gap") }}</div>
        <div class="kpi-value">{{ f1(c.v) }}</div>
        <div class="kpi-foot">
          <template v-if="c.b != null">{{ L("基线", "baseline") }} {{ f1(c.b) }} ·
            <span :class="dclass(c.v - c.b)">{{ (c.v - c.b > 0 ? "+" : "") + (c.v - c.b).toFixed(1) }}pp</span></template>
          <span v-if="c.v != null"> · {{ L("距目标 5%", "to the 5% target") }} {{ Math.max(0, c.v - 5).toFixed(1) }}pp</span>
        </div>
      </div>
    </div>

    <section v-if="batches.length" class="panel">
      <div class="panel-head"><h2>该版本的批次</h2></div>
      <div class="table-scroll">
        <table class="grid">
          <thead><tr><th>批次</th><th>状态</th><th class="w-progress">进度</th><th>任务</th><th>开始时间</th></tr></thead>
          <tbody>
            <tr v-for="b in batches" :key="b.batch_name" class="clickable" @click="go(`#/batches/${b.batch_name}`)">
              <td><a :href="`#/batches/${b.batch_name}`">{{ b.batch_name }}</a></td>
              <td><span class="badge" :class="statusClass(b.status)">{{ statusLabel(b.status) }}</span></td>
              <td data-no-i18n>
                <template v-if="reporting(b)">
                  <div class="bar"><div class="bar-fill bar-fill--report" :style="{ width: reportPct(b.report) + '%' }"></div></div>
                  <span class="bar-text">{{ reportText(b.report, langRef) }}</span></template>
                <template v-else><div class="bar"><div class="bar-fill" :style="{ width: b.overall_pct + '%' }"></div></div>
                  <span class="bar-text">{{ b.overall_pct }}%</span></template>
              </td>
              <td>{{ b.done }}/{{ b.total_runs }}</td>
              <td>{{ shortTime(b.started_at) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <div class="file-tabs" data-no-i18n>
          <button v-for="f in rep.files" :key="f" class="chip-btn" :class="{ on: rep.file === f }" @click="file = f">{{ fname(f) }}</button>
        </div>
        <span v-if="rep.lang_fallback" class="chip chip--warn-bg" data-no-i18n>{{ L("", "This report exists in Chinese only") }}</span>
      </div>
      <article class="prose-md" data-no-i18n v-html="html"></article>
    </section>
  </template>

  <template v-else>
    <div v-if="!tagMeta" class="empty">找不到版本 {{ tag }}</div>
    <template v-else>
      <div class="page-head"><h1>{{ tag }}</h1><div class="page-sub">{{ tagMeta.date }}</div></div>
      <section class="panel"><div class="panel-head"><h2>摘要</h2></div><div class="prose">{{ tagMeta.summary || "—" }}</div></section>
      <section class="panel"><div class="panel-head"><h2>文档</h2></div>
        <pre v-if="tagDoc" class="doc">{{ tagDoc.content }}</pre><div v-else class="empty">该版本没有独立文档</div></section>
    </template>
  </template>
</template>
