<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { t, tl, lang } from "../i18n.js";
import { getBatch, getReport, getReportData } from "../api.js";
import { renderMarkdown } from "../markdown.js";
import { pct, num, steps, dur, when } from "../format.js";
import { scenName, mechName, learnerName, roleName, groupName, purposeName, statusName, STATUS_TONE } from "../labels.js";
import Status from "../components/Status.vue";
import Progress from "../components/Progress.vue";
import BatchResults from "../components/BatchResults.vue";
import BatchCharts from "../components/BatchCharts.vue";

const props = defineProps({ name: String });
const route = useRoute();
const router = useRouter();
const b = ref(null);
const err = ref(false);
const data = ref({ report: null, sweeps: [] });
const tab = computed(() => route.query.tab || "results");
const TABS = computed(() => [["results", t("结果", "Results")], ["charts", t("图表", "Charts")], ["runs", t("全部 run", "All runs")],
  ["report", t("报告", "Report")], ["config", t("配置", "Config")]]);
let timer;
async function load() {
  const x = await getBatch(props.name);
  err.value = !x;
  if (x) b.value = x;
  data.value = await getReportData(props.name);
}
onMounted(() => { load(); timer = setInterval(() => { if (b.value?.status === "running" || b.value?.report?.finished === false) load(); }, 15000); });
onUnmounted(() => clearInterval(timer));

// runs tab
const fRole = ref("all"), fScen = ref("all"), fStatus = ref("all");
const runs = computed(() => (b.value?.runs || []).filter((r) => (fRole.value === "all" || r.role === fRole.value)
  && (fScen.value === "all" || r.scenario === fScen.value) && (fStatus.value === "all" || r.status === fStatus.value)));
const roles = computed(() => [...new Set((b.value?.runs || []).map((r) => r.role))]);
const statuses = computed(() => [...new Set((b.value?.runs || []).map((r) => r.status))]);
const paramText = (p) => Object.entries(p || {}).filter(([, x]) => x != null && x !== "").map(([k, x]) => (k === "variant" ? x : `${k}=${x}`)).join(" · ");
const openRun = (r) => r.space && r.exp_id && router.push(`/runs/${r.space}/${r.scenario}/${r.algo}/${r.exp_id}`);

// report tab
const reportFile = ref("report.md");
const report = ref(null);
async function loadReport() {
  if (tab.value !== "report" || !b.value?.reports?.length) return;
  if (!b.value.reports.some((f) => f.file === reportFile.value)) reportFile.value = b.value.reports[0].file;
  report.value = await getReport(props.name, reportFile.value, lang.value);
}
watch([tab, reportFile, lang, () => b.value?.reports?.length], loadReport);
const reportHtml = computed(() => (report.value ? renderMarkdown(report.value.text) : ""));
const reportName = (f) => {
  const m = /^lam_sweep_(\w+)\.md$/.exec(f.file);
  if (m) return t(`λ_max 扫描（${m[1].toUpperCase()}）`, `λ_max sweep (${m[1].toUpperCase()})`);
  if (f.file === "report.md") return t("主报告", "Main report");
  return f.file.replace(/\.md$/, "");
};
const cfgRows = computed(() => Object.entries(b.value?.config || {}).filter(([k]) => !["title"].includes(k)));
const fmtCfg = (v) => (typeof v === "object" && v !== null ? JSON.stringify(v) : String(v));
const reportStage = computed(() => b.value?.report);
</script>

<template>
  <div class="page">
    <div class="crumbs"><router-link to="/batches">{{ t("批次", "Batches") }}</router-link><span>/</span><span v-if="b?.version">v{{ b.version }}</span><span v-if="b?.version">/</span><span>{{ name }}</span></div>
    <div v-if="err && !b" class="card empty">{{ t("没有这个批次。", "No such batch.") }}</div>
    <div v-else-if="!b" class="empty">{{ t("加载中…", "Loading…") }}</div>
    <template v-else>
      <div class="head">
        <div style="min-width: 0">
          <div class="row"><h1>{{ name }}</h1><Status :s="b.status" /><span v-if="b.purpose !== 'main'" class="tag">{{ purposeName(b.purpose) }}</span></div>
          <div class="head-sub">{{ tl(b.title) || b.description }}</div>
          <div class="row" style="margin-top: 8px">
            <span v-if="b.steps" class="tag">{{ steps(b.steps) }} {{ t("步", "steps") }}</span>
            <span v-if="b.seeds?.length" class="tag">{{ t("种子", "seeds") }} {{ b.seeds.join(", ") }}</span>
            <span v-if="b.space" class="tag mono">results/{{ b.space }}</span>
            <span v-if="b.commit" class="tag mono">{{ b.commit }}</span>
            <span class="tag">{{ t("开始", "started") }} {{ when(b.started_at) }}</span>
            <span class="tag">{{ b.status === "running" ? t("已运行", "elapsed") : t("用时", "took") }} {{ dur(b.elapsed) }}</span>
          </div>
        </div>
        <a class="btn" :href="`/classic/#/batches/${name}`">{{ t("在旧版面板中打开", "Open in classic panel") }} ↗</a>
      </div>

      <div v-if="b.kind !== 'eval'" class="card"><div class="card-b">
        <div class="row" style="gap: 16px">
          <div style="flex: 1; min-width: 260px"><Progress :b="b" legend /></div>
          <div class="kpi-v">{{ b.pct.toFixed(1) }}%</div>
        </div>
        <div v-if="reportStage && !reportStage.finished" class="notice info" style="margin-top: 10px">
          {{ t("训练已结束，正在生成报告", "Training is over; writing the report") }}<template v-if="reportStage.overall_pct != null"> · {{ reportStage.overall_pct }}%</template>
          <template v-if="reportStage.eta_minutes != null"> · {{ t("预计还需", "ETA") }} {{ dur(reportStage.eta_minutes * 60) }}</template>
        </div>
      </div></div>

      <nav class="tabs">
        <router-link v-for="[k, n] in TABS" :key="k" :to="{ query: { ...route.query, tab: k } }" :class="{ on: tab === k }" replace>
          {{ n }}<template v-if="k === 'report' && b.reports?.length"> · {{ b.reports.length }}</template>
        </router-link>
      </nav>

      <template v-if="b.kind === 'eval'">
        <div class="card"><div class="card-b dim">{{ t("评估批次（重新评估已保存的模型，不训练）。详细结果见旧版面板。", "Evaluation batch (re-evaluates saved models, no training). See the classic panel for details.") }}</div></div>
      </template>
      <BatchResults v-else-if="tab === 'results'" :b="b" :sweeps="data.sweeps" />
      <BatchCharts v-else-if="tab === 'charts'" :b="b" :data="data" />

      <div v-else-if="tab === 'runs'" class="card">
        <div class="card-h">
          <h2>{{ t("全部 run", "All runs") }}</h2><span class="faint small">{{ runs.length }} / {{ b.runs.length }}</span>
          <span class="spacer" />
          <select v-model="fRole"><option value="all">{{ t("全部角色", "All roles") }}</option><option v-for="r in roles" :key="r" :value="r">{{ roleName(r) }}</option></select>
          <select v-model="fScen"><option value="all">{{ t("全部场景", "All scenarios") }}</option><option v-for="s in b.scenarios" :key="s" :value="s">{{ s.toUpperCase() }}</option></select>
          <select v-model="fStatus"><option value="all">{{ t("全部状态", "All statuses") }}</option><option v-for="s in statuses" :key="s" :value="s">{{ statusName(s) }}</option></select>
        </div>
        <div class="card-b table-wrap">
          <table class="tbl">
            <thead><tr>
              <th>{{ t("场景", "Scen.") }}</th><th>{{ t("模型", "Model") }}</th><th>{{ t("约束处理", "Constraint") }}</th><th>{{ t("类别", "Role") }}</th><th>{{ t("参数", "Params") }}</th>
              <th>{{ t("种子", "Seed") }}</th><th>{{ t("状态", "Status") }}</th><th class="num">gap</th><th class="num">{{ t("容量违约", "Cap. viol.") }}</th><th class="num">{{ t("隐私违约", "Priv. viol.") }}</th>
              <th class="num">{{ t("末 λ", "Final λ") }}</th><th class="num">{{ t("训练用时", "Train time") }}</th><th>exp_id</th>
            </tr></thead>
            <tbody>
              <tr v-for="r in runs" :key="(r.exp_id || '') + r.key + r.scenario + r.seed" :class="{ link: r.exp_id && r.space }" @click="openRun(r)">
                <td>{{ r.scenario.toUpperCase() }}</td><td>{{ learnerName(r.learner) }}</td><td>{{ mechName(r.mechanism) }}</td>
                <td><span class="tag">{{ r.group ? groupName(r.group) : roleName(r.role) }}</span></td>
                <td class="small">{{ paramText(r.params) || "—" }}<span v-if="r.steps && r.steps !== b.steps" class="faint"> · {{ steps(r.steps) }}</span></td>
                <td>{{ r.seed ?? "—" }}</td>
                <td><Status :s="r.status" :pct="r.progress" /></td>
                <td class="num">{{ pct(r.metrics?.rel_gap) }}</td><td class="num">{{ pct(r.metrics?.cap_viol) }}</td><td class="num">{{ pct(r.metrics?.priv_viol) }}</td>
                <td class="num">{{ r.mechanism === "lagrange" ? num(r.metrics?.final_lambda, 3) : "" }}</td>
                <td class="num small">{{ r.metrics?.train_seconds != null ? dur(r.metrics.train_seconds) : "" }}</td>
                <td class="mono faint">{{ r.exp_id || "—" }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div v-else-if="tab === 'report'" class="card">
        <div v-if="!b.reports?.length" class="empty">{{ t("还没有报告（训练全部结束后生成）。", "No report yet (written once training is over).") }}</div>
        <template v-else>
          <div class="card-h">
            <div class="seg"><button v-for="f in b.reports" :key="f.file" :class="{ on: reportFile === f.file }" @click="reportFile = f.file">{{ reportName(f) }}</button></div>
            <span class="spacer" /><span v-if="report" class="faint small mono">{{ report.dir }}/{{ report.file }}</span>
          </div>
          <div class="card-b md" v-html="reportHtml" />
        </template>
      </div>

      <div v-else-if="tab === 'config'" class="card">
        <div class="card-b table-wrap">
          <table class="tbl">
            <tbody><tr v-for="[k, v] in cfgRows" :key="k"><td class="mono" style="width: 200px">{{ k }}</td><td class="mono small" style="word-break: break-all">{{ fmtCfg(v) }}</td></tr></tbody>
          </table>
        </div>
        <div class="card-note">{{ t("来自 logs/", "From logs/") }}{{ name }}/manifest.json{{ t("（计划清单 runs 见「全部 run」）", " (the planned runs are under All runs)") }}</div>
      </div>
    </template>
  </div>
</template>
