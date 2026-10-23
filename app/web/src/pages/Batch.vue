<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { t, tl, lang } from "../i18n.js";
import { getBatch, getReport, getReportData } from "../api.js";
import { renderMarkdown } from "../markdown.js";
import { pct, num, steps, dur, when } from "../format.js";
import { scenName, roleName, groupName, purposeName, statusName, STATUS_TONE } from "../labels.js";
import Status from "../components/Status.vue";
import Progress from "../components/Progress.vue";
import BatchResults from "../components/BatchResults.vue";
import BatchCharts from "../components/BatchCharts.vue";
import LiveRuns from "../components/LiveRuns.vue";
import Th from "../components/Th.vue";
import { useSort } from "../sort.js";

const props = defineProps({ name: String });
const route = useRoute();
const router = useRouter();
const b = ref(null);
const err = ref(false);
const data = ref({ report: null, sweeps: [] });
// a batch that trains nothing (eval / diag / data, docs/data_spec.md §6): its report, log and config only
const training = computed(() => b.value?.training !== false);
const tab = computed(() => route.query.tab || (training.value ? "results" : "report"));
const TABS = computed(() => [...(training.value ? [["results", t("结果", "Results")], ["charts", t("图表", "Charts")], ["runs", t("全部 run", "All runs")]] : []),
  ["report", training.value ? t("报告", "Report") : t("报告 / 日志", "Report / log")], ["config", t("配置", "Config")]]);
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
const SCEN = { lt: 0, eq: 1, gt: 2 };
const RLI = { PPO: 0, DQN: 1, DDQN: 2 };
const STI = { unconstraint: 0, lagrangian: 1, maskable: 2, repair: 3 };
const STATI = { running: 0, stopped: 1, waiting: 2, queued: 3, done: 4, skipped: 5 };
const runSort = useSort(runs, {
  scen: (r) => SCEN[r.scenario], rl: (r) => RLI[r.RL_algorithm], st: (r) => STI[r.strategy], role: (r) => r.group || r.role,
  params: (r) => paramText(r.params), seed: (r) => r.seed, status: (r) => STATI[r.status], gap: (r) => r.metrics?.rel_gap,
  cap: (r) => r.metrics?.cap_viol, priv: (r) => r.metrics?.priv_viol, exit: (r) => (r.strategy === "maskable" ? r.metrics?.exit : null), steps: (r) => r.live?.step ?? r.metrics?.total_steps,
  sps: (r) => r.live?.steps_per_sec, lam: (r) => (r.strategy === "lagrangian" ? r.live?.lambda ?? r.metrics?.final_lambda : null),
  time: (r) => r.live?.eta_seconds ?? r.metrics?.train_seconds, id: (r) => r.exp_id,
});
const hasMask = computed(() => (b.value?.runs || []).some((r) => r.strategy === "maskable"));
const hasLag = computed(() => (b.value?.runs || []).some((r) => r.strategy === "lagrangian"));
const roles = computed(() => [...new Set((b.value?.runs || []).map((r) => r.role))]);
const statuses = computed(() => [...new Set((b.value?.runs || []).map((r) => r.status))]);
// a run's params that differ from the batch's (docs/data_spec.md §1)
const paramText = (p) => Object.entries(p || {}).filter(([k, x]) => x != null && x !== "" && JSON.stringify(x) !== JSON.stringify(b.value?.params?.[k]))
  .map(([k, x]) => `${k}=${k === "steps" ? steps(x) : x}`).join(" · ");
const openRun = (r) => r.space && r.exp_id && router.push(`/runs/${r.space}/${r.scenario}/${r.dir}/${r.exp_id}`);

// report tab
const reportFile = ref("report.md");
const report = ref(null);
async function loadReport() {
  if (tab.value !== "report" || !b.value?.reports?.length) return;
  if (!b.value.reports.some((f) => f.file === reportFile.value)) reportFile.value = b.value.reports[0].file;
  report.value = await getReport(props.name, reportFile.value, lang.value);
}
watch([tab, reportFile, lang, () => b.value?.reports?.length], loadReport);
const reportHtml = computed(() => (report.value && !report.value.log ? renderMarkdown(report.value.text) : ""));
const reportName = (f) => {
  const m = /^lam_sweep_(\w+)\.md$/.exec(f.file);
  if (m) return t(`λ_max 扫描（${m[1].toUpperCase()}）`, `λ_max sweep (${m[1].toUpperCase()})`);
  if (f.file === "report.md") return t("主报告", "Main report");
  if (f.log) return `${t("日志", "log")} · ${f.file}`;
  return f.file.replace(/\.md$/, "");
};
const cfgRows = computed(() => Object.entries(b.value?.config || {}).filter(([k]) => !["title"].includes(k)));
const fmtCfg = (v) => (typeof v === "object" && v !== null ? JSON.stringify(v) : String(v));
const reportStage = computed(() => b.value?.report);
const liveRuns = computed(() => (b.value?.runs || []).filter((r) => r.status === "running"));
// steps per run: the main runs' (or, for a sweep-only batch, the sweep runs') planned steps
const runSteps = computed(() => {
  const rs = b.value?.runs || [];
  const main = rs.filter((r) => r.role === "main");
  const xs = (main.length ? main : rs).map((r) => r.params?.steps);
  return new Set(xs).size === 1 ? xs[0] : b.value?.params?.steps ?? null;
});
</script>

<template>
  <div class="page">
    <div class="crumbs"><router-link to="/batches">{{ t("批次", "Batches") }}</router-link><span>/</span><span v-if="b?.version">v{{ b.version }}</span><span v-if="b?.version">/</span><span>{{ name }}</span></div>
    <div v-if="err && !b" class="card empty">{{ t("新面板里没有这个批次（只显示 v5.0.3 起按新数据规范写出的批次）。", "No such batch in the new panel (only batches written to the new data spec, v5.0.3 on).") }}
      <a href="/classic/#/batches">{{ t("旧版面板", "Classic panel") }}</a></div>
    <div v-else-if="!b" class="empty">{{ t("加载中…", "Loading…") }}</div>
    <template v-else>
      <div class="head">
        <div style="min-width: 0">
          <div class="row"><h1>{{ name }}</h1><Status :s="b.status" /><span v-if="b.purpose !== 'main'" class="tag">{{ purposeName(b.purpose) }}</span></div>
          <div class="head-sub">{{ tl(b.title) }}</div>
          <div v-if="b.links?.length" class="row"><a v-for="l in b.links" :key="l.url" :href="l.url" target="_blank" rel="noopener">↗ {{ tl(l) }}</a></div>
          <div class="row" style="margin-top: 8px">
            <span v-if="runSteps" class="tag">{{ t("每个 run", "per run") }} {{ steps(runSteps) }} {{ t("步", "steps") }}</span>
            <span v-if="b.seeds?.length" class="tag">{{ t("种子", "seeds") }} {{ b.seeds.join(", ") }}</span>
            <span v-if="b.space" class="tag mono">results/{{ b.space }}</span>
            <router-link v-if="b.lambda_max_from" class="tag accent" :to="`/batches/${b.lambda_max_from}`">
              λ_max {{ t("来自", "from") }} {{ b.lambda_max_from }}<template v-if="b.params?.lambda_max">：{{ Object.entries(b.params?.lambda_max).map(([l, v]) => `${l.toUpperCase()} ${v}`).join(" · ") }}</template>
            </router-link>
            <span v-if="b.script" class="tag mono">{{ b.script }}</span>
            <router-link v-for="d in b.depends_on" :key="d" class="tag accent" :to="`/batches/${d}`">{{ t("输入", "input") }} {{ d }}</router-link>
            <span v-if="b.commit" class="tag mono">{{ b.commit }}</span>
            <span class="tag">{{ t("开始", "started") }} {{ when(b.started_at) }}</span>
            <span class="tag">{{ b.status === "running" ? t("已运行", "elapsed") : t("用时", "took") }} {{ dur(b.elapsed) }}</span>
          </div>
        </div>
      </div>

      <div v-if="training" class="card"><div class="card-b">
        <div class="row" style="gap: 16px">
          <div style="flex: 1; min-width: 260px"><Progress :b="b" legend /></div>
          <div class="kpi-v">{{ b.pct.toFixed(1) }}%</div>
        </div>
        <div class="row small dim" style="gap: 18px; margin-top: 10px">
          <span v-if="runSteps">{{ t("每个 run", "per run") }} <b>{{ steps(runSteps) }}</b> {{ t("步", "steps") }}</span>
          <span v-if="b.steps_per_sec">{{ t("合计", "throughput") }} <b>{{ b.steps_per_sec.toLocaleString() }}</b> {{ t("步/秒", "steps/s") }}</span>
          <span v-if="b.eta">{{ t("在跑的这批 run 结束", "running runs done") }} <b>{{ dur(b.eta.running_done_seconds) }}</b> {{ t("后", "from now") }}</span>
          <span v-if="b.eta">{{ t("预计训练全部结束", "training done") }} <b>{{ dur(b.eta.seconds) }}</b> {{ t("后", "from now") }}（{{ t("约", "at") }} {{ when(Date.now() / 1000 + b.eta.seconds) }}，{{ t("不含报告", "report not included") }}）</span>
        </div>
        <div v-if="reportStage && !reportStage.finished" class="notice info" style="margin-top: 10px">
          {{ t("训练已结束，正在生成报告", "Training is over; writing the report") }}<template v-if="reportStage.step"> · <span class="mono">{{ reportStage.step }}</span></template>
        </div>
      </div></div>

      <nav class="tabs">
        <router-link v-for="[k, n] in TABS" :key="k" :to="{ query: { ...route.query, tab: k } }" :class="{ on: tab === k }" replace>
          {{ n }}<template v-if="k === 'report' && b.reports?.length"> · {{ b.reports.length }}</template>
        </router-link>
      </nav>

      <div v-if="tab === 'results' && liveRuns.length" class="card">
        <div class="card-h"><h2>{{ t("运行中的 run", "Running runs") }}</h2><span class="faint small">{{ liveRuns.length }} · {{ t("数据来自各 run 的训练日志，每 15 秒刷新", "from each run's training log, refreshes every 15 s") }}</span></div>
        <div class="card-b"><LiveRuns :runs="liveRuns" :params="b.params" :collapse="4" /></div>
      </div>
      <BatchResults v-if="tab === 'results'" :b="b" :sweeps="data.sweeps" />
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
          <table class="tbl pin">
            <thead><tr>
              <Th :s="runSort" k="scen">{{ t("场景", "Scen.") }}</Th><Th :s="runSort" k="rl">RL_algorithm</Th><Th :s="runSort" k="st">strategy</Th><Th :s="runSort" k="role">{{ t("类别", "Role") }}</Th><Th :s="runSort" k="params">params</Th>
              <Th :s="runSort" k="seed">seed</Th><Th :s="runSort" k="status">{{ t("状态", "Status") }}</Th><Th :s="runSort" k="gap" num>gap</Th><Th :s="runSort" k="cap" num>{{ t("容量违约", "Cap. viol.") }}</Th><Th :s="runSort" k="priv" num>{{ t("隐私违约", "Priv. viol.") }}</Th><Th v-if="hasMask" :s="runSort" k="exit" num :title="t('只有 maskable 有 EXIT', 'EXIT exists for maskable only')">EXIT</Th>
              <Th :s="runSort" k="steps" num>{{ t("步数", "Steps") }}</Th><Th :s="runSort" k="sps" num>{{ t("步/秒", "Steps/s") }}</Th><Th v-if="hasLag" :s="runSort" k="lam" num>{{ t("λ（实时 / 末）", "λ (live / final)") }}</Th><Th :s="runSort" k="time" num>{{ t("剩余 / 用时", "ETA / took") }}</Th><Th :s="runSort" k="id">exp_id</Th>
            </tr></thead>
            <tbody>
              <tr v-for="r in runSort.sorted.value" :key="(r.exp_id || '') + r.key + r.scenario + r.seed" :class="{ link: r.exp_id && r.space }" @click="openRun(r)">
                <td>{{ r.scenario.toUpperCase() }}</td><td>{{ r.RL_algorithm }}</td><td>{{ r.strategy }}</td>
                <td><span class="tag">{{ r.group ? groupName(r.group) : roleName(r.role) }}</span></td>
                <td class="small">{{ paramText(r.params) || "—" }}</td>
                <td>{{ r.seed ?? "—" }}</td>
                <td><Status :s="r.status" :pct="r.progress" /></td>
                <td class="num">{{ pct(r.metrics?.rel_gap) }}</td><td class="num">{{ pct(r.metrics?.cap_viol) }}</td><td class="num">{{ pct(r.metrics?.priv_viol) }}</td><td v-if="hasMask" class="num">{{ r.strategy === "maskable" ? pct(r.metrics?.exit) : "" }}</td>
                <td class="num small mono" style="white-space: nowrap">{{ r.live ? `${steps(r.live.step)} / ${steps(r.live.total)}` : r.metrics ? steps(r.metrics.total_steps) : "" }}</td>
                <td class="num mono">{{ r.live?.steps_per_sec ?? "" }}</td>
                <td v-if="hasLag" class="num mono">{{ r.strategy === "lagrangian" ? (r.live ? num(r.live.lambda, 3) : num(r.metrics?.final_lambda, 3)) : "" }}</td>
                <td class="num small">{{ r.live?.eta_seconds != null ? dur(r.live.eta_seconds) : r.metrics?.train_seconds != null ? dur(r.metrics.train_seconds) : "" }}</td>
                <td class="mono faint">{{ r.exp_id || "—" }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div v-else-if="tab === 'report'" class="card">
        <div v-if="!b.reports?.length" class="empty">{{ training ? t("还没有报告（训练全部结束后生成）。", "No report yet (written once training is over).") : t("这个批次没有报告或日志。", "This batch has no report or log.") }}</div>
        <template v-else>
          <div class="card-h">
            <div class="seg"><button v-for="f in b.reports" :key="f.file" :class="{ on: reportFile === f.file }" @click="reportFile = f.file">{{ reportName(f) }}</button></div>
            <span class="spacer" /><span v-if="report" class="faint small mono">{{ report.dir }}/{{ report.file }}</span>
          </div>
          <div v-if="report?.log" class="card-b"><pre class="log">{{ report.text }}</pre></div>
          <div v-else class="card-b md" v-html="reportHtml" />
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
