<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from "vue";
import { t, lang } from "../i18n.js";
import { getRun, getTrace, runFile } from "../api.js";
import { pct, num, steps, dur } from "../format.js";
import { scenName, modelName, roleName, groupName, STRATEGY_COLOR } from "../labels.js";
import Status from "../components/Status.vue";
import Chart from "../components/Chart.vue";

const props = defineProps({ space: String, scenario: String, algo: String, exp_id: String });
const r = ref(null);
const tr = ref(null);
const win = ref(1000);
const err = ref(false);
let timer;
async function load() {
  const x = await getRun(props);
  err.value = !x;
  if (x) r.value = x;
  tr.value = await getTrace(props, 500, win.value);
}
onMounted(() => { load(); timer = setInterval(() => r.value?.status === "running" && load(), 20000); });
onUnmounted(() => clearInterval(timer));
watch(win, async () => { tr.value = await getTrace(props, 500, win.value); });

const m = computed(() => r.value?.metrics || {});
const total = computed(() => r.value?.params?.steps || m.value.total_steps);
const P = (v) => (v == null ? null : +(100 * v).toFixed(3));
const line = (name, color, ys, extra = {}) => ({ type: "line", name, color, showSymbol: false, data: tr.value.step.map((x, i) => [x, ys?.[i] ?? null]), ...extra });
const xAxis = computed(() => ({ type: "value", min: 0, max: total.value || undefined, axisLabel: { formatter: (v) => steps(v) } }));
const grid = { left: 52, right: 16, top: 34, bottom: 30 };
const tip = { trigger: "axis", confine: true, valueFormatter: (v) => (v == null ? "—" : v.toFixed(3)) };
const legend = { top: 0, right: 0, itemWidth: 14, textStyle: { fontSize: 11 } };
const arOpt = computed(() => { lang.value; const S = tr.value.series; return { grid, legend, tooltip: tip, xAxis: xAxis.value,
  yAxis: { type: "value", scale: true, axisLabel: { formatter: "{value}%" } },
  series: [line(t("训练局 AR", "episode AR"), STRATEGY_COLOR[r.value.strategy], S.ar?.map(P)),
    ...(m.value.ilp_ar ? [{ type: "line", data: [], markLine: { silent: true, symbol: "none", lineStyle: { type: "dashed", color: "#059669" },
      label: { formatter: `ILP AR ${(100 * m.value.ilp_ar).toFixed(1)}%（${t("测试集", "test set")}）`, position: "insideEndTop" }, data: [{ yAxis: P(m.value.ilp_ar) }] } }] : [])] }; });
const violOpt = computed(() => { lang.value; const S = tr.value.series; return { grid, legend, tooltip: tip, xAxis: xAxis.value,
  yAxis: { type: "value", min: 0, axisLabel: { formatter: "{value}%" } },
  series: [line(t("容量违约局", "capacity violated"), "#dc2626", S.cap_viol?.map(P)), line(t("隐私违约局", "privacy violated"), "#7c3aed", S.priv_viol?.map(P)),
    ...(S.success ? [line(t("无违约完成", "violation-free"), "#059669", S.success.map(P))] : [])] }; });
const rewOpt = computed(() => { lang.value; return { grid, legend, tooltip: tip, xAxis: xAxis.value, yAxis: { type: "value", scale: true },
  series: [line(t("每局回报", "episode return"), "#4f46e5", tr.value.series.reward)] }; });
const capText = (x) => (x === "none" ? t("无上限", "no cap") : x ?? "—");
const lamOpt = computed(() => { lang.value; const cap = [r.value.params?.lambda_max, m.value.lambda_max].find((x) => typeof x === "number"); /* "none": no cap line */ return { grid, legend, tooltip: tip, xAxis: xAxis.value,
  yAxis: { type: "value", min: 0, name: "λ" },
  series: [line("λ", "#d97706", tr.value.series.lambda),
    ...(cap != null ? [{ type: "line", data: [], markLine: { silent: true, symbol: "none", lineStyle: { type: "dashed", color: "#b91c1c" }, label: { formatter: `λ_max = ${cap}`, position: "insideEndTop" }, data: [{ yAxis: cap }] } }] : [])] }; });
// while training (no training_curve.csv yet): the [train] lines of the log
const ls = computed(() => r.value?.log_series);
const lsOpt = (key, name, color, scale = 1, unit = "") => { lang.value; return { grid, legend, tooltip: tip, xAxis: xAxis.value,
  yAxis: { type: "value", scale: true, axisLabel: { formatter: `{value}${unit}` } },
  series: [{ type: "line", name, color, symbolSize: 5, data: ls.value.step.map((x, i) => [x, ls.value[key][i] == null ? null : +(ls.value[key][i] * scale).toFixed(4)]) }] }; };
const hasLam = computed(() => tr.value?.series?.lambda?.some((v) => v));
const pngs = computed(() => (r.value?.files || []).filter((f) => f.endsWith(".png")));
const cfg = computed(() => Object.entries(r.value?.config || {}));
const fmt = (v) => (typeof v === "object" && v !== null ? JSON.stringify(v) : String(v));
</script>

<template>
  <div class="page">
    <div class="crumbs">
      <router-link to="/batches">{{ t("批次", "Batches") }}</router-link><span>/</span>
      <template v-if="r?.batch"><router-link :to="`/batches/${r.batch}`">{{ r.batch }}</router-link><span>/</span></template>
      <span>{{ scenario.toUpperCase() }} · {{ algo }} · {{ exp_id }}</span>
    </div>
    <div v-if="err && !r" class="card empty">{{ t("新面板里没有这个 run（只显示新批次的 run）。", "No such run in the new panel (runs of new batches only).") }}</div>
    <div v-else-if="!r" class="empty">{{ t("加载中…", "Loading…") }}</div>
    <template v-else>
      <div class="head">
        <div>
          <div class="row"><h1>{{ modelName(r.RL_algorithm, r.strategy) }}</h1><span class="tag">{{ scenName(scenario) }}</span><Status :s="r.status" :pct="r.live?.pct" /></div>
          <div class="row" style="margin-top: 8px">
            <span v-if="r.role" class="tag">{{ r.group ? groupName(r.group) : roleName(r.role) }}</span>
            <span v-for="(v, k) in r.params || {}" :key="k" class="tag accent">{{ k }} = {{ v }}</span>
            <span class="tag">{{ t("种子", "seed") }} {{ r.seed ?? r.config?.seed ?? "—" }}</span>
            <span class="tag">{{ steps(total) }} {{ t("步", "steps") }}</span>
            <span class="tag mono">results/{{ space }}/{{ scenario }}/{{ algo }}/{{ exp_id }}</span>
          </div>
        </div>
      </div>

      <div v-if="r.live" class="card">
        <div class="card-h"><h2>{{ t("训练中", "Training") }}</h2><span class="faint small">{{ t("来自训练日志，每 20 秒刷新", "from the training log, refreshes every 20 s") }}</span></div>
        <div class="card-b">
          <div class="row" style="gap: 12px; flex-wrap: nowrap">
            <div class="bar" style="flex: 1"><span class="s-running" :style="{ width: r.live.pct + '%' }" /></div>
            <b class="mono">{{ steps(r.live.step) }} / {{ steps(r.live.total) }}</b><span class="faint">{{ r.live.pct.toFixed(1) }}%</span>
          </div>
          <div class="kpis" style="margin-top: 12px">
            <div class="kpi"><div class="kpi-l">{{ t("步/秒（最近）", "Steps/s (recent)") }}</div><div class="kpi-v">{{ r.live.steps_per_sec ?? "—" }}</div><div class="kpi-s">{{ t("平均", "mean") }} {{ r.live.steps_per_sec_avg ?? "—" }}</div></div>
            <div class="kpi"><div class="kpi-l">{{ t("预计剩余", "ETA") }}</div><div class="kpi-v" style="font-size: 18px">{{ r.live.eta_seconds != null ? dur(r.live.eta_seconds) : "—" }}</div></div>
            <div class="kpi"><div class="kpi-l">{{ t("局数", "Episodes") }}</div><div class="kpi-v" style="font-size: 18px">{{ r.live.episodes != null ? Math.round(r.live.episodes).toLocaleString() : "—" }}</div></div>
            <div class="kpi"><div class="kpi-l">{{ t("近 500 局完成率", "Success (last 500)") }}</div><div class="kpi-v" style="font-size: 18px">{{ pct(r.live.success_last500) }}</div></div>
            <div v-if="r.live.lambda != null" class="kpi"><div class="kpi-l">{{ t("当前 λ", "Current λ") }}</div><div class="kpi-v" style="font-size: 18px">{{ num(r.live.lambda, 3) }}</div><div class="kpi-s">λ_max {{ capText(r.params?.lambda_max) }}</div></div>
          </div>
        </div>
      </div>

      <div v-if="r.metrics" class="kpis">
        <div class="kpi"><div class="kpi-l">{{ t("测试集相对 gap", "Test relative gap") }}</div><div class="kpi-v">{{ pct(m.rel_gap) }}</div><div class="kpi-s">AR {{ num(m.ar) }} / ILP {{ num(m.ilp_ar) }}</div></div>
        <div class="kpi"><div class="kpi-l">{{ t("容量违约", "Capacity violations") }}</div><div class="kpi-v">{{ pct(m.cap_viol) }}</div><div class="kpi-s">{{ t("测试实例比例", "of test instances") }}</div></div>
        <div class="kpi"><div class="kpi-l">{{ t("隐私违约", "Privacy violations") }}</div><div class="kpi-v">{{ pct(m.priv_viol) }}</div><div class="kpi-s">{{ t("测试实例比例", "of test instances") }}</div></div>
        <div class="kpi"><div class="kpi-l">EXIT</div><div class="kpi-v">{{ pct(m.exit) }}</div><div class="kpi-s">{{ t("死局", "dead end") }} {{ pct(m.dead_end) }}</div></div>
        <div v-if="r.strategy === 'lagrangian'" class="kpi"><div class="kpi-l">{{ t("末 λ", "Final λ") }}</div><div class="kpi-v">{{ num(m.final_lambda, 3) }}</div><div class="kpi-s">λ_max {{ capText(r.params?.lambda_max ?? m.lambda_max) }}</div></div>
        <div class="kpi"><div class="kpi-l">{{ t("训练用时", "Training time") }}</div><div class="kpi-v" style="font-size: 18px">{{ m.train_seconds != null ? dur(m.train_seconds) : "—" }}</div><div class="kpi-s">{{ m.episodes ?? "—" }} {{ t("局", "episodes") }} · {{ m.device || "" }}</div></div>
      </div>

      <div class="card">
        <div class="card-h">
          <h2>{{ t("训练过程", "During training") }}</h2>
          <span class="faint small" v-if="tr">{{ tr.n_episodes }} {{ t("局；滑动平均", "episodes; moving average") }}</span>
          <span class="spacer" />
          <select v-model.number="win"><option :value="100">{{ t("窗口 100 局", "window 100") }}</option><option :value="1000">{{ t("窗口 1000 局", "window 1000") }}</option><option :value="10000">{{ t("窗口 10000 局", "window 10000") }}</option></select>
        </div>
        <div class="card-b">
          <template v-if="!tr && ls?.step?.length">
            <div class="faint small" style="margin-bottom: 8px">{{ t("训练中：曲线来自训练日志（每个记录间隔一个点）；训练结束后换成每局的 training_curve.csv。", "Training: points from the training log (one per logging interval); replaced by the per-episode training_curve.csv when training ends.") }}</div>
            <div class="grid-2">
              <div><h3>{{ t("近 500 局完成率", "Success (last 500 episodes)") }}</h3><Chart :option="lsOpt('success_last500', t('近 500 局完成率', 'success (last 500)'), '#059669', 100, '%')" :height="220" /></div>
              <div v-if="ls.lambda.some((x) => x != null)"><h3>λ</h3><Chart :option="lsOpt('lambda', 'λ', '#d97706')" :height="220" /></div>
              <div><h3>{{ t("平均步/秒", "Mean steps/s") }}</h3><Chart :option="lsOpt('steps_per_sec', t('步/秒', 'steps/s'), '#4f46e5')" :height="220" /></div>
            </div>
          </template>
          <div v-else-if="!tr" class="empty">{{ t("没有 training_curve.csv。", "No training_curve.csv.") }}</div>
          <div v-else class="grid-2">
            <div><h3>{{ t("训练局 AR", "Episode AR") }}</h3><Chart :option="arOpt" :height="240" /></div>
            <div><h3>{{ t("违约局占比", "Episodes with violations") }}</h3><Chart :option="violOpt" :height="240" /></div>
            <div><h3>{{ t("每局回报", "Episode return") }}</h3><Chart :option="rewOpt" :height="240" /></div>
            <div v-if="hasLam"><h3>{{ t("λ（按局对偶上升）", "λ (dual ascent per episode)") }}</h3><Chart :option="lamOpt" :height="240" /></div>
          </div>
        </div>
      </div>

      <div v-if="r.checkpoints?.length" class="card">
        <div class="card-h"><h2>Checkpoints</h2></div>
        <div class="card-b row">
          <span v-for="c in r.checkpoints" :key="c" class="tag">{{ steps(c) }}<template v-if="r.ckpt_lambda?.[String(c)] != null"> · λ {{ num(r.ckpt_lambda[String(c)], 3) }}</template></span>
        </div>
      </div>

      <div v-if="pngs.length" class="card">
        <div class="card-h"><h2>{{ t("训练脚本生成的图", "Plots written by the training script") }}</h2></div>
        <div class="card-b grid-2"><a v-for="f in pngs" :key="f" :href="runFile(r, f)" target="_blank"><img :src="runFile(r, f)" :alt="f" style="width: 100%; border: 1px solid var(--border); border-radius: 8px" /></a></div>
      </div>

      <div class="card">
        <div class="card-h"><h2>{{ t("配置", "Config") }}</h2><span class="faint small">results.json</span></div>
        <div class="card-b table-wrap">
          <table class="tbl"><tbody><tr v-for="[k, v] in cfg" :key="k"><td class="mono" style="width: 200px">{{ k }}</td><td class="mono small" style="word-break: break-all">{{ fmt(v) }}</td></tr></tbody></table>
        </div>
      </div>
    </template>
  </div>
</template>
