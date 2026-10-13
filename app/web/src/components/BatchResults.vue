<script setup>
// Result tab of a batch. Learner x mechanism batches (v5.0.x): one matrix per scenario (rows = learner, columns =
// constraint handling), then the lambda_max sweep per learner. Other batches: one row per variant, one column per
// scenario. Cells average the finished seeds.
import { computed } from "vue";
import { useRouter } from "vue-router";
import { t, tl } from "../i18n.js";
import { pct, num, steps } from "../format.js";
import { SCENARIOS, LEARNERS, MECHS, scenName, mechName, learnerName, MECH_COLOR, statusName, STATUS_TONE } from "../labels.js";

const props = defineProps({ b: Object, sweeps: { type: Array, default: () => [] } });
const router = useRouter();

const mean = (xs) => { const v = xs.filter((x) => x != null && !Number.isNaN(x)); return v.length ? v.reduce((a, c) => a + c, 0) / v.length : null; };
const STAT_ORDER = ["running", "stopped", "waiting", "queued", "skipped", "done"];
function cellOf(runs) {
  if (!runs.length) return null;
  const done = runs.filter((r) => r.status === "done" && r.metrics);
  const m = (k) => mean(done.map((r) => r.metrics[k]));
  const status = done.length === runs.length ? "done" : STAT_ORDER.find((s) => runs.some((r) => r.status === s));
  const prog = runs.find((r) => r.status === "running")?.progress;
  return { runs, done, status, progress: prog, ref: runs[0].role === "reference", label: runs[0].label,
    gap: m("rel_gap"), cap: m("cap_viol"), priv: m("priv_viol"), exit: m("exit"), lam: m("final_lambda"),
    lamMax: runs[0].params?.lambda_max ?? done[0]?.metrics?.lambda_max, waiting: runs.flatMap((r) => r.depends_on || []) };
}
const open = (r) => r && r.space && r.exp_id && router.push(`/runs/${r.space}/${r.scenario}/${r.algo}/${r.exp_id}`);

const main = computed(() => props.b.runs.filter((r) => r.role === "main"));
const isMatrix = computed(() => {
  if (!main.value.length) return false;
  const seen = new Map();
  for (const r of main.value) {
    if (!LEARNERS.includes(r.learner) || !MECHS.includes(r.mechanism)) return false;
    const k = `${r.scenario}|${r.learner}|${r.mechanism}`, sig = `${r.key}|${JSON.stringify(r.params)}`;
    if (seen.has(k) && seen.get(k) !== sig) return false;
    seen.set(k, sig);
  }
  return true;
});
const learners = computed(() => LEARNERS.filter((l) => main.value.some((r) => r.learner === l)));
const mechs = computed(() => MECHS.filter((m) => main.value.some((r) => r.mechanism === m) || props.b.references.some((r) => r.mechanism === m)));
const scens = computed(() => SCENARIOS.filter((s) => props.b.runs.some((r) => r.scenario === s)));
function cell(s, l, m) {
  const own = main.value.filter((r) => r.scenario === s && r.learner === l && r.mechanism === m);
  if (own.length) return cellOf(own);
  return cellOf(props.b.references.filter((r) => r.scenario === s && r.learner === l && r.mechanism === m));
}

// lambda_max sweeps: per learner (groups lam_sweep_<learner>; v5.0.3 before per-learner sweeps: lam_sweep)
const chosen = (l) => { const x = props.b.lambda_max; return x == null ? null : typeof x === "object" ? x[l] : x; };
const sweepRuns = computed(() => props.b.runs.filter((r) => r.role === "sweep"));
const sweepLearners = computed(() => LEARNERS.filter((l) => sweepRuns.value.some((r) => r.learner === l)));
const sweepValues = (l) => [...new Set(sweepRuns.value.filter((r) => r.learner === l).map((r) => r.params?.lambda_max))].sort((a, b) => a - b);
const sweepCell = (l, v, s) => cellOf(sweepRuns.value.filter((r) => r.learner === l && r.params?.lambda_max === v && r.scenario === s));
const sweepJson = (l) => props.sweeps.find((x) => x.learner === l);
const effGap = (l, v, s) => sweepJson(l)?.rows.find((r) => r.lambda_max === v && r.scenario === s)?.eff_gap;
const effMean = (l, v) => sweepJson(l)?.score?.[String(v)] ?? sweepJson(l)?.score?.[`${v}`];

// other batches: one row per variant
const variants = computed(() => {
  const out = new Map();
  for (const r of main.value) {
    const k = r.key || `${r.algo}|${JSON.stringify(r.params)}`;
    if (!out.has(k)) out.set(k, { k, algo: r.algo, key: r.key, params: r.params, learner: r.learner, mechanism: r.mechanism });
  }
  return [...out.values()];
});
const variantCell = (v, s) => cellOf(main.value.filter((r) => (r.key || `${r.algo}|${JSON.stringify(r.params)}`) === v.k && r.scenario === s));
const paramText = (p) => Object.entries(p || {}).filter(([, x]) => x != null && x !== "").map(([k, x]) => (k === "variant" ? x : `${k}=${x}`)).join(" · ");
const gapTone = (g) => (g == null ? "" : g <= 0.05 ? "good" : g <= 0.1 ? "mid" : "bad");
</script>

<template>
  <div class="stack">
    <template v-if="isMatrix">
      <div class="legend-row small dim">
        {{ t("每格一个模型：大字为测试集相对 gap =（ILP AR − AR）/ ILP AR，只在无违约的测试实例上计算；下方为有容量 / 隐私违约的测试实例比例。点击已完成的格子查看该 run。",
             "One model per cell: the large number is the test relative gap = (ILP AR − AR) / ILP AR over the violation-free test instances; below, the share of test instances with a capacity / privacy violation. Click a finished cell to open its run.") }}
      </div>
      <div class="mx-grid">
        <div v-for="s in scens" :key="s" class="card">
          <div class="card-h"><h2>{{ scenName(s) }}</h2></div>
          <div class="card-b table-wrap">
            <table class="mx">
              <thead><tr><th></th><th v-for="m in mechs" :key="m"><i class="dot" :style="{ background: MECH_COLOR[m] }" />{{ mechName(m) }}</th></tr></thead>
              <tbody>
                <tr v-for="l in learners" :key="l">
                  <th class="rowh">{{ learnerName(l) }}</th>
                  <td v-for="m in mechs" :key="m">
                    <template v-for="c in [cell(s, l, m)]" :key="m">
                      <div v-if="!c" class="mx-cell empty-cell">—</div>
                      <div v-else class="mx-cell" :class="{ click: c.done.length, ref: c.ref }" @click="c.done.length && open(c.done[0])">
                        <template v-if="c.status === 'done'">
                          <div class="mx-gap" :class="gapTone(c.gap)">{{ pct(c.gap) }}</div>
                          <div class="mx-sub">{{ t("容量", "cap") }} {{ pct(c.cap) }} · {{ t("隐私", "priv") }} {{ pct(c.priv) }}</div>
                          <div v-if="c.exit" class="mx-sub">EXIT {{ pct(c.exit) }}</div>
                          <div v-if="m === 'lagrange'" class="mx-sub lam">λ_max {{ c.lamMax ?? "?" }} · {{ t("末 λ", "final λ") }} {{ num(c.lam, 2) }}</div>
                          <div v-if="c.ref" class="mx-sub faint">{{ t("参照 · 非本批次训练", "reference · not trained here") }}</div>
                        </template>
                        <template v-else>
                          <span class="badge" :class="STATUS_TONE[c.status]">{{ statusName(c.status) }}<template v-if="c.progress != null"> · {{ c.progress.toFixed(0) }}%</template></span>
                          <div v-if="c.status === 'waiting'" class="mx-sub faint">{{ t("等 λ_max 扫描", "for the λ_max sweep") }}</div>
                          <div v-if="m === 'lagrange' && c.lamMax != null" class="mx-sub lam">λ_max {{ c.lamMax }}</div>
                        </template>
                      </div>
                    </template>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </template>

    <div v-else-if="main.length" class="card">
      <div class="card-h"><h2>{{ t("结果", "Results") }}</h2><span class="faint small">{{ t("每行一个变体，每格为各种子的均值", "one row per variant; cells average the seeds") }}</span></div>
      <div class="card-b table-wrap">
        <table class="tbl">
          <thead><tr><th>{{ t("变体", "Variant") }}</th><th v-for="s in scens" :key="s">{{ scenName(s) }}</th></tr></thead>
          <tbody>
            <tr v-for="v in variants" :key="v.k">
              <td><b>{{ v.key || v.algo }}</b><div class="faint small">{{ v.algo }}<template v-if="paramText(v.params)"> · {{ paramText(v.params) }}</template></div></td>
              <td v-for="s in scens" :key="s">
                <template v-for="c in [variantCell(v, s)]" :key="s">
                  <span v-if="!c" class="faint">—</span>
                  <div v-else-if="c.status === 'done'" :class="{ click: c.done.length }" @click="open(c.done[0])" style="cursor: pointer">
                    <b :class="['gapv', gapTone(c.gap)]">{{ pct(c.gap) }}</b>
                    <span class="faint small"> · {{ t("容量", "cap") }} {{ pct(c.cap) }} · {{ t("隐私", "priv") }} {{ pct(c.priv) }}</span>
                    <span class="faint small" v-if="c.runs.length > 1"> · {{ c.done.length }} {{ t("种子", "seeds") }}</span>
                  </div>
                  <span v-else class="badge" :class="STATUS_TONE[c.status]">{{ statusName(c.status) }}<template v-if="c.progress != null"> · {{ c.progress.toFixed(0) }}%</template></span>
                </template>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div v-if="sweepLearners.length" class="card">
      <div class="card-h">
        <h2>{{ t("λ_max 扫描", "λ_max sweep") }}</h2>
        <span class="faint small">{{ t("每个 learner 各扫一次", "one sweep per learner") }}<template v-if="b.lam_sweep"> · {{ steps(b.lam_sweep.steps) }} {{ t("步", "steps") }}</template></span>
      </div>
      <div class="card-b">
        <div class="sw-grid">
          <div v-for="l in sweepLearners" :key="l">
            <div class="row" style="margin-bottom: 6px">
              <b>Lagrangian-{{ learnerName(l) }}</b>
              <span v-if="chosen(l) != null && !Number.isNaN(chosen(l))" class="tag accent">{{ t("选定", "chosen") }} λ_max = {{ chosen(l) }}</span>
              <span v-else class="tag">{{ t("尚未选定", "not chosen yet") }}</span>
            </div>
            <table class="tbl">
              <thead><tr><th>λ_max</th><th v-for="s in scens" :key="s">{{ s.toUpperCase() }}</th><th v-if="sweepJson(l)" class="num">{{ t("平均有效 gap", "mean eff. gap") }}</th></tr></thead>
              <tbody>
                <tr v-for="v in sweepValues(l)" :key="v" :class="{ chosen: chosen(l) === v }">
                  <td><b>{{ v }}</b></td>
                  <td v-for="s in scens" :key="s">
                    <template v-for="c in [sweepCell(l, v, s)]" :key="s">
                      <div v-if="c && c.status === 'done'" style="cursor: pointer" @click="open(c.done[0])">
                        <b>{{ pct(effGap(l, v, s) ?? c.gap) }}</b>
                        <div class="faint small">{{ t("违约", "viol.") }} {{ pct((c.cap ?? 0) + (c.priv ?? 0)) }} · λ {{ num(c.lam, 2) }}</div>
                      </div>
                      <span v-else-if="c" class="badge" :class="STATUS_TONE[c.status]">{{ statusName(c.status) }}<template v-if="c.progress != null"> · {{ c.progress.toFixed(0) }}%</template></span>
                    </template>
                  </td>
                  <td v-if="sweepJson(l)" class="num"><b>{{ pct(effMean(l, v)) }}</b></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <div class="card-note">
        {{ t("λ 在训练中按局对偶上升自适应：λ ← clip(λ + η(J_C − d), 0, λ_max)，λ_max 只是上限。有效 gap：有违约的实例按 AR = 0 计；每个 learner 取三场景平均有效 gap 最低的 λ_max，它的 Lagrangian 正式 run 随后开始。没有扫描报告时格内显示无违约实例上的 gap。",
             "λ adapts during training by dual ascent per episode: λ ← clip(λ + η(J_C − d), 0, λ_max); λ_max is only its cap. Effective gap: a violating instance counts as AR = 0; each learner takes the λ_max with the lowest mean effective gap over the scenarios, then its main Lagrangian runs start. Without the sweep report a cell shows the gap on violation-free instances.") }}
      </div>
    </div>
  </div>
</template>

<style scoped>
.legend-row { margin-bottom: -6px; }
.mx-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 16px; }
.sw-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
@media (max-width: 860px) { .mx-grid, .sw-grid { grid-template-columns: 1fr; } }
table.mx { width: 100%; border-collapse: separate; border-spacing: 6px; margin: -6px; }
.mx th { font-size: 12px; color: var(--dim); font-weight: 600; text-align: left; white-space: nowrap; }
.mx th.rowh { font-size: 13px; color: var(--text); padding-right: 4px; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 5px; }
.mx-cell { border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; min-height: 64px; background: var(--panel-2); min-width: 92px; }
.mx-cell.click { cursor: pointer; }
.mx-cell.click:hover { border-color: var(--accent); }
.mx-cell.ref { border-style: dashed; }
.empty-cell { color: var(--faint); display: grid; place-items: center; }
.mx-gap { font-size: 19px; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1.25; }
.mx-sub { font-size: 11.5px; color: var(--dim); white-space: nowrap; }
.mx-sub.lam { color: #b45309; }
.good { color: var(--ok); }
.mid { color: var(--text); }
.bad { color: var(--bad); }
tr.chosen td { background: var(--accent-soft); }
</style>
