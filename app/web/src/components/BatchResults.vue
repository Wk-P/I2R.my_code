<script setup>
// Result tab of a batch. RL_algorithm x strategy batches: one matrix per scenario (rows = RL_algorithm, columns =
// strategy), then the lambda_max sweep per RL_algorithm. Other batches: one row per variant, one column per
// scenario. Cells average the finished seeds.
import { computed } from "vue";
import { useRouter } from "vue-router";
import { t, tl } from "../i18n.js";
import { pct, num, steps } from "../format.js";
import { SCENARIOS, RL_ALGORITHMS, STRATEGIES, scenName, STRATEGY_COLOR, statusName, STATUS_TONE } from "../labels.js";

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
  const live = runs.find((r) => r.status === "running")?.live;
  return { runs, done, status, progress: prog, live, ref: runs[0].role === "reference", label: runs[0].label,
    gap: m("rel_gap"), cap: m("cap_viol"), priv: m("priv_viol"), viol: m("viol"), exit: m("exit"), lam: m("final_lambda"),
    lamMax: runs[0].params?.lambda_max ?? done[0]?.metrics?.lambda_max, waiting: runs.flatMap((r) => r.depends_on || []) };
}
const open = (r) => r && r.space && r.exp_id && router.push(`/runs/${r.space}/${r.scenario}/${r.dir}/${r.exp_id}`);

const main = computed(() => props.b.runs.filter((r) => r.role === "main"));
const isMatrix = computed(() => {
  if (!main.value.length) return false;
  const seen = new Map();
  for (const r of main.value) {
    if (!RL_ALGORITHMS.includes(r.RL_algorithm) || !STRATEGIES.includes(r.strategy)) return false;
    const k = `${r.scenario}|${r.RL_algorithm}|${r.strategy}`, sig = `${r.key}|${JSON.stringify(r.params)}`;
    if (seen.has(k) && seen.get(k) !== sig) return false;
    seen.set(k, sig);
  }
  return true;
});
const learners = computed(() => RL_ALGORITHMS.filter((l) => main.value.some((r) => r.RL_algorithm === l)));
const mechs = computed(() => STRATEGIES.filter((m) => main.value.some((r) => r.strategy === m) || props.b.references.some((r) => r.strategy === m)));
const scens = computed(() => SCENARIOS.filter((s) => props.b.runs.some((r) => r.scenario === s)));
function cell(s, l, m) {
  const own = main.value.filter((r) => r.scenario === s && r.RL_algorithm === l && r.strategy === m);
  if (own.length) return cellOf(own);
  return cellOf(props.b.references.filter((r) => r.scenario === s && r.RL_algorithm === l && r.strategy === m));
}

// lambda_max sweeps: one per RL_algorithm (groups lambda_max_sweep_<RL_algorithm>)
const chosen = (l) => { const x = props.b.params?.lambda_max; return x == null ? null : typeof x === "object" ? x[l] : x; };
const sweepRuns = computed(() => props.b.runs.filter((r) => r.role === "sweep"));
const sweepLearners = computed(() => RL_ALGORITHMS.filter((l) => sweepRuns.value.some((r) => r.RL_algorithm === l)));
const sweepValues = (l) => [...new Set(sweepRuns.value.filter((r) => r.RL_algorithm === l).map((r) => r.params?.lambda_max))].sort((a, b) => a - b);
const sweepCell = (l, v, s) => cellOf(sweepRuns.value.filter((r) => r.RL_algorithm === l && r.params?.lambda_max === v && r.scenario === s));
const sweepJson = (l) => props.sweeps.find((x) => x.RL_algorithm === l);
const effGap = (l, v, s) => sweepJson(l)?.rows.find((r) => r.lambda_max === v && r.scenario === s)?.eff_gap;
const freeMean = (l, v) => {
  const j = sweepJson(l);
  if (j?.viol_free_mean) return j.viol_free_mean[String(v)];
  const rs = (j?.rows || []).filter((r) => r.lambda_max === v);
  return rs.length ? rs.reduce((a, r) => a + r.viol_free, 0) / rs.length : null;
};
const effMean = (l, v) => sweepJson(l)?.score?.[String(v)] ?? sweepJson(l)?.score?.[`${v}`];

// other batches: one row per variant
const variants = computed(() => {
  const out = new Map();
  for (const r of main.value) {
    const k = r.key || `${r.dir}|${JSON.stringify(r.params)}`;
    if (!out.has(k)) out.set(k, { k, key: r.key, params: r.params, rl: r.RL_algorithm, strategy: r.strategy });
  }
  return [...out.values()];
});
const variantCell = (v, s) => cellOf(main.value.filter((r) => (r.key || `${r.dir}|${JSON.stringify(r.params)}`) === v.k && r.scenario === s));
const paramList = (p) => Object.entries(p || {}).filter(([k, x]) => x != null && x !== "" && JSON.stringify(x) !== JSON.stringify(props.b.params?.[k]))
  .map(([k, x]) => `${k}=${k === "steps" ? steps(x) : x}`);
const paramText = (p) => paramList(p).join(" · ");
const gapTone = (g) => (g == null ? "" : g <= 0.05 ? "good" : g <= 0.1 ? "mid" : "bad");
</script>

<template>
  <div class="stack">
    <template v-if="isMatrix">
      <div class="legend-row small dim">
        {{ t("每格一个模型：大字为测试集相对 gap =（ILP AR − AR）/ ILP AR，只在无违约的测试实例上计算；下方为有容量 / 隐私违约的测试实例比例。unconstraint（v6.0.1 起）不罚违约，其 AR 为全部测试实例上的目标 AR（违约放置也计入，ECU 利用率截断到 1）。点击已完成的格子查看该 run。",
             "One model per cell: the large number is the test relative gap = (ILP AR − AR) / ILP AR over the violation-free test instances; below, the share of test instances with a capacity / privacy violation. Unconstraint (from v6.0.1) is not penalised for violations; its AR is the objective AR over all test instances (violating placements included, ECU utilisation clipped at 1). Click a finished cell to open its run.") }}
      </div>
      <div class="mx-grid" :style="{ gridTemplateColumns: `repeat(auto-fit, minmax(min(${80 + 150 * mechs.length}px, 100%), 1fr))` }">
        <div v-for="s in scens" :key="s" class="card">
          <div class="card-h"><h2>{{ scenName(s) }}</h2></div>
          <div class="card-b table-wrap">
            <table class="mx">
              <thead><tr><th></th><th v-for="m in mechs" :key="m"><i class="dot" :style="{ background: STRATEGY_COLOR[m] }" />{{ m }}</th></tr></thead>
              <tbody>
                <tr v-for="l in learners" :key="l">
                  <th class="rowh">{{ l }}</th>
                  <td v-for="m in mechs" :key="m">
                    <template v-for="c in [cell(s, l, m)]" :key="m">
                      <div v-if="!c" class="mx-cell empty-cell">—</div>
                      <div v-else class="mx-cell" :class="{ click: c.done.length, ref: c.ref }" @click="c.done.length && open(c.done[0])">
                        <template v-if="c.status === 'done'">
                          <div class="mx-gap" :class="gapTone(c.gap)">{{ pct(c.gap) }}</div>
                          <div class="mx-sub">{{ t("容量", "cap") }} {{ pct(c.cap) }} · {{ t("隐私", "priv") }} {{ pct(c.priv) }}</div>
                          <div v-if="m === 'maskable'" class="mx-sub">EXIT {{ pct(c.exit) }}</div>
                          <div v-if="m === 'lagrangian'" class="mx-sub lam">λ_max {{ c.lamMax ?? "?" }} · {{ t("末 λ", "final λ") }} {{ num(c.lam, 2) }}</div>
                          <div v-if="c.ref" class="mx-sub faint">{{ t("参照 · 非本批次训练", "reference · not trained here") }}</div>
                        </template>
                        <template v-else>
                          <template v-if="c.done.length">
                            <div class="mx-gap" :class="gapTone(c.gap)">{{ pct(c.gap) }}</div>
                            <div class="mx-sub">{{ t("容量", "cap") }} {{ pct(c.cap) }} · {{ t("隐私", "priv") }} {{ pct(c.priv) }}<template v-if="m === 'maskable'"> · EXIT {{ pct(c.exit) }}</template></div>
                            <div class="mx-sub faint">{{ t("已完成种子", "seeds done") }} {{ c.done.length }}/{{ c.runs.length }}</div>
                          </template>
                          <span class="badge" :class="STATUS_TONE[c.status]">{{ statusName(c.status) }}<template v-if="c.progress != null"> · {{ c.progress.toFixed(0) }}%</template></span>
                          <div v-if="c.live" class="mx-sub mono">{{ steps(c.live.step) }} / {{ steps(c.live.total) }} · {{ c.live.steps_per_sec ?? "—" }} {{ t("步/秒", "/s") }}</div>
                          <div v-if="m === 'lagrangian' && c.live?.lambda != null" class="mx-sub lam">{{ t("当前 λ", "λ now") }} {{ num(c.live.lambda, 3) }}</div>
                          <div v-if="c.status === 'waiting'" class="mx-sub faint">{{ t("等 λ_max 扫描", "for the λ_max sweep") }}</div>
                          <div v-if="m === 'lagrangian' && c.lamMax != null" class="mx-sub lam">λ_max {{ c.lamMax }}</div>
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
      <!-- phones: one block per variant, its params as separate tags, one line per scenario -->
      <div class="card-b show-sm">
        <div v-for="v in variants" :key="'c' + v.k" class="vc">
          <div class="vc-h"><b>{{ v.rl }} · {{ v.strategy }}</b><span v-for="x in paramList(v.params)" :key="x" class="tag">{{ x }}</span></div>
          <div v-for="s in scens" :key="s" class="vc-r">
            <span class="dim small">{{ scenName(s) }}</span>
            <div><template v-for="c in [variantCell(v, s)]" :key="s">
                <span v-if="!c" class="faint">—</span>
                <div v-else-if="c.status === 'done'" :class="{ click: c.done.length }" @click="open(c.done[0])" style="cursor: pointer">
                  <b :class="['gapv', gapTone(c.gap)]">{{ pct(c.gap) }}</b>
                  <span class="faint small"> · {{ t("容量", "cap") }} {{ pct(c.cap) }} · {{ t("隐私", "priv") }} {{ pct(c.priv) }}</span>
                  <span class="faint small" v-if="v.strategy === 'maskable'"> · EXIT {{ pct(c.exit) }}</span>
                  <span class="faint small" v-if="c.runs.length > 1"> · {{ c.done.length }} {{ t("种子", "seeds") }}</span>
                </div>
                <div v-else>
                  <span v-if="c.done.length" :class="{ click: true }" @click="open(c.done[0])" style="cursor: pointer"><b :class="['gapv', gapTone(c.gap)]">{{ pct(c.gap) }}</b>
                    <span class="faint small"> · {{ t("容量", "cap") }} {{ pct(c.cap) }} · {{ t("隐私", "priv") }} {{ pct(c.priv) }}<template v-if="v.strategy === 'maskable'"> · EXIT {{ pct(c.exit) }}</template> · {{ t("已完成种子", "seeds done") }} {{ c.done.length }}/{{ c.runs.length }}</span><br /></span>
                  <span class="badge" :class="STATUS_TONE[c.status]">{{ statusName(c.status) }}<template v-if="c.progress != null"> · {{ c.progress.toFixed(0) }}%</template></span>
                </div>
              </template></div>
          </div>
        </div>
      </div>
      <div class="card-b table-wrap hide-sm">
        <table class="tbl">
          <thead><tr><th>{{ t("变体", "Variant") }}</th><th v-for="s in scens" :key="s">{{ scenName(s) }}</th></tr></thead>
          <tbody>
            <tr v-for="v in variants" :key="v.k">
              <td><b>{{ v.rl }} · {{ v.strategy }}</b><div v-if="paramText(v.params)" class="faint small">{{ paramText(v.params) }}</div></td>
              <td v-for="s in scens" :key="s">
                <template v-for="c in [variantCell(v, s)]" :key="s">
                  <span v-if="!c" class="faint">—</span>
                  <div v-else-if="c.status === 'done'" :class="{ click: c.done.length }" @click="open(c.done[0])" style="cursor: pointer">
                    <b :class="['gapv', gapTone(c.gap)]">{{ pct(c.gap) }}</b>
                    <span class="faint small"> · {{ t("容量", "cap") }} {{ pct(c.cap) }} · {{ t("隐私", "priv") }} {{ pct(c.priv) }}</span>
                    <span class="faint small" v-if="v.strategy === 'maskable'"> · EXIT {{ pct(c.exit) }}</span>
                    <span class="faint small" v-if="c.runs.length > 1"> · {{ c.done.length }} {{ t("种子", "seeds") }}</span>
                  </div>
                  <div v-else>
                  <span v-if="c.done.length" :class="{ click: true }" @click="open(c.done[0])" style="cursor: pointer"><b :class="['gapv', gapTone(c.gap)]">{{ pct(c.gap) }}</b>
                    <span class="faint small"> · {{ t("容量", "cap") }} {{ pct(c.cap) }} · {{ t("隐私", "priv") }} {{ pct(c.priv) }}<template v-if="v.strategy === 'maskable'"> · EXIT {{ pct(c.exit) }}</template> · {{ t("已完成种子", "seeds done") }} {{ c.done.length }}/{{ c.runs.length }}</span><br /></span>
                  <span class="badge" :class="STATUS_TONE[c.status]">{{ statusName(c.status) }}<template v-if="c.progress != null"> · {{ c.progress.toFixed(0) }}%</template></span>
                </div>
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
        <span class="faint small">{{ t("每个 RL_algorithm 各扫一次", "one sweep per RL_algorithm") }}<template v-if="b.params?.lambda_max_sweep?.steps ?? b.params?.steps"> · {{ steps(b.params?.lambda_max_sweep?.steps ?? b.params?.steps) }} {{ t("步", "steps") }}</template></span>
      </div>
      <div class="card-b">
        <div class="sw-grid">
          <div v-for="l in sweepLearners" :key="l">
            <div class="row" style="margin-bottom: 6px">
              <b>{{ l }} · lagrangian</b>
              <span v-if="chosen(l) != null && !Number.isNaN(chosen(l))" class="tag accent">{{ t("选定", "chosen") }} λ_max = {{ chosen(l) }}</span>
              <span v-else class="tag">{{ t("尚未选定", "not chosen yet") }}</span>
            </div>
            <div class="table-wrap"><table class="tbl sw">
              <thead><tr><th>λ_max</th><th v-for="s in scens" :key="s">{{ s.toUpperCase() }}</th><th v-if="sweepJson(l)" class="num">{{ t("平均无违约", "mean violation-free") }}</th><th v-if="sweepJson(l)" class="num">{{ t("平均有效 gap", "mean eff. gap") }}</th></tr></thead>
              <tbody>
                <tr v-for="v in sweepValues(l)" :key="v" :class="{ chosen: chosen(l) === v }">
                  <td><b>{{ v }}</b></td>
                  <td v-for="s in scens" :key="s">
                    <template v-for="c in [sweepCell(l, v, s)]" :key="s">
                      <div v-if="c && c.status === 'done'" style="cursor: pointer" @click="open(c.done[0])">
                        <b>{{ pct(effGap(l, v, s) ?? c.gap) }}</b>
                        <div class="faint small">{{ t("违约", "viol.") }} {{ pct(c.viol) }} · λ {{ num(c.lam, 2) }}</div>
                      </div>
                      <div v-else-if="c">
                        <span class="badge hide-sm" :class="STATUS_TONE[c.status]">{{ statusName(c.status) }}<template v-if="c.progress != null"> · {{ c.progress.toFixed(0) }}%</template></span>
                        <div v-if="c.live" class="faint small mono hide-sm">{{ steps(c.live.step) }} / {{ steps(c.live.total) }}<template v-if="c.live.lambda != null"> · λ {{ num(c.live.lambda, 2) }}</template></div>
                        <span class="show-sm-i small"><template v-if="c.progress != null"><b style="color: var(--run)">{{ c.progress.toFixed(0) }}%</b></template><template v-else>{{ statusName(c.status) }}</template><template v-if="c.live?.lambda != null"> · λ {{ num(c.live.lambda, 2) }}</template></span>
                      </div>
                    </template>
                  </td>
                  <td v-if="sweepJson(l)" class="num"><b>{{ pct(freeMean(l, v)) }}</b></td>
                  <td v-if="sweepJson(l)" class="num">{{ pct(effMean(l, v)) }}</td>
                </tr>
              </tbody>
            </table></div>
          </div>
        </div>
      </div>
      <div class="card-note">
        {{ t("λ 在训练中按局对偶上升自适应：λ ← clip(λ + η(J_C − d), 0, λ_max)，λ_max 只是上限。选定规则（违约优先）：三场景平均无违约比例最高；相差 0.5 个百分点以内视为持平，持平时选平均有效 gap 更低的（有效 gap：有违约的实例按 AR = 0 计）。选定值写入本批次 manifest，由正式实验批次读取（v5.0.3 → v5.0.4）。",
             "λ adapts during training by dual ascent per episode: λ ← clip(λ + η(J_C − d), 0, λ_max); λ_max is only its cap. Rule (violations first): highest mean violation-free share over the scenarios; within 0.5 pp is a tie, broken by the lower mean effective gap (a violating instance counts as AR = 0). The choice is written to this batch's manifest and read by the main batch (v5.0.3 → v5.0.4).") }}
      </div>
    </div>
  </div>
</template>

<style scoped>
.legend-row { margin-bottom: -6px; }
.mx-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(520px, 100%), 1fr)); gap: 16px; }
.sw-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(400px, 100%), 1fr)); gap: 20px; }
.mx-grid > *, .sw-grid > * { min-width: 0; }   /* a wide table scrolls inside its cell instead of widening the page */
@media (max-width: 860px) { .mx-grid, .sw-grid { grid-template-columns: 1fr; } }
@media (max-width: 700px) {
  table.mx { border-spacing: 4px; margin: -4px; }
  .mx th { font-size: 11px; }
  .mx th.rowh { font-size: 12px; }
  .mx-cell { min-width: 0; padding: 6px 6px; min-height: 52px; }
  .mx-gap { font-size: 15px; }
  .mx-sub { font-size: 10.5px; white-space: normal; }
  .mx .badge { font-size: 11px; padding: 0 6px; }
  .sw td, .sw th { padding: 6px 6px; }
  .sw .badge { font-size: 11px; padding: 0 6px; }
}
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
.vc { border-top: 1px solid var(--border); padding: 9px 0; }
.vc:first-child { border-top: 0; padding-top: 0; }
.vc-h { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 4px; }
.vc-h .tag { font-size: 11.5px; }
.vc-r { display: grid; grid-template-columns: 76px minmax(0, 1fr); gap: 6px; align-items: baseline; padding: 2px 0; }
</style>
