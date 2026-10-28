<script setup>
// Summary (/summary): rows = strategy, columns = RL_algorithm, each cell the gap averaged over LT / GT / EQ on the
// fixed v6.0.4 test set (paper_contents/v6.0.0-merge/table_2d.json, written by src/scripts/merge_2d_table.py).
// A heatmap of the means, then the per-scenario breakdown with the violation rate.
import { ref, computed, onMounted } from "vue";
import { t, tl } from "../i18n.js";
import { getSummary } from "../api.js";
import { pct } from "../format.js";
import { scenName } from "../labels.js";

const d = ref(null);
const failed = ref(false);
onMounted(async () => { d.value = await getSummary(); failed.value = !d.value; });

const mean = (xs) => xs.reduce((a, c) => a + c, 0) / xs.length;
const cell = (st, rl) => d.value.cells.find((c) => c.strategy === st && c.RL_algorithm === rl);
const rowMean = (st) => mean(d.value.RL_algorithms.map((rl) => cell(st, rl).mean));
const colMean = (rl) => mean(d.value.strategies.map((st) => cell(st, rl).mean));
const all = computed(() => (d.value ? mean(d.value.cells.map((c) => c.mean)) : null));
const best = computed(() => (d.value ? Math.min(...d.value.cells.map((c) => c.mean)) : null));
const range = computed(() => (d.value ? [best.value, Math.max(...d.value.cells.map((c) => c.mean))] : [0, 1]));
// white -> accent by gap (log scale so the low cells stay apart)
function heat(v) {
  const [lo, hi] = range.value.map(Math.log);
  const k = hi > lo ? (Math.log(v) - lo) / (hi - lo) : 0;
  const mix = (a, b) => Math.round(a + (b - a) * k);
  return { background: `rgb(${mix(238, 49)},${mix(240, 46)},${mix(255, 129)})`, color: k > 0.55 ? "#fff" : "var(--text)" };
}
</script>

<template>
  <div class="page">
    <div class="head"><div>
      <h1>{{ t("总图：strategy × RL_algorithm", "Summary: strategy × RL_algorithm") }}</h1>
      <div class="head-sub">{{ t("每格为 LT、GT、EQ 三个场景 gap 的算术平均（每场景先对种子取均值）；gap = 1 − mean(AR) / mean(ILP AR)，AR 只计合法放置。越低越好。",
        "Each cell is the plain mean of the LT, GT and EQ gaps (each scenario averaged over seeds first); gap = 1 − mean(AR) / mean(ILP AR), AR over legal placements. Lower is better.") }}</div>
    </div></div>

    <div v-if="failed" class="card empty">{{ t("没有数据：先运行 src/scripts/merge_2d_table.py。", "No data: run src/scripts/merge_2d_table.py first.") }}</div>
    <template v-else-if="d">
      <div class="card">
        <div class="card-h"><h2>{{ t("三场景平均 gap", "Gap averaged over LT / GT / EQ") }}</h2>
          <span class="faint small">{{ t("加粗为全表最低；虚线框为低于 5% 目标", "bold = lowest; dashed outline = below the 5% target") }}</span></div>
        <div class="card-b table-wrap">
          <table class="heat">
            <thead><tr>
              <th class="corner">strategy \ RL_algorithm</th>
              <th v-for="rl in d.RL_algorithms" :key="rl">{{ rl }}</th>
              <th class="m">{{ t("行平均", "Row mean") }}</th>
            </tr></thead>
            <tbody>
              <tr v-for="st in d.strategies" :key="st">
                <th class="rowh">{{ st }}</th>
                <td v-for="rl in d.RL_algorithms" :key="rl" :style="heat(cell(st, rl).mean)"
                    :class="{ best: cell(st, rl).mean === best, target: cell(st, rl).mean < 0.05 }"
                    :title="d.scenarios.map((s) => `${s.toUpperCase()} ${pct(cell(st, rl).gap[s], 2)}`).join(' · ')">
                  {{ pct(cell(st, rl).mean, 2) }}
                </td>
                <td class="m">{{ pct(rowMean(st), 2) }}</td>
              </tr>
              <tr>
                <th class="rowh m">{{ t("列平均", "Column mean") }}</th>
                <td v-for="rl in d.RL_algorithms" :key="rl" class="m">{{ pct(colMean(rl), 2) }}</td>
                <td class="m"><b>{{ pct(all, 2) }}</b></td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="card-note">{{ t("只有 maskable 的违约率为 0；其余 strategy 的 gap 与 maskable 不能直接等价比较（见下表违约率）。",
          "Only maskable has a zero violation rate; other strategies' gaps are not directly equivalent to maskable's (see the violation rate below).") }}</div>
      </div>

      <div class="card">
        <div class="card-h"><h2>{{ t("分场景明细", "Per-scenario breakdown") }}</h2></div>
        <div class="card-b table-wrap">
          <table class="tbl">
            <thead><tr>
              <th>strategy</th><th>RL_algorithm</th><th>{{ t("种子", "Seeds") }}</th>
              <th v-for="s in d.scenarios" :key="s" class="num">{{ scenName(s) }}</th>
              <th class="num">{{ t("三场景平均", "Mean") }}</th>
              <th class="num">{{ t("违约率（容量或隐私）", "Violation (cap or privacy)") }}</th>
              <th class="num">EXIT</th>
            </tr></thead>
            <tbody>
              <template v-for="st in d.strategies" :key="st">
                <tr v-for="(rl, i) in d.RL_algorithms" :key="rl" :class="{ first: i === 0 }">
                  <td><b v-if="i === 0">{{ st }}</b></td><td>{{ rl }}</td>
                  <td class="mono small">{{ cell(st, rl).seeds.join(",") }}</td>
                  <td v-for="s in d.scenarios" :key="s" class="num">{{ pct(cell(st, rl).gap[s], 2) }}</td>
                  <td class="num"><b>{{ pct(cell(st, rl).mean, 2) }}</b></td>
                  <td class="num" :class="{ bad: cell(st, rl).viol > 0.005 }">{{ pct(cell(st, rl).viol, 2) }}</td>
                  <td class="num">{{ st === "maskable" ? pct(cell(st, rl).exit, 2) : "" }}</td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>
        <div class="card-note">
          {{ t("违约率 = 1 − 无违约比例 − EXIT 比例，三场景、种子平均。", "Violation rate = 1 − violation-free − EXIT, averaged over scenarios and seeds.") }}
          {{ t("测试集", "Test set") }}: <code>{{ d.test_set }}</code>（{{ d.n_test }} / {{ t("场景", "scenario") }}）。
          {{ t("来源", "Sources") }}: {{ tl(d.sources) }}。
          <span class="faint">{{ t("生成于", "Generated") }} {{ d.generated_at }} · <code>{{ d.script }}</code></span>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.heat { border-collapse: separate; border-spacing: 4px; margin: 0 auto; }
.heat th { font-weight: 650; color: var(--dim); padding: 6px 10px; font-size: 13px; }
.heat th.corner { font-weight: 500; color: var(--faint); font-size: 12px; text-align: left; }
.heat th.rowh { text-align: right; }
.heat td { min-width: 120px; height: 56px; text-align: center; border-radius: 8px; font-size: 16px;
  font-variant-numeric: tabular-nums; }
.heat td.best { font-weight: 800; }
.heat td.target { outline: 2px dashed var(--ok); outline-offset: -4px; }
.heat .m { background: var(--panel-2); color: var(--dim); font-size: 14px; }
.tbl tr.first td { border-top: 1px solid var(--border-strong); }
.bad { color: var(--bad); }
</style>
