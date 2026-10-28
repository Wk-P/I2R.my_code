<script setup>
// Summary (/summary): rows = strategy, columns = RL_algorithm, each cell the gap averaged over LT / GT / EQ on the
// fixed v6.0.4 test set (paper_contents/v6.0.0-merge/table_2d.json, written by src/scripts/merge_2d_table.py).
// No per-scenario split on this page: a heatmap of the means, then the paper figures and table.
import { ref, computed, onMounted } from "vue";
import { t, tl } from "../i18n.js";
import { getSummary, summaryFile } from "../api.js";
import { pct } from "../format.js";

const d = ref(null);
const failed = ref(false);
const tex = ref("");
onMounted(async () => {
  d.value = await getSummary(); failed.value = !d.value;
  if (d.value?.files?.includes("table_main.tex")) tex.value = await (await fetch(summaryFile("table_main.tex", d.value.generated_at))).text();
});
const file = (f) => summaryFile(f, d.value?.generated_at);
// best / second best, as in table_main.tex
function mark(v) {
  const xs = d.value.cells.map((c) => c.mean).sort((a, b) => a - b);
  return v === xs[0] ? "best" : v === xs[1] ? "second" : "";
}
const pm = (c) => `${(100 * c.mean).toFixed(2)} ± ${(100 * c.mean_std).toFixed(2)}`;

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
      <div class="head-sub">{{ t("每格为 LT、GT、EQ 三个场景 gap 的算术平均；gap = 1 − mean(AR) / mean(ILP AR)，AR 只计合法放置。越低越好。",
        "Each cell is the plain mean of the LT, GT and EQ gaps; gap = 1 − mean(AR) / mean(ILP AR), AR over legal placements. Lower is better.") }}</div>
    </div></div>

    <div v-if="failed" class="card empty">{{ t("没有数据：先运行 src/scripts/merge_2d_table.py。", "No data: run src/scripts/merge_2d_table.py first.") }}</div>
    <template v-else-if="d">
      <div class="card">
        <div class="card-h"><h2>{{ t("AR gap", "AR gap") }}</h2>
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
                    :class="{ best: cell(st, rl).mean === best, target: cell(st, rl).mean < 0.05 }">
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
          "Only maskable has a zero violation rate; other strategies' gaps are not directly equivalent to maskable's (see the violation rate below).") }}
          {{ t("测试集", "Test set") }}: <code>{{ d.test_set }}</code>。{{ t("来源", "Sources") }}: {{ tl(d.sources) }}。
          <span class="faint">{{ t("生成于", "Generated") }} {{ d.generated_at }} · <code>{{ d.script }}</code></span></div>
      </div>

      <div class="card">
        <div class="card-h"><h2>{{ t("论文图表", "Paper figures") }}</h2>
          <span class="faint small">{{ t("与论文中使用的文件完全一致（矢量 PDF / LaTeX）；误差棒 = 种子间标准差", "Exactly the files used in the paper (vector PDF / LaTeX); error bars = std over seeds") }}</span></div>
        <div class="card-b">
          <div class="fig">
            <div class="fig-h"><b>{{ t("图 1　AR gap（分组柱状图）", "Fig. 1　AR gap (grouped bars)") }}</b>
              <a :href="file('fig_gap_bars.pdf')">PDF</a> · <a :href="file('fig_gap_bars.png')" target="_blank">PNG</a></div>
            <img :src="file('fig_gap_bars.png')" alt="fig_gap_bars" />
            <div class="small dim">{{ t("横轴为 strategy，组内三根柱为 PPO / DQN / Double DQN；误差棒 = 种子间标准差。",
              "Strategy on the x-axis, PPO / DQN / Double DQN within each group; error bars = std over seeds.") }}</div>
          </div>
          <div class="fig narrow">
            <div class="fig-h"><b>{{ t("图 2　gap–违约率（左下最好）", "Fig. 2　Gap vs violation rate (lower-left is better)") }}</b>
              <a :href="file('fig_gap_viol.pdf')">PDF</a> · <a :href="file('fig_gap_viol.png')" target="_blank">PNG</a></div>
            <img :src="file('fig_gap_viol.png')" alt="fig_gap_viol" />
            <div class="small dim">{{ t("颜色 = strategy，形状 = 算法；横向误差棒 = gap 的种子间标准差；内嵌图放大左下角。只有 Mask 落在违约率 0 上。",
              "Colour = strategy, marker = algorithm; horizontal bars = std over seeds of the gap; the inset zooms into the lower-left corner. Only Mask lies on zero violation.") }}</div>
          </div>
          <div class="fig">
            <div class="fig-h"><b>{{ t("表 1　主结果表（booktabs）", "Table 1　Main results (booktabs)") }}</b>
              <a :href="file('table_main.tex')">.tex</a></div>
            <div class="table-wrap">
              <table class="tbl paper">
                <thead><tr><th>Strategy</th><th>Algorithm</th><th class="num">AR gap (%)</th><th class="num">Viol. (%)</th></tr></thead>
                <tbody>
                  <template v-for="st in d.strategies" :key="st">
                    <tr v-for="(rl, i) in d.RL_algorithms" :key="rl" :class="{ first: i === 0 }">
                      <td>{{ i === 0 ? st : "" }}</td><td>{{ rl }}</td>
                      <td class="num" :class="mark(cell(st, rl).mean)">{{ pm(cell(st, rl)) }}</td>
                      <td class="num">{{ (100 * cell(st, rl).viol).toFixed(2) }}</td>
                    </tr>
                  </template>
                </tbody>
              </table>
            </div>
            <div class="small dim">{{ t("均值 ± 种子间标准差（%）；加粗为最好、下划线为次好。", "Mean ± std over seeds (%); bold = best, underlined = second best.") }}</div>
            <details><summary class="small">LaTeX</summary><pre class="tex">{{ tex }}</pre></details>
          </div>
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
.fig { margin-bottom: 22px; }
.fig img { width: 100%; max-width: 560px; display: block; margin: 8px 0; border: 1px solid var(--border); border-radius: 6px; background: #fff; }
.fig.narrow img { max-width: 560px; }
.fig-h { display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
.fig-h b { margin-right: 8px; }
.tbl.paper td.best { font-weight: 750; }
.tbl.paper td.second { text-decoration: underline; }
pre.tex { font-size: 11.5px; background: var(--panel-2); padding: 10px; border-radius: 6px; overflow-x: auto; }
</style>
