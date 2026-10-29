<script setup>
// Summary (/summary): rows = strategy, columns = RL_algorithm, each cell the gap averaged over LT / GT / EQ on the
// fixed v6.0.4 test set (paper_contents/v6.0.0-merge/table_2d.json, written by src/scripts/merge_2d_table.py).
// No per-scenario split for the gap: a heatmap of the means, then AR per scenario (ILP / PPO / DQN / DDQN), the mean
// training reward curves, the ILP-vs-variant timing (src/scripts/bench_time_v6.py), the Dinkelbach steps of the ILP
// baseline, then the paper figures and table.
import { ref, computed, onMounted } from "vue";
import Chart from "../components/Chart.vue";
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
// Okabe-Ito, as the paper figures (merge_2d_table.py ALGO_STYLE); ILP in grey
const COLOR = { ILP: "#7a7f8c", PPO: "#0072B2", DQN: "#E69F00", DDQN: "#009E73" };
const SCEN = { lt: "LT", gt: "GT", eq: "EQ" };
const arStrat = ref("maskable");
const arOption = computed(() => {
  const sc = d.value.scenarios, rls = d.value.RL_algorithms;
  const ilp = sc.map((s) => mean(rls.map((rl) => cell(arStrat.value, rl).ilp_ar[s])));    // same test set: equal for every cell
  const bar = (name, data) => ({ name, type: "bar", data: data.map((v) => +v.toFixed(4)), itemStyle: { color: COLOR[name] },
    label: { show: true, position: "top", fontSize: 10, formatter: (p) => p.value.toFixed(3) } });
  return { tooltip: { trigger: "axis", valueFormatter: (v) => v.toFixed(4) }, legend: { top: 0 },
    grid: { left: 50, right: 16, top: 36, bottom: 28 },
    xAxis: { type: "category", data: sc.map((s) => SCEN[s]) },
    yAxis: { type: "value", name: "AR", min: (v) => Math.max(0, Math.floor((v.min - 0.05) * 10) / 10), max: 1 },
    series: [bar("ILP", ilp), ...rls.map((rl) => bar(rl, sc.map((s) => cell(arStrat.value, rl).ar[s])))] };
});
const curveOption = (st) => ({
  tooltip: { trigger: "axis", valueFormatter: (v) => (v == null ? "—" : v.toFixed(3)) }, legend: { top: 0 },
  grid: { left: 50, right: 16, top: 30, bottom: 30 },
  xAxis: { type: "value", min: 0, max: d.value.curve_steps.at(-1) + d.value.curve_steps[0],
    axisLabel: { formatter: (v) => `${v / 1e6}M` } },
  yAxis: { type: "value", name: "reward", scale: true },
  series: d.value.RL_algorithms.map((rl) => ({ name: rl, type: "line", showSymbol: false, itemStyle: { color: COLOR[rl] },
    data: cell(st, rl).reward_curve.map((y, i) => [d.value.curve_steps[i], y]) })),
});
const tb = computed(() => d.value?.time_bench);
const tmean = (o) => mean(d.value.scenarios.map((s) => o[s]));
const ilpMs = (s) => tb.value.ilp[s].mean_ms;
const timeRows = computed(() => d.value.strategies.flatMap((st) => d.value.RL_algorithms.map((rl) => {
  const v = tb.value.rl[`${st}|${rl}`];
  return { st, rl, v, mean: tmean(v), speed: mean(d.value.scenarios.map((s) => ilpMs(s) / v[s])) };
})));
const timeOption = computed(() => {
  const rows = [{ name: "ILP", mean: tmean(Object.fromEntries(d.value.scenarios.map((s) => [s, ilpMs(s)]))), rl: "ILP" },
    ...timeRows.value.map((r) => ({ name: `${r.st} ${r.rl}`, mean: r.mean, rl: r.rl }))];
  return { tooltip: { trigger: "axis", valueFormatter: (v) => `${v.toFixed(2)} ms` },
    grid: { left: 140, right: 60, top: 10, bottom: 30 },
    xAxis: { type: "log", name: "ms", logBase: 10 },
    yAxis: { type: "category", inverse: true, data: rows.map((r) => r.name) },
    series: [{ type: "bar", data: rows.map((r) => ({ value: +r.mean.toFixed(3), itemStyle: { color: COLOR[r.rl] } })),
      label: { show: true, position: "right", fontSize: 10, formatter: (p) => `${p.value >= 10 ? p.value.toFixed(0) : p.value.toFixed(2)}` } }] };
});
const ms = (v) => (v >= 10 ? v.toFixed(1) : v.toFixed(2));

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
      <div class="head-sub small">{{ t("算法：PPO；DQN；DDQN = Double DQN（双重 DQN：在线网络选出下一状态的动作、目标网络给该动作估值，以减小 DQN 对 Q 值的高估）。",
        "Algorithms: PPO; DQN; DDQN = Double DQN (the online network picks the next-state action and the target network evaluates it, reducing DQN's Q-value overestimation).") }}</div>
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
        <div class="card-h"><h2>{{ t("AR 对比：ILP / PPO / DQN / DDQN", "AR: ILP / PPO / DQN / DDQN") }}</h2>
          <div class="seg"><button v-for="st in d.strategies" :key="st" :class="{ on: arStrat === st }" @click="arStrat = st">{{ st }}</button></div></div>
        <div class="card-b"><Chart :option="arOption" :height="320" /></div>
        <div class="card-note">{{ t("每个场景 5000 个测试实例的平均 AR，再对种子取平均；AR 只计合法放置；ILP = Dinkelbach 求得的最优 AR。DDQN = Double DQN。",
          "Mean AR over the 5000 test instances per scenario, then over seeds; AR over legal placements; ILP = the optimal AR from Dinkelbach. DDQN = Double DQN.") }}</div>
      </div>

      <div class="card">
        <div class="card-h"><h2>{{ t("训练 reward 曲线", "Training reward curves") }}</h2>
          <span class="faint small">{{ t("每个 strategy 一张图；横轴为训练步数 [0, 总步数]", "One chart per strategy; x = training steps [0, total]") }}</span></div>
        <div class="card-b grid-2">
          <div v-for="st in d.strategies" :key="st"><b class="small">{{ st }}</b><Chart :option="curveOption(st)" :height="240" /></div>
        </div>
        <div class="card-note">{{ t(`每个 run 的 episode reward 按步数等分为 ${d.curve_steps.length} 段取均值，再对种子与 LT/GT/EQ 三个场景取平均。各 strategy 的 reward 定义不同（lagrangian 每步减 λc），不同图之间不能直接比较数值。`,
          `Each run's episode reward averaged in ${d.curve_steps.length} equal step bins, then over seeds and the LT / GT / EQ scenarios. The reward differs by strategy (lagrangian subtracts λc each step), so values are not comparable across charts.`) }}</div>
      </div>

      <div class="card">
        <div class="card-h"><h2>{{ t("耗时对比：ILP vs 12 个变种", "Time: ILP vs the 12 variants") }}</h2>
          <span class="faint small">{{ t("单实例平均耗时（ms，对数轴），CPU 单线程", "Mean time per instance (ms, log axis), one CPU thread") }}</span></div>
        <div v-if="!tb" class="card-b dim">{{ t("还没有计时数据：运行 src/scripts/bench_time_v6.py，再运行 merge_2d_table.py。",
          "No timing yet: run src/scripts/bench_time_v6.py, then merge_2d_table.py.") }}</div>
        <template v-else>
          <div class="card-b grid-2">
            <Chart :option="timeOption" :height="420" />
            <div class="table-wrap">
              <table class="tbl">
                <thead><tr><th>strategy</th><th>RL_algorithm</th><th v-for="s in d.scenarios" :key="s" class="num">{{ SCEN[s] }}</th>
                  <th class="num">{{ t("平均", "Mean") }}</th><th class="num">{{ t("加速比", "Speed-up") }}</th></tr></thead>
                <tbody>
                  <tr><td colspan="2"><b>ILP</b> <span class="faint small">Dinkelbach</span></td>
                    <td v-for="s in d.scenarios" :key="s" class="num">{{ ms(ilpMs(s)) }}</td>
                    <td class="num">{{ ms(mean(d.scenarios.map(ilpMs))) }}</td><td class="num">1×</td></tr>
                  <tr v-for="r in timeRows" :key="r.st + r.rl"><td>{{ r.st }}</td><td>{{ r.rl }}</td>
                    <td v-for="s in d.scenarios" :key="s" class="num">{{ ms(r.v[s]) }}</td>
                    <td class="num">{{ ms(r.mean) }}</td><td class="num">{{ r.speed.toFixed(0) }}×</td></tr>
                </tbody>
              </table>
            </div>
          </div>
          <div class="card-note">{{ t(`v6.0.4 测试集每场景前 ${tb.n_inst} 个实例，所有方法相同。RL：每格种子 ${tb.seed} 模型的确定性完整 episode（reset + 掩码 + 前向 + step）；ILP：solve_ilp_max_ar（CBC 单线程，含建模与全部 Dinkelbach 迭代，平均 ${mean(d.scenarios.map((s) => tb.ilp[s].iters)).toFixed(2)} 次）。加速比 = 三场景 ILP 平均 / 变种平均 的平均。`,
            `First ${tb.n_inst} v6.0.4 test instances per scenario, the same for every method. RL: deterministic full episode of each cell's seed-${tb.seed} model (reset + mask + forward + step); ILP: solve_ilp_max_ar (CBC single thread, model building and every Dinkelbach iteration, ${mean(d.scenarios.map((s) => tb.ilp[s].iters)).toFixed(2)} on average). Speed-up = mean over scenarios of ILP mean / variant mean.`) }}
            <span class="faint">{{ t("生成于", "Generated") }} {{ tb.generated_at }} · <code>{{ tb.script }}</code></span></div>
        </template>
      </div>

      <div class="card">
        <div class="card-h"><h2>{{ t("ILP 基准：Dinkelbach 更新步骤", "ILP baseline: Dinkelbach update steps") }}</h2>
          <span class="faint small"><code>shared/ilp_utils.py: solve_ilp_max_ar</code></span></div>
        <div class="card-b dk">
          <p>{{ t("AR 是分式目标：", "AR is a fractional objective:") }}
            <span class="f">AR(x, y) = F(x) / G(y)，F(x) = Σ<sub>m,n</sub> x<sub>mn</sub> e<sub>m</sub> / c<sub>n</sub>，G(y) = Σ<sub>n</sub> y<sub>n</sub></span>。
            {{ t("Dinkelbach 把它化为一串参数化 ILP：", "Dinkelbach turns it into a sequence of parametric ILPs:") }}</p>
          <ol>
            <li>{{ t("初始化", "Initialise") }} <span class="f">ρ<sub>0</sub> = 0，k = 0</span>。</li>
            <li>{{ t("求解参数化 ILP", "Solve the parametric ILP") }}
              <div class="f blk">(x<sup>k</sup>, y<sup>k</sup>) = argmax F(x) − ρ<sub>k</sub> G(y)<br>
                s.t. Σ<sub>n</sub> x<sub>mn</sub> = 1 ∀m；Σ<sub>m</sub> x<sub>mn</sub> e<sub>m</sub> ≤ c<sub>n</sub> y<sub>n</sub> ∀n；x<sub>mn</sub> ≤ y<sub>n</sub>；
                Σ<sub>m∈C<sub>j</sub></sub> x<sub>mn</sub> ≤ 1 ∀j, n；x, y ∈ {0, 1}</div></li>
            <li>{{ t("更新参数", "Update the parameter") }} <span class="f">ρ<sub>k+1</sub> = F(x<sup>k</sup>) / G(y<sup>k</sup>)</span>
              {{ t("（即当前解的 AR）。", "(the AR of the current solution).") }}</li>
            <li>{{ t("若", "If") }} <span class="f">|ρ<sub>k+1</sub> − ρ<sub>k</sub>| &lt; 10<sup>−6</sup></span>
              {{ t("则停止，ILP AR = ρ", "stop: ILP AR = ρ") }}<sub>k+1</sub>{{ t("；否则 k ← k + 1 回到第 2 步（最多 30 次）。", "; otherwise k ← k + 1 and go to step 2 (at most 30 iterations).") }}</li>
          </ol>
          <p class="small dim">{{ t("依据：max F − ρG 的最优值 = 0 当且仅当 ρ 等于最大 AR；ρ 单调不减，有限步收敛到全局最优。C_j 为隐私冲突集。",
            "Why: the optimum of max F − ρG is 0 iff ρ equals the maximum AR; ρ is non-decreasing and reaches the global optimum in finitely many steps. C_j are the privacy conflict sets.") }}</p>
        </div>
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
.dk p, .dk li { line-height: 1.8; }
.dk ol { padding-left: 22px; margin: 6px 0; }
.dk .f { font-family: "Cambria Math", "STIX Two Math", serif; font-style: italic; }
.dk .f.blk { display: block; background: var(--panel-2); border-radius: 6px; padding: 8px 12px; margin: 4px 0; }
pre.tex { font-size: 11.5px; background: var(--panel-2); padding: 10px; border-radius: 6px; overflow-x: auto; }
</style>
