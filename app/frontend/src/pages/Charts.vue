<script setup>
// Charts (v5.0.0): every chart of the version reports on one page, each answering one question.
import { ref, computed, onMounted, watch } from "vue";
import { getReports, getCurves } from "../api.js";
import { langRef as curLang } from "../i18n.js";
import GapTrend from "../components/GapTrend.vue";
import LearningCurves from "../components/charts/LearningCurves.vue";
import VersionBars from "../components/charts/VersionBars.vue";
import BaselineDumbbell from "../components/charts/BaselineDumbbell.vue";
import { fmt1 } from "../components/charts/common.js";

const L = (zh, en) => (curLang.value === "en" ? en : zh);
const reports = ref([]);
const curves = ref([]);
async function load() {
  reports.value = (await getReports()).reports;
  curves.value = (await getCurves(curLang.value)).versions;
}
onMounted(load);
watch(curLang, async () => { curves.value = (await getCurves(curLang.value)).versions; });

// headline tiles: best mean so far, and the latest version
const withGap = computed(() => reports.value.filter((r) => r.new_avg != null));
const best = computed(() => withGap.value.slice().sort((a, b) => a.new_avg - b.new_avg)[0] ?? null);
const latest = computed(() => withGap.value[0] ?? null);
const bestScen = computed(() => best.value?.gaps?.[best.value.new_model] ?? {});
</script>

<template>
  <div class="page-head" data-no-i18n>
    <h1>{{ L("图表", "Charts") }}</h1>
    <div class="page-sub">{{ L("所有版本报告里的图，一张图回答一个问题；数据来自 paper_contents/v*/report.md，报告更新后刷新即可",
      "Every chart from the version reports, one question per chart; data from paper_contents/v*/report.md — refresh after a report changes") }}</div>
  </div>

  <div class="kpis" data-no-i18n>
    <div class="kpi"><div class="kpi-label">{{ L("最好的三场景平均 gap", "Best 3-scenario mean gap") }}</div>
      <div class="kpi-value">{{ best ? fmt1(best.new_avg) : "—" }}</div>
      <div class="kpi-foot" v-if="best"><a :href="`#/versions/${best.version}`">{{ best.version }}</a> · LT {{ fmt1(bestScen.LT) }} · EQ {{ fmt1(bestScen.EQ) }} · GT {{ fmt1(bestScen.GT) }}</div></div>
    <div class="kpi"><div class="kpi-label">{{ L("目标", "Goal") }}</div><div class="kpi-value">5.0%</div>
      <div class="kpi-foot" v-if="best">{{ best.new_avg <= 5 ? L("已达到", "reached") : L(`还差 ${(best.new_avg - 5).toFixed(1)} 个百分点`, `${(best.new_avg - 5).toFixed(1)} points to go`) }}</div></div>
    <div class="kpi"><div class="kpi-label">{{ L("最新的版本报告", "Latest version report") }}</div>
      <div class="kpi-value kpi-value--sm">{{ latest?.version ?? "—" }}</div>
      <div class="kpi-foot" v-if="latest">{{ L("新变体平均", "new variant mean") }} {{ fmt1(latest.new_avg) }}</div></div>
    <div class="kpi"><div class="kpi-label">{{ L("有学习曲线的版本", "Versions with a learning curve") }}</div>
      <div class="kpi-value">{{ curves.length }}</div><div class="kpi-foot">{{ curves.map((v) => v.version).slice(0, 4).join(" · ") }}</div></div>
  </div>

  <section class="panel" data-no-i18n>
    <div class="panel-head"><h2>{{ L("1. 版本走势：每个版本的新变体，测试集相对 gap", "1. Across versions: the new variant's test relative gap") }}</h2></div>
    <GapTrend :reports="reports" />
  </section>

  <section class="panel" data-no-i18n>
    <div class="panel-head"><h2>{{ L("2. 学习曲线：gap 随训练步数怎么下降", "2. Learning curves: how the gap falls with training steps") }}</h2></div>
    <LearningCurves :versions="curves" />
  </section>

  <section class="panel" data-no-i18n>
    <div class="panel-head"><h2>{{ L("3. 单个版本：报告里每个模型的三场景 gap", "3. One version: every model's gap per scenario") }}</h2></div>
    <VersionBars :reports="reports" />
  </section>

  <section class="panel" data-no-i18n>
    <div class="panel-head"><h2>{{ L("4. 每个版本相对自己的基线：分场景看改进", "4. Each version against its own baseline, per scenario") }}</h2></div>
    <BaselineDumbbell :reports="reports" />
  </section>
</template>
