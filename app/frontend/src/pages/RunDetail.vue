<script setup>
import { ref, computed, watch } from "vue";
import { getExperiments, resultFile } from "../api.js";
import { setViewBranch } from "../store.js";
import { algoLabel, SCENARIO_LABEL, zeroRepairViol } from "../labels.js";
import { fmt, pct, shortTime } from "../format.js";

const props = defineProps({
  branch: { type: String, required: true },
  scenario: { type: String, required: true },
  algo: { type: String, required: true },
  run: { type: String, required: true },
});

const row = ref(null);
const loaded = ref(false);
const siblings = ref([]);   // other seeds of the same batch / scenario / algo / variant

watch(() => [props.branch, props.scenario, props.algo, props.run], async () => {
  loaded.value = false;
  setViewBranch(props.branch);
  const all = (await getExperiments(props.branch)).map(zeroRepairViol);
  row.value = all.find((r) => r.scenario === props.scenario && r.algo === props.algo && r.run === props.run) ?? null;
  siblings.value = row.value?.batch
    ? all.filter((r) => r.batch === row.value.batch && r.scenario === props.scenario && r.algo === props.algo &&
        r.variant === row.value.variant).sort((a, b) => (a.seed ?? 0) - (b.seed ?? 0))
    : [];
  loaded.value = true;
}, { immediate: true });

const q = (o) => new URLSearchParams(Object.entries(o).filter(([, v]) => v)).toString();
const versionLink = computed(() => `#/results?${q({ branch: props.branch, version: row.value?.version })}`);
const algoLink = computed(() => `#/results?${q({ branch: props.branch, version: row.value?.version, batch: row.value?.batch, scen: props.scenario, algo: props.algo })}`);
const gap = computed(() => (row.value?.ilp_ar == null || row.value?.test_ar_mean == null ? null : row.value.ilp_ar - row.value.test_ar_mean));
const img = (file) => resultFile(props.branch, props.scenario, props.algo, props.run, file);
</script>

<template>
  <div class="crumbs">
    <a href="#/results">实验结果</a> /
    <a :href="`#/results?branch=${encodeURIComponent(branch)}`">{{ branch }}</a> /
    <template v-if="row?.version"><a :href="versionLink">{{ row.version }}</a> / </template>
    <template v-if="row?.batch"><a :href="`#/batches/${row.batch}`">{{ row.batch }}</a> / </template>
    {{ scenario.toUpperCase() }} /
    <a :href="algoLink">{{ algoLabel(algo) }}</a> /
    {{ run }}
  </div>

  <div v-if="!loaded" class="empty">加载中…</div>
  <div v-else-if="!row" class="empty">找不到该运行（{{ branch }}/{{ scenario }}/{{ algo }}/{{ run }}）</div>
  <template v-else>
    <div class="page-head">
      <h1>{{ algoLabel(algo) }} · {{ SCENARIO_LABEL[scenario] ?? scenario }}
        <span v-if="row.variant" class="tag tag--lg">{{ row.variant }}</span>
        <span v-if="row.seed != null" class="tag tag--lg">种子 {{ row.seed }}</span>
      </h1>
      <div class="page-sub">exp_id {{ row.exp_id }} · 完成于 {{ shortTime(row.created_at) }} · {{ row.train_steps?.toLocaleString() ?? "—" }} 步 · N/M = {{ row.N }}/{{ row.M }}</div>
    </div>

    <div class="kpis">
      <div class="kpi"><div class="kpi-label" title="success_rate = 测试实例中 M 个服务全部合法放置且无违规的比例">success_rate（测试集）</div><div class="kpi-value">{{ pct(row.test_success_rate) }}</div></div>
      <div class="kpi"><div class="kpi-label" title="AR = average resource utilization（平均资源利用率，优化目标）">AR（average resource utilization，测试集）</div><div class="kpi-value">{{ fmt(row.test_ar_mean) }}</div><div class="kpi-foot">± {{ fmt(row.test_ar_std, 3) }}</div></div>
      <div class="kpi"><div class="kpi-label">ILP AR / AR gap</div><div class="kpi-value kpi-value--sm">{{ fmt(row.ilp_ar) }} / {{ fmt(gap) }}</div></div>
      <div class="kpi"><div class="kpi-label">容量 / 冲突违规</div><div class="kpi-value kpi-value--sm">{{ pct(row.test_cap_viol_rate) }} / {{ pct(row.test_conflict_viol_rate) }}</div>
        <div class="kpi-foot">次数 {{ row.test_cap_viol_total ?? "—" }} / {{ row.test_conflict_viol_total ?? "—" }}</div></div>
    </div>

    <div class="two-col two-col--wide">
      <section class="panel">
        <div class="panel-head"><h2>基本信息</h2></div>
        <table class="kv">
          <tr><th>分支</th><td>{{ branch }}</td></tr>
          <tr><th>版本 / 批次</th><td>{{ row.version ?? "—" }} / {{ row.batch ?? "—" }}</td></tr>
          <tr><th>训练 / 测试实例数</th><td>{{ row.train_count ?? "—" }} / {{ row.test_count ?? "—" }}</td></tr>
          <tr><th>训练回合数</th><td>{{ row.train_episodes?.toLocaleString() ?? "—" }}</td></tr>
          <tr><th>训练末 50 回合 AR</th><td>{{ fmt(row.train_ar_last50) }}</td></tr>
          <tr><th>结果目录</th><td class="mono small">results/{{ branch }}/{{ scenario }}/{{ algo }}/{{ run }}/</td></tr>
        </table>
      </section>
      <section v-if="siblings.length > 1" class="panel">
        <div class="panel-head"><h2>同组其他种子</h2></div>
        <table class="grid">
          <thead><tr><th>种子</th><th>exp_id</th><th class="num" title="success_rate = 测试实例中 M 个服务全部合法放置且无违规的比例">success_rate</th><th class="num" title="AR = average resource utilization（平均资源利用率，优化目标）">AR</th></tr></thead>
          <tbody>
            <tr v-for="s in siblings" :key="s.run" :class="{ current: s.run === run }">
              <td>{{ s.seed ?? "—" }}</td>
              <td><a :href="`#/run/${branch}/${scenario}/${algo}/${s.run}`" class="mono">{{ s.run }}</a></td>
              <td class="num">{{ pct(s.test_success_rate) }}</td>
              <td class="num">{{ fmt(s.test_ar_mean) }}</td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>

    <section class="panel">
      <div class="panel-head"><h2>训练曲线</h2></div>
      <img class="figure" :src="img('training_curve.png')" alt="training_curve.png 不存在" />
    </section>
    <section class="panel">
      <div class="panel-head"><h2>测试对比</h2></div>
      <img class="figure" :src="img('comparison.png')" alt="comparison.png 不存在" />
    </section>
  </template>
</template>
