<script setup>
// Running runs with their live progress (from each run's own training log, /api/v2 run.live):
// step / total, steps per second, lambda (lagrangian), success rate of the last 500 episodes, episodes, ETA.
import { ref, computed } from "vue";
import { useRouter } from "vue-router";
import { t } from "../i18n.js";
import { steps, dur, num, pct } from "../format.js";
import Th from "./Th.vue";
import { useSort } from "../sort.js";

// collapse (phones only): show a summary per RL_algorithm and the first `collapse` cards, the rest behind a button
const props = defineProps({ runs: { type: Array, default: () => [] }, batch: { type: Boolean, default: false }, params: { type: Object, default: () => ({}) },
  collapse: { type: Number, default: 0 } });
const expanded = ref(false);
const SCEN = { lt: 0, eq: 1, gt: 2 }, RLI = { PPO: 0, DQN: 1, DDQN: 2 };
const sort = useSort(computed(() => props.runs), {
  batch: (r) => r.batch, scen: (r) => SCEN[r.scenario], rl: (r) => RLI[r.RL_algorithm], st: (r) => r.strategy, params: (r) => paramText(r.params),
  seed: (r) => r.seed, steps: (r) => r.live?.pct, sps: (r) => r.live?.steps_per_sec, lam: (r) => r.live?.lambda,
  succ: (r) => r.live?.success_last500, eps: (r) => r.live?.episodes, eta: (r) => r.live?.eta_seconds,
});
const shown = computed(() => (props.collapse && !expanded.value ? props.runs.slice(0, props.collapse) : props.runs));
const groups = computed(() => {
  const m = new Map();
  for (const r of props.runs) {
    const g = m.get(r.RL_algorithm) ?? { rl: r.RL_algorithm, n: 0, pct: 0, sps: 0, eta: 0 };
    g.n += 1; g.pct += r.live?.pct ?? 0; g.sps += r.live?.steps_per_sec ?? 0; g.eta = Math.max(g.eta, r.live?.eta_seconds ?? 0);
    m.set(r.RL_algorithm, g);
  }
  return [...m.values()];
});
const router = useRouter();
const open = (r) => r.space && r.exp_id && router.push(`/runs/${r.space}/${r.scenario}/${r.dir}/${r.exp_id}`);
const paramText = (p) => Object.entries(p || {}).filter(([k, x]) => x != null && JSON.stringify(x) !== JSON.stringify(props.params?.[k]))
  .map(([k, x]) => `${k}=${k === "steps" ? steps(x) : x}`).join(" · ");
const stale = (r) => r.live?.updated && Date.now() / 1000 - r.live.updated > 600;
</script>

<template>
  <!-- phones: one card per run -->
  <div class="show-sm">
    <div v-if="collapse && runs.length > collapse" class="grp">
      <div v-for="g in groups" :key="g.rl" class="grp-r">
        <b>{{ g.rl }}</b><span>{{ g.n }} {{ t("个", "runs") }}</span><span>{{ t("平均", "avg") }} {{ (g.pct / g.n).toFixed(0) }}%</span>
        <span>{{ g.sps.toLocaleString() }} {{ t("步/秒", "steps/s") }}</span><span>{{ t("最晚", "last") }} {{ dur(g.eta) }}</span>
      </div>
    </div>
    <div v-for="r in shown" :key="'c' + r.exp_id" class="lr" @click="open(r)">
      <div class="lr-h">
        <b>{{ r.scenario.toUpperCase() }} · {{ r.RL_algorithm }}</b><span class="dim">{{ r.strategy }}</span>
        <span v-if="paramText(r.params)" class="tag">{{ paramText(r.params) }}</span>
        <router-link v-if="batch" class="small" :to="`/batches/${r.batch}`" @click.stop>{{ r.batch }}</router-link>
      </div>
      <div class="row" style="flex-wrap: nowrap; gap: 8px; margin: 6px 0">
        <div class="bar" style="flex: 1"><span class="s-running" :style="{ width: (r.live?.pct ?? 0) + '%' }" /></div>
        <span class="small mono">{{ steps(r.live?.step) }}/{{ steps(r.live?.total) }}</span>
        <b class="small">{{ r.live?.pct?.toFixed(0) ?? "—" }}%</b>
      </div>
      <div class="lr-s">
        <span><i>{{ t("步/秒", "steps/s") }}</i>{{ r.live?.steps_per_sec ?? "—" }}</span>
        <span v-if="r.live?.lambda != null"><i>λ</i>{{ num(r.live.lambda, 3) }}</span>
        <span><i>{{ t("近500局", "last 500") }}</i>{{ pct(r.live?.success_last500) }}</span>
        <span><i>{{ t("剩余", "ETA") }}</i>{{ r.live?.eta_seconds != null ? dur(r.live.eta_seconds) : "—" }}</span>
      </div>
      <div v-if="stale(r)" class="small" style="color: var(--warn)">{{ t("日志超过 10 分钟没更新", "log not updated for 10+ min") }}</div>
    </div>
    <button v-if="collapse && runs.length > collapse" class="btn more" @click.stop="expanded = !expanded">
      {{ expanded ? t("收起", "Show fewer") : t(`展开全部 ${runs.length} 个 run`, `Show all ${runs.length} runs`) }}
    </button>
  </div>
  <div class="table-wrap hide-sm">
    <table class="tbl">
      <thead>
        <tr>
          <Th v-if="batch" :s="sort" k="batch">{{ t("批次", "Batch") }}</Th>
          <Th :s="sort" k="scen">{{ t("场景", "Scen.") }}</Th><Th :s="sort" k="rl">RL_algorithm</Th><Th :s="sort" k="st">strategy</Th><Th :s="sort" k="params">params</Th><Th :s="sort" k="seed">seed</Th>
          <Th :s="sort" k="steps" style="min-width: 220px">{{ t("步数", "Steps") }}</Th>
          <Th :s="sort" k="sps" num>{{ t("步/秒", "Steps/s") }}</Th>
          <Th :s="sort" k="lam" num :title="t('训练日志里当前的 λ（lagrangian）', 'Current λ in the training log (lagrangian)')">λ</Th>
          <Th :s="sort" k="succ" num :title="t('最近 500 局无违约完成的比例（训练策略）', 'Share of the last 500 training episodes completed without violation')">{{ t("近 500 局完成率", "Success (last 500)") }}</Th>
          <Th :s="sort" k="eps" num>{{ t("局数", "Episodes") }}</Th>
          <Th :s="sort" k="eta" num>{{ t("预计剩余", "ETA") }}</Th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="r in sort.sorted.value" :key="r.exp_id" class="link" @click="open(r)">
          <td v-if="batch"><router-link :to="`/batches/${r.batch}`" @click.stop>{{ r.batch }}</router-link></td>
          <td>{{ r.scenario.toUpperCase() }}</td><td><b>{{ r.RL_algorithm }}</b></td><td>{{ r.strategy }}</td>
          <td class="small">{{ paramText(r.params) || "—" }}</td><td>{{ r.seed }}</td>
          <td>
            <div class="row" style="flex-wrap: nowrap; gap: 8px">
              <div class="bar" style="flex: 1; min-width: 80px"><span class="s-running" :style="{ width: (r.live?.pct ?? 0) + '%' }" /></div>
              <span class="small mono" style="white-space: nowrap">{{ steps(r.live?.step) }} / {{ steps(r.live?.total) }}</span>
              <span class="small faint" style="width: 42px; text-align: right">{{ r.live?.pct?.toFixed(1) ?? "—" }}%</span>
            </div>
            <div v-if="stale(r)" class="small" style="color: var(--warn)">{{ t("日志超过 10 分钟没更新", "log not updated for 10+ min") }}</div>
          </td>
          <td class="num mono">{{ r.live?.steps_per_sec ?? "—" }}</td>
          <td class="num mono">{{ r.live?.lambda != null ? num(r.live.lambda, 3) : "" }}</td>
          <td class="num">{{ pct(r.live?.success_last500) }}</td>
          <td class="num small">{{ r.live?.episodes != null ? Math.round(r.live.episodes).toLocaleString() : "—" }}</td>
          <td class="num small">{{ r.live?.eta_seconds != null ? dur(r.live.eta_seconds) : "—" }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.lr { border-top: 1px solid var(--border); padding: 9px 0; cursor: pointer; }
.lr:first-child { border-top: 0; padding-top: 0; }
.lr-h { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.lr-s { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 4px; font-variant-numeric: tabular-nums; font-size: 12.5px; }
.grp { background: var(--panel-2); border-radius: 8px; padding: 6px 10px; margin-bottom: 8px; }
.grp-r { display: grid; grid-template-columns: 52px 40px 1fr 1.4fr 1fr; gap: 4px; font-size: 12px; padding: 3px 0; font-variant-numeric: tabular-nums; }
.more { width: 100%; justify-content: center; margin-top: 8px; }
.lr-s i { display: block; font-style: normal; font-size: 11px; color: var(--faint); }
</style>
