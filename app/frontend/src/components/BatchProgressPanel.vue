<script setup>
// Progress + results view for an ad-hoc parallel batch (many algos x many
// seeds running at once), e.g. scripts/run_full_5M_campaign.py — deliberately
// separate from ProgressPanel.vue, which assumes one sequential process per
// scenario and shows "N/6 models done" against a fixed ALGO_ORDER. That
// model breaks down for a batch like this: many algos x many seeds run
// concurrently, so "which one is 'current'" is meaningless.
//
// Per-(scenario, algo) rows show seed progress as a compact dot row plus a
// seed-averaged mean±std once at least one seed is done -- not one row per
// seed. Dedup (a slot can have more than one run behind it after a
// crashed/retried attempt) and the mean/std math both happen server-side
// (see app/backend/main.py's get_batch_progress) so this component only
// renders what it's given.
import { ref, onMounted, onUnmounted } from "vue";
import { getBatchProgress } from "../api.js";
import { secToClock } from "../format.js";

const props = defineProps({ batchName: { type: String, required: true } });

const data = ref(null);
let pollTimer = null;

// Elapsed time ticks every second on the client so it reads like a live
// clock, but is resynced to the server's elapsed_seconds (derived from the
// batch's oldest log file) on every 5s poll — client-side drift never
// accumulates for more than one poll interval.
const displaySeconds = ref(null);
let tickTimer = null;
let serverElapsedAtLastPoll = null;
let clientTimeAtLastPoll = null;

async function load() {
  data.value = await getBatchProgress(props.batchName);
  if (data.value?.elapsed_seconds != null) {
    serverElapsedAtLastPoll = data.value.elapsed_seconds;
    clientTimeAtLastPoll = Date.now();
    displaySeconds.value = serverElapsedAtLastPoll;
  }
}

onMounted(() => {
  load();
  pollTimer = setInterval(load, 5000);
  tickTimer = setInterval(() => {
    if (serverElapsedAtLastPoll != null) {
      displaySeconds.value = serverElapsedAtLastPoll + (Date.now() - clientTimeAtLastPoll) / 1000;
    }
  }, 1000);
});
onUnmounted(() => {
  clearInterval(pollTimer);
  clearInterval(tickTimer);
});

const SEED_ORDER = [1, 2, 3, 4, 5];
function algosFor(scenarioBlock) {
  const seen = [];
  for (const r of scenarioBlock?.runs || []) {
    if (!seen.includes(r.algo)) seen.push(r.algo);
  }
  return seen;
}
function runFor(scenarioBlock, algo, seed) {
  // Backend already dedups to one run per (scenario, algo, seed) slot, so
  // this is a plain lookup, not a rank-and-pick.
  return (scenarioBlock?.runs || []).find((r) => r.algo === algo && r.seed === seed);
}
function aggregateFor(scenario, algo) {
  return (data.value?.aggregates?.[scenario] || []).find((a) => a.algo === algo);
}
function pct(x) {
  return x === undefined || x === null ? "—" : (x * 100).toFixed(1) + "%";
}
function fixed(x, n = 4) {
  return x === undefined || x === null ? "—" : x.toFixed(n);
}
</script>

<template>
  <div v-if="data" class="card batch-panel">
    <div class="title">
      Batch: {{ data.batch_name }} — {{ data.done }}/{{ data.total_runs }} runs done ({{ data.overall_pct }}%)
      <span class="batch-elapsed" :title="'Time since the batch\'s first run was launched'">
        ⏱ {{ secToClock(displaySeconds) }}
      </span>
    </div>
    <div class="bar-bg">
      <div class="bar-fill batch" :style="{ width: data.overall_pct + '%' }"></div>
    </div>

    <div v-for="(block, scenario) in data.by_scenario" :key="scenario" class="batch-scenario-block">
      <div class="batch-scenario-label">
        {{ scenario }} — {{ block.done }} done · {{ block.running }} running · {{ block.queued }} queued
      </div>
      <table class="batch-table">
        <thead>
          <tr>
            <th>algo</th>
            <th>seeds</th>
            <th>success_rate (mean±std)</th>
            <th>AR (mean±std)</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="algo in algosFor(block)" :key="algo">
            <td class="batch-algo-name">{{ algo }}</td>
            <td class="batch-cell">
              <span class="batch-seed-dots">
                <span
                  v-for="seed in SEED_ORDER"
                  :key="seed"
                  class="batch-seed-dot"
                  :class="`batch-seed-dot--${runFor(block, algo, seed)?.status ?? 'queued'}`"
                  :title="`seed ${seed}: ${runFor(block, algo, seed)?.status ?? 'queued'}`"
                ></span>
              </span>
            </td>
            <template v-if="aggregateFor(scenario, algo)?.n_done">
              <td class="batch-cell-metric">
                {{ pct(aggregateFor(scenario, algo).success_rate_mean) }} ± {{ pct(aggregateFor(scenario, algo).success_rate_std) }}
                <span v-if="aggregateFor(scenario, algo).n_done < aggregateFor(scenario, algo).n_total" class="batch-partial-tag"
                      :title="'Only ' + aggregateFor(scenario, algo).n_done + '/' + aggregateFor(scenario, algo).n_total + ' seeds done — average will shift as more finish'">
                  ({{ aggregateFor(scenario, algo).n_done }}/{{ aggregateFor(scenario, algo).n_total }}, partial)
                </span>
              </td>
              <td class="batch-cell-metric">
                {{ fixed(aggregateFor(scenario, algo).ar_mean) }} ± {{ fixed(aggregateFor(scenario, algo).ar_std) }}
              </td>
            </template>
            <template v-else>
              <td class="batch-cell-metric">—</td>
              <td class="batch-cell-metric">—</td>
            </template>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="batch-legend">
      <span class="batch-legend-item"><span class="batch-seed-dot batch-seed-dot--done"></span> done</span>
      <span class="batch-legend-item"><span class="batch-seed-dot batch-seed-dot--running"></span> running</span>
      <span class="batch-legend-item"><span class="batch-seed-dot batch-seed-dot--queued"></span> queued</span>
      <span>rows = algorithm · dots = per-seed progress · metrics = mean±std across done seeds ("partial" until all seeds finish)</span>
    </div>
  </div>
</template>
