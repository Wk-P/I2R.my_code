<script setup>
// Generic "what's actually running right now" view — independent of
// ProgressPanel's one-process-per-scenario training-pipeline assumption and
// BatchProgressPanel's scripts/logs/ naming convention. Backs onto
// GET /api/system (see app/backend/main.py), which surfaces every live
// process tied to this project (venv interpreter or cwd inside the repo),
// so an ad-hoc hyperparameter sweep or a one-off timing probe shows up here
// too, not just the formal run_all.py campaign.
import { ref, onMounted, onUnmounted } from "vue";
import { getSystem } from "../api.js";
import { secToClock } from "../format.js";

const data = ref(null);
let timer = null;

async function load() {
  data.value = await getSystem();
}

onMounted(() => {
  load();
  timer = setInterval(load, 4000);
});
onUnmounted(() => clearInterval(timer));

function loadClass(v) {
  if (v === null || v === undefined) return "";
  const perCore = v / (data.value?.cpu_count || 1);
  if (perCore > 0.85) return "sys-load--high";
  if (perCore > 0.5) return "sys-load--mid";
  return "";
}
</script>

<template>
  <div v-if="data" class="card system-panel">
    <div class="title">
      Running now
      <span class="sys-load" :class="loadClass(data.load_avg['1m'])">
        load {{ data.load_avg["1m"]?.toFixed(1) }} / {{ data.load_avg["5m"]?.toFixed(1) }} / {{ data.load_avg["15m"]?.toFixed(1) }}
        (of {{ data.cpu_count }} cores)
      </span>
    </div>
    <table v-if="data.processes.length" class="system-table">
      <thead>
        <tr>
          <th>what</th>
          <th>pid</th>
          <th>core</th>
          <th>cpu%</th>
          <th>elapsed</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="p in data.processes" :key="p.pid">
          <td class="system-label">{{ p.label }}</td>
          <td class="system-dim">{{ p.pid }}</td>
          <td class="system-dim">{{ p.core }}</td>
          <td class="system-dim">{{ p.cpu_percent.toFixed(0) }}%</td>
          <td class="system-dim">{{ secToClock(p.elapsed_seconds) }}</td>
        </tr>
      </tbody>
    </table>
    <div v-else class="system-empty">Nothing project-related running right now.</div>
  </div>
</template>
