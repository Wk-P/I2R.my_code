<script setup>
import { statusName, STATUS_TONE } from "../labels.js";
defineProps({ s: String, pct: Number });
</script>
<template>
  <span class="badge" :class="STATUS_TONE[s] || 'off'">{{ statusName(s) }}<template v-if="s === 'running' && pct != null"> · {{ pct.toFixed(0) }}%</template></span>
</template>
