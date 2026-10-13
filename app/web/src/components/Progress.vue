<script setup>
// Stacked progress of a batch: done / running (by their own progress) / stopped, out of the planned total.
import { computed } from "vue";
import { t } from "../i18n.js";
const props = defineProps({ b: Object, legend: Boolean });
const w = (n) => `${(100 * n) / Math.max(1, props.b.total)}%`;
const c = computed(() => props.b.counts || {});
</script>
<template>
  <div>
    <div class="bar">
      <span class="s-done" :style="{ width: w(c.done || 0) }" />
      <span class="s-running" :style="{ width: w(c.running || 0) }" />
      <span class="s-stopped" :style="{ width: w(c.stopped || 0) }" />
    </div>
    <div v-if="legend" class="legend-dots" style="margin-top: 8px">
      <span><i style="background: var(--ok)" />{{ t("已完成", "Done") }} {{ c.done || 0 }}</span>
      <span v-if="c.running"><i style="background: var(--run)" />{{ t("运行中", "Running") }} {{ c.running }}</span>
      <span v-if="c.waiting"><i style="background: var(--off)" />{{ t("等待依赖", "Waiting") }} {{ c.waiting }}</span>
      <span v-if="c.queued"><i style="background: var(--off-soft); border: 1px solid var(--border-strong)" />{{ t("排队中", "Queued") }} {{ c.queued }}</span>
      <span v-if="c.stopped"><i style="background: var(--warn)" />{{ t("已中断", "Stopped") }} {{ c.stopped }}</span>
      <span v-if="c.skipped"><i style="background: var(--off-soft); border: 1px solid var(--border-strong)" />{{ t("未运行", "Not run") }} {{ c.skipped }}</span>
      <span class="faint">{{ t("共", "Total") }} {{ b.total }}</span>
    </div>
  </div>
</template>
