<script setup>
// A sortable table header cell (see sort.js): click to sort by `k`; ▲ ascending, ▼ descending.
defineProps({ s: { type: Object, required: true }, k: { type: String, required: true }, num: Boolean, title: String });
</script>
<template>
  <th class="sortable" :class="{ num, sorted: s.by.value === k }" :title="title" @click="s.toggle(k)">
    <slot /><span class="sort-mark">{{ s.by.value === k ? (s.dir.value === 1 ? "▲" : "▼") : "↕" }}</span>
  </th>
</template>
