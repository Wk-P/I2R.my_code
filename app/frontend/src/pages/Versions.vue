<script setup>
import { ref, computed, onMounted } from "vue";
import { getTags } from "../api.js";

const tags = ref([]);
const loading = ref(true);
const q = ref("");
onMounted(async () => { tags.value = await getTags(); loading.value = false; });
const shown = computed(() => {
  const s = q.value.trim().toLowerCase();
  return s ? tags.value.filter((t) => (t.tag + " " + (t.summary ?? "")).toLowerCase().includes(s)) : tags.value;
});
const go = (h) => (window.location.hash = h);
</script>

<template>
  <div class="page-head">
    <h1>版本记录</h1>
    <div class="page-sub">每个 git tag 一行，摘要来自 version/VERSION.md，有独立文档的可点击查看</div>
  </div>
  <section class="panel">
    <div class="panel-head"><input v-model="q" class="search" placeholder="搜索版本号或摘要…" /><span class="dim small">{{ shown.length }} 个版本</span></div>
    <div v-if="loading" class="empty">加载中…</div>
    <table v-else class="grid">
      <thead><tr><th>版本</th><th>日期</th><th>摘要</th><th>文档</th></tr></thead>
      <tbody>
        <tr v-for="t in shown" :key="t.tag" class="clickable" @click="go(`#/versions/${t.tag}`)">
          <td class="mono nowrap"><a :href="`#/versions/${t.tag}`">{{ t.tag }}</a></td>
          <td class="nowrap">{{ t.date }}</td>
          <td class="summary">{{ t.summary || "—" }}</td>
          <td><span v-if="t.has_doc" class="badge badge--ok">有</span></td>
        </tr>
      </tbody>
    </table>
  </section>
</template>
