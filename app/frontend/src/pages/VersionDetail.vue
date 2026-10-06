<script setup>
import { ref, watch } from "vue";
import { getTags, getTagDoc } from "../api.js";
import { langRef } from "../i18n.js";

const props = defineProps({ tag: { type: String, required: true } });
const meta = ref(null);
const doc = ref(null);
const loading = ref(true);

watch([() => props.tag, langRef], async ([tag, lang]) => {
  loading.value = true;
  meta.value = (await getTags()).find((t) => t.tag === tag) ?? null;
  doc.value = meta.value?.has_doc ? await getTagDoc(tag, lang) : null;
  loading.value = false;
}, { immediate: true });
</script>

<template>
  <div class="crumbs"><a href="#/versions">版本记录</a> / {{ tag }}</div>
  <div v-if="loading" class="empty">加载中…</div>
  <div v-else-if="!meta" class="empty">找不到版本 {{ tag }}</div>
  <template v-else>
    <div class="page-head">
      <h1>{{ tag }}</h1>
      <div class="page-sub">{{ meta.date }}</div>
    </div>
    <section class="panel">
      <div class="panel-head"><h2>摘要</h2></div>
      <div class="prose">{{ meta.summary || "—" }}</div>
    </section>
    <section class="panel">
      <div class="panel-head"><h2>文档</h2></div>
      <pre v-if="doc" class="doc">{{ doc.content }}</pre>
      <div v-else class="empty">该版本没有独立文档</div>
    </section>
  </template>
</template>
