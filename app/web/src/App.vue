<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue";
import { useRoute } from "vue-router";
import { t, lang, setLang } from "./i18n.js";
import { getOverview } from "./api.js";

const route = useRoute();
const open = ref(false);
const running = ref(0);
const nav = computed(() => [
  { key: "overview", to: "/", icon: "◎", label: t("总览", "Overview") },
  { key: "monitor", to: "/monitor", icon: "◉", label: t("实时监控", "Live monitor") },
  { key: "batches", to: "/batches", icon: "☰", label: t("批次", "Batches") },
]);
let timer;
const poll = async () => { const o = await getOverview(); if (o) running.value = o.running.length; };
onMounted(() => { poll(); timer = setInterval(poll, 20000); setLang(lang.value); });
onUnmounted(() => clearInterval(timer));
</script>

<template>
  <div class="shell">
    <div class="side-mask" :class="{ open }" @click="open = false" />
    <aside class="side" :class="{ open }" @click="open = false">
      <div class="brand">
        <div class="brand-mark">R</div>
        <div><div class="brand-name">{{ t("实验面板", "Experiments") }}</div><div class="brand-sub">ILP → Safe RL</div></div>
      </div>
      <nav class="nav">
        <router-link v-for="n in nav" :key="n.key" :to="n.to" :class="{ active: route.meta.nav === n.key }">
          <span class="nav-icon">{{ n.icon }}</span>{{ n.label }}
          <span v-if="n.key === 'overview' && running" class="nav-badge" :title="t('运行中的批次', 'Running batches')">{{ running }}</span>
        </router-link>
      </nav>
      <div class="side-foot">
        <div class="seg">
          <button :class="{ on: lang === 'zh' }" @click.stop="setLang('zh')">中文</button>
          <button :class="{ on: lang === 'en' }" @click.stop="setLang('en')">English</button>
        </div>
        <a class="classic-link" href="/classic/" :title="t('旧版面板：v5.0.2 及更早的批次、版本报告、结果对比、论文草稿', 'Classic panel: v5.0.2 and older batches, version reports, result comparison, paper draft')">
          <span>{{ t("旧版面板", "Classic panel") }}</span><span>↗</span>
        </a>
      </div>
    </aside>
    <div class="main">
      <div class="mobile-bar">
        <button class="btn" @click="open = !open">☰</button>
        <b>{{ t("实验面板", "Experiments") }}</b>
      </div>
      <router-view :key="route.path" />
    </div>
  </div>
</template>
