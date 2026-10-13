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
  { key: "batches", to: "/batches", icon: "☰", label: t("批次", "Batches") },
  { key: "machine", to: "/machine", icon: "▣", label: t("机器", "Machine") },
]);
// classic panel: the same page there when one exists
const classic = computed(() => {
  const p = route.path;
  let m;
  if ((m = p.match(/^\/batches\/(.+)$/))) return `/classic/#/batches/${m[1]}`;
  if ((m = p.match(/^\/runs\/(.+)$/))) return `/classic/#/run/${m[1]}`;
  if (p === "/machine") return "/classic/#/monitor";
  if (p === "/batches") return "/classic/#/batches";
  return "/classic/";
});
let timer;
const poll = async () => { const o = await getOverview(); if (o) running.value = o.running.length; };
onMounted(() => { poll(); timer = setInterval(poll, 20000); setLang(lang.value); });
onUnmounted(() => clearInterval(timer));
</script>

<template>
  <div class="shell">
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
        <a class="classic-link" :href="classic" :title="t('旧版面板（版本报告、结果对比、图表、论文草稿仍在那里）', 'Classic panel (version reports, result comparison, charts and the paper draft are still there)')">
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
