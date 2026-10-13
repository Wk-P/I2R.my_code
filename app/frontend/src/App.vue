<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue";
import { store, startPolling, trainingProcs } from "./store.js";
import { getBatches } from "./api.js";
import Overview from "./pages/Overview.vue";
import Monitor from "./pages/Monitor.vue";
import Batches from "./pages/Batches.vue";
import BatchDetail from "./pages/BatchDetail.vue";
import Results from "./pages/Results.vue";
import RunDetail from "./pages/RunDetail.vue";
import Versions from "./pages/Versions.vue";
import VersionDetail from "./pages/VersionDetail.vue";
import Charts from "./pages/Charts.vue";
import PaperDraft from "./components/PaperDraft.vue";
import PaperDraftEn from "./components/PaperDraftEn.vue";
import { langRef as curLang, setLang } from "./i18n.js";

// Hash routes (sidebar groups: 当前 / 实验 / 分析 / 论文):
//   #/                       总览（进行中的批次、实验队列、机器）
//   #/monitor                进程监控
//   #/batches[/<name>]       批次（按版本分组，按阶段筛选）/ 批次详情（结果矩阵）
//   #/run/<space>/<scen>/<algo>/<run>   run 详情（面包屑：批次 / 版本 / 批次名 / run）
//   #/versions[/<tag>]       版本报告（时间线）/ 某版本的报告
//   #/results                结果对比（跨批次明细 / 透视）
//   #/charts                 图表
//   #/paper                  论文草稿
const hash = ref(window.location.hash || "#/");
const menuOpen = ref(false);   // mobile navigation drawer
const onHash = () => {
  menuOpen.value = false;
  const prev = hash.value.split("?")[0];
  hash.value = window.location.hash || "#/";
  if (hash.value.split("?")[0] !== prev) window.scrollTo(0, 0);
};
// Whole-row click: a table row that contains a link opens its first link when clicked anywhere, except on a
// control of its own, while selecting text, or when the row already has its own click handler (.clickable).
function onRowClick(e) {
  if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey) return;
  if (e.target.closest("a, button, input, select, textarea, summary, label")) return;
  if (window.getSelection()?.toString()) return;
  const tr = e.target.closest("table.grid tbody tr");
  if (!tr || tr.classList.contains("clickable")) return;
  const a = tr.querySelector("a[href]");
  if (!a) return;
  const href = a.getAttribute("href");
  if (href.startsWith("#")) window.location.hash = href;
  else window.location.href = href;
}
onMounted(() => { window.addEventListener("hashchange", onHash); document.addEventListener("click", onRowClick); startPolling(); });
onUnmounted(() => { window.removeEventListener("hashchange", onHash); document.removeEventListener("click", onRowClick); });

const route = computed(() => {
  const h = decodeURIComponent(hash.value.replace(/^#/, "").split("?")[0]) || "/";
  let m;
  if ((m = h.match(/^\/run\/([^/]+)\/([^/]+)\/([^/]+)\/([^/]+)$/)))
    return { page: "run", nav: "batches", props: { branch: m[1], scenario: m[2], algo: m[3], run: m[4] } };
  if ((m = h.match(/^\/batches\/(.+)$/))) return { page: "batch", nav: "batches", props: { name: m[1] } };
  if ((m = h.match(/^\/versions\/(.+)$/))) return { page: "version", nav: "versions", props: { tag: m[1] } };
  const simple = { "/monitor": "monitor", "/batches": "batches", "/results": "results", "/versions": "versions", "/paper": "paper", "/charts": "charts" };
  const page = simple[h] ?? "overview";
  return { page, nav: page, props: {} };
});

const NAV = [
  { group: "当前", items: [
    { key: "overview", href: "#/", icon: "▦", label: "总览" },
    { key: "monitor", href: "#/monitor", icon: "◉", label: "进程监控" },
  ] },
  { group: "实验", items: [
    { key: "batches", href: "#/batches", icon: "☰", label: "批次" },
    { key: "versions", href: "#/versions", icon: "⎇", label: "版本报告" },
  ] },
  { group: "分析", items: [
    { key: "results", href: "#/results", icon: "▤", label: "结果对比" },
    { key: "charts", href: "#/charts", icon: "◔", label: "图表" },
  ] },
  { group: "论文", items: [
    { key: "paper", href: "#/paper", icon: "✎", label: "论文草稿" },
  ] },
];

const running = computed(() => trainingProcs().filter((p) => !p.frozen).length);
const frozen = computed(() => trainingProcs().filter((p) => p.frozen).length);   // SIGSTOP-ed training processes
const loadPct = computed(() => {
  const s = store.system;
  if (!s?.load_avg?.["1m"]) return null;
  return Math.round((100 * s.load_avg["1m"]) / (s.cpu_count || 1));
});
const loadClass = computed(() =>
  loadPct.value === null ? "" : loadPct.value > 90 ? "chip--bad" : loadPct.value > 60 ? "chip--warn" : "chip--ok");
// Global search: a batch name opens that batch, anything else searches results.
const search = ref("");
async function doSearch() {
  const q = search.value.trim();
  if (!q) return;
  const { batches } = await getBatches(true);
  const hit = batches.find((b) => b.batch_name === q) ?? batches.find((b) => b.batch_name.includes(q));
  window.location.hash = hit && hit.batch_name.includes(q) && !/^[0-9a-f]{8}$/.test(q)
    ? `#/batches/${hit.batch_name}`
    : `#/results?q=${encodeURIComponent(q)}`;
  search.value = "";
}

const updated = computed(() =>
  store.updatedAt ? store.updatedAt.toTimeString().slice(0, 8) : "—");
</script>

<template>
  <div class="shell" :class="{ 'menu-open': menuOpen }">
    <div class="drawer-mask" @click="menuOpen = false"></div>
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-title">实验管理台</div>
        <div class="brand-sub">ILP vs RL · 服务部署</div>
      </div>
      <nav>
        <template v-for="g in NAV" :key="g.group">
          <div class="nav-group">{{ g.group }}</div>
          <a v-for="n in g.items" :key="n.key" :href="n.href" class="nav-item" :class="{ active: route.nav === n.key }">
            <span class="nav-icon">{{ n.icon }}</span>{{ n.label }}
            <span v-if="n.key === 'monitor' && running" class="nav-count">{{ running }}</span>
          </a>
        </template>
      </nav>
      <a class="new-panel-link" href="/">新版面板 →</a>
      <div class="sidebar-foot">只读面板，不影响任何训练进程</div>
    </aside>

    <div class="main">
      <header class="topbar">
        <div class="topbar-left">
          <button class="menu-btn" aria-label="菜单" @click="menuOpen = !menuOpen">☰</button>
          <input v-model="search" class="global-search" placeholder="搜索 exp_id / 批次名 / 算法，回车" @keyup.enter="doSearch" />
        </div>
        <div class="topbar-right">
          <a href="#/monitor" class="chip" :class="running ? 'chip--run' : ''">训练中 <b>{{ running }}</b><template v-if="frozen"> · 冻结 <b>{{ frozen }}</b></template></a>
          <span class="chip" :class="loadClass" :title="store.system ? `load ${store.system.load_avg['1m']?.toFixed(1)} / ${store.system.cpu_count} 核` : ''">
            CPU 负载 <b>{{ loadPct === null ? "—" : loadPct + "%" }}</b>
          </span>
          <span class="chip chip--plain">更新 {{ updated }}</span>
          <div class="lang-switch" role="group" aria-label="Language / 语言" data-no-i18n>
            <button type="button" :class="{ on: curLang !== 'en' }" :aria-pressed="curLang !== 'en'" @click="setLang('zh')">中文</button>
            <button type="button" :class="{ on: curLang === 'en' }" :aria-pressed="curLang === 'en'" @click="setLang('en')">EN</button>
          </div>
        </div>
      </header>

      <main class="content">
        <Overview v-if="route.page === 'overview'" />
        <Monitor v-else-if="route.page === 'monitor'" />
        <Batches v-else-if="route.page === 'batches'" />
        <BatchDetail v-else-if="route.page === 'batch'" v-bind="route.props" />
        <Results v-else-if="route.page === 'results'" />
        <RunDetail v-else-if="route.page === 'run'" v-bind="route.props" />
        <Versions v-else-if="route.page === 'versions'" />
        <Charts v-else-if="route.page === 'charts'" />
        <VersionDetail v-else-if="route.page === 'version'" v-bind="route.props" />
        <PaperDraftEn v-else-if="route.page === 'paper' && curLang === 'en'" />
        <PaperDraft v-else-if="route.page === 'paper'" />
      </main>
    </div>
  </div>
</template>
