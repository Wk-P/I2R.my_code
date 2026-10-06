<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue";
import { store, startPolling, trainingProcs, setViewBranch } from "./store.js";
import { getBatches } from "./api.js";
import Overview from "./pages/Overview.vue";
import Monitor from "./pages/Monitor.vue";
import Batches from "./pages/Batches.vue";
import BatchDetail from "./pages/BatchDetail.vue";
import Results from "./pages/Results.vue";
import RunDetail from "./pages/RunDetail.vue";
import Versions from "./pages/Versions.vue";
import VersionDetail from "./pages/VersionDetail.vue";
import PaperDraft from "./components/PaperDraft.vue";
import PaperDraftEn from "./components/PaperDraftEn.vue";
import { langRef as curLang, setLang } from "./i18n.js";
const toggleLang = () => setLang(curLang.value === "en" ? "zh" : "en");

// Hash routes:
//   #/                       总览
//   #/monitor                训练监控
//   #/batches[/<name>]       批次管理 / 批次详情
//   #/results                实验结果
//   #/run/<branch>/<scen>/<algo>/<run>   运行详情
//   #/versions[/<tag>]       版本记录 / 版本详情
//   #/paper                  论文草稿
const hash = ref(window.location.hash || "#/");
const menuOpen = ref(false);   // mobile navigation drawer
const onHash = () => {
  menuOpen.value = false;
  const prev = hash.value.split("?")[0];
  hash.value = window.location.hash || "#/";
  if (hash.value.split("?")[0] !== prev) window.scrollTo(0, 0);
};
onMounted(() => { window.addEventListener("hashchange", onHash); startPolling(); });
onUnmounted(() => window.removeEventListener("hashchange", onHash));

const route = computed(() => {
  const h = decodeURIComponent(hash.value.replace(/^#/, "").split("?")[0]) || "/";
  let m;
  if ((m = h.match(/^\/run\/([^/]+)\/([^/]+)\/([^/]+)\/([^/]+)$/)))
    return { page: "run", nav: "results", props: { branch: m[1], scenario: m[2], algo: m[3], run: m[4] } };
  if ((m = h.match(/^\/batches\/(.+)$/))) return { page: "batch", nav: "batches", props: { name: m[1] } };
  if ((m = h.match(/^\/versions\/(.+)$/))) return { page: "version", nav: "versions", props: { tag: m[1] } };
  const simple = { "/monitor": "monitor", "/batches": "batches", "/results": "results", "/versions": "versions", "/paper": "paper" };
  const page = simple[h] ?? "overview";
  return { page, nav: page, props: {} };
});

const NAV = [
  { key: "overview", href: "#/", icon: "▦", label: "总览" },
  { key: "monitor", href: "#/monitor", icon: "◉", label: "训练监控" },
  { key: "batches", href: "#/batches", icon: "☰", label: "批次管理" },
  { key: "results", href: "#/results", icon: "▤", label: "实验结果" },
  { key: "versions", href: "#/versions", icon: "⎇", label: "版本记录" },
  { key: "paper", href: "#/paper", icon: "✎", label: "论文草稿" },
];

const running = computed(() => trainingProcs().length);
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
        <a v-for="n in NAV" :key="n.key" :href="n.href" class="nav-item" :class="{ active: route.nav === n.key }">
          <span class="nav-icon">{{ n.icon }}</span>{{ n.label }}
          <span v-if="n.key === 'monitor' && running" class="nav-count">{{ running }}</span>
        </a>
      </nav>
      <div class="sidebar-foot">只读面板，不影响任何训练进程</div>
    </aside>

    <div class="main">
      <header class="topbar">
        <div class="topbar-left">
          <button class="menu-btn" aria-label="菜单" @click="menuOpen = !menuOpen">☰</button>
          <input v-model="search" class="global-search" placeholder="搜索 exp_id / 批次名 / 算法，回车" @keyup.enter="doSearch" />
          <label class="branch-select" title="切换要查看的数据空间（results/ 下的目录，不会执行 git checkout）">
            <span>数据空间</span>
            <select :value="store.viewBranch" @change="setViewBranch($event.target.value)">
              <option v-for="b in store.resultBranches" :key="b.name" :value="b.name">
                {{ b.label }}（{{ b.versions }}）· {{ b.runs }} 次{{ b.name === store.branch ? " · 新训练写入此处" : "" }}
              </option>
            </select>
          </label>
          <span v-if="store.viewBranch && store.viewBranch !== store.branch" class="chip chip--warn-bg"
            :title="`新训练写入 results/${store.branch}/`">新训练写入 <b>{{ store.branch }}</b></span>
        </div>
        <div class="topbar-right">
          <a href="#/monitor" class="chip" :class="running ? 'chip--run' : ''">训练中 <b>{{ running }}</b></a>
          <span class="chip" :class="loadClass" :title="store.system ? `load ${store.system.load_avg['1m']?.toFixed(1)} / ${store.system.cpu_count} 核` : ''">
            CPU 负载 <b>{{ loadPct === null ? "—" : loadPct + "%" }}</b>
          </span>
          <span class="chip chip--plain">更新 {{ updated }}</span>
          <button class="chip" title="中文 / English" @click="toggleLang">{{ curLang === "en" ? "中文" : "EN" }}</button>
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
        <VersionDetail v-else-if="route.page === 'version'" v-bind="route.props" />
        <PaperDraftEn v-else-if="route.page === 'paper' && curLang === 'en'" />
        <PaperDraft v-else-if="route.page === 'paper'" />
      </main>
    </div>
  </div>
</template>
