import { createRouter, createWebHashHistory } from "vue-router";
import Overview from "./pages/Overview.vue";
import Batches from "./pages/Batches.vue";
import Batch from "./pages/Batch.vue";
import Run from "./pages/Run.vue";
import Machine from "./pages/Machine.vue";

export const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: "/", component: Overview, meta: { nav: "overview" } },
    { path: "/batches", component: Batches, meta: { nav: "batches" } },
    { path: "/batches/:name", component: Batch, props: true, meta: { nav: "batches" } },
    { path: "/runs/:space/:scenario/:algo/:exp_id", component: Run, props: true, meta: { nav: "batches" } },
    { path: "/machine", component: Machine, meta: { nav: "machine" } },
    { path: "/:p(.*)*", redirect: "/" },
  ],
  scrollBehavior: (to, from) => (to.path !== from.path ? { top: 0 } : undefined),
});
