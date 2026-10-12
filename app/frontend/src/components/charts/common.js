// Shared bits of the chart page (v5.0.0): palette, width observer, version ordering.
import { ref, onMounted, onUnmounted } from "vue";

// Categorical slots (validated light-surface palette, adjacent pairs): fixed order, never cycled.
export const CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
export const SCEN_COLOR = { LT: CAT[0], EQ: CAT[1], GT: CAT[2], AVG: "var(--text)" };
export const TARGET = 5;                                   // the shared 5% gap goal

export function useWidth(min = 280) {
  const el = ref(null);
  const width = ref(600);
  let ro = null;
  onMounted(() => {
    ro = new ResizeObserver(([e]) => (width.value = Math.max(min, Math.floor(e.contentRect.width))));
    if (el.value) ro.observe(el.value);
  });
  onUnmounted(() => ro?.disconnect());
  return { el, width };
}

const vkey = (v) => v.replace(/^v/, "").split(".").map(Number);
export const vcmp = (a, b) => {
  const x = vkey(a), y = vkey(b);
  for (let i = 0; i < Math.max(x.length, y.length); i++) { const d = (x[i] ?? 0) - (y[i] ?? 0); if (d) return d; }
  return 0;
};

// "nice" upper bound for a % axis
export const niceMax = (v, step = 2) => Math.max(step, Math.ceil(v / step) * step);
export const fmt1 = (v) => (v == null ? "—" : `${v.toFixed(1)}%`);

// learner / mechanism of a series label (v5.0.2 labels: "无约束 PPO", "Lagrangian DQN", "Maskable DDQN（v4.5.2）", ...)
export const learnerOf = (s) => (/DDQN/.test(s) ? "DDQN" : /DQN/.test(s) ? "DQN" : /PPO/.test(s) ? "PPO" : null);
const MECHS = [["无约束|Unconstrained", 0], ["Lagrang", 1], ["Mask", 2], ["Repair", 3]];
export const mechSlot = (s) => { for (const [re, k] of MECHS) if (new RegExp(re, "i").test(s)) return k; return null; };
export const MECH_NAMES = [["无约束", "Unconstrained"], ["Lagrangian", "Lagrangian"], ["Maskable", "Maskable"], ["Repair", "Repair"]];
