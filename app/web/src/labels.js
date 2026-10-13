// Names and colours shared by every page.
import { t } from "./i18n.js";

export const SCENARIOS = ["lt", "eq", "gt"];
export const scenName = (s) => ({ lt: t("LT 资源紧缺", "LT scarce"), eq: t("EQ 供需均衡", "EQ balanced"), gt: t("GT 资源充裕", "GT ample") }[s] ?? s);
export const LEARNERS = ["ppo", "dqn", "ddqn"];
export const MECHS = ["none", "lagrange", "mask", "repair"];
export const mechName = (m) => ({ none: t("无约束", "Unconstrained"), lagrange: "Lagrangian", mask: "Maskable", repair: "Repair" }[m] ?? m);
export const learnerName = (l) => (l ?? "").toUpperCase();
export const modelName = (l, m) => (m === "none" ? `${learnerName(l)}（${t("无约束", "unconstrained")}）` : `${mechName(m)}-${learnerName(l)}`);
export const MECH_COLOR = { none: "#64748b", lagrange: "#d97706", mask: "#4f46e5", repair: "#059669" };
export const LEARNER_SYMBOL = { ppo: "circle", dqn: "triangle", ddqn: "rect" };
export const SCEN_COLOR = { lt: "#dc2626", eq: "#2563eb", gt: "#059669" };
export const LEARNER_COLOR = { ppo: "#4f46e5", dqn: "#d97706", ddqn: "#0891b2" };

export const statusName = (s) => ({
  running: t("运行中", "Running"), finished: t("已完成", "Finished"), done: t("已完成", "Done"),
  stopped: t("已中断", "Stopped"), cancelled: t("已作废", "Cancelled"), queued: t("排队中", "Queued"),
  waiting: t("等待依赖", "Waiting"), skipped: t("未运行", "Not run"),
}[s] ?? s);
export const STATUS_TONE = { running: "run", finished: "ok", done: "ok", stopped: "warn", cancelled: "off", queued: "off", waiting: "off", skipped: "off" };
export const purposeName = (p) => ({ main: t("正式", "Main"), smoke: "smoke", test: t("测试", "Test") }[p] ?? p);
export const roleName = (r) => ({ main: t("正式", "Main"), sweep: t("参数扫描", "Sweep"), reference: t("参照", "Reference") }[r] ?? r);
export const groupName = (g) => {
  const m = /^lam_sweep(?:_(\w+))?$/.exec(g ?? "");
  if (m) return t("λ_max 扫描", "λ_max sweep") + (m[1] ? `（${m[1].toUpperCase()}）` : "");
  return g ?? "";
};
export function splitAlgo(a) {
  const old = { ppo_lagrangian: "lagrange", ppo_mask: "mask", ppo_opt: "repair" };
  if (old[a]) return ["ppo", old[a]];
  const p = (a ?? "").split("_");
  if (p.length > 1 && MECHS.includes(p[0]) && LEARNERS.includes(p[p.length - 1])) return [p[p.length - 1], p[0]];
  return [a, "none"];
}
