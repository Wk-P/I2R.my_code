// Names and colours shared by every page. Field names and values follow docs/data_spec.md (schema 3):
// RL_algorithm PPO / DQN / DDQN, strategy unconstraint / lagrangian / maskable / repair.
import { t } from "./i18n.js";

export const SCENARIOS = ["lt", "eq", "gt"];
export const scenName = (s) => ({ lt: t("LT 资源紧缺", "LT scarce"), eq: t("EQ 供需均衡", "EQ balanced"), gt: t("GT 资源充裕", "GT ample") }[s] ?? s);
export const RL_ALGORITHMS = ["PPO", "DQN", "DDQN"];
export const STRATEGIES = ["unconstraint", "lagrangian", "maskable", "repair"];
export const modelName = (rl, st) => `${rl} · ${st}`;
export const STRATEGY_COLOR = { unconstraint: "#64748b", lagrangian: "#d97706", maskable: "#4f46e5", repair: "#059669" };
export const RL_SYMBOL = { PPO: "circle", DQN: "triangle", DDQN: "rect" };
export const SCEN_COLOR = { lt: "#dc2626", eq: "#2563eb", gt: "#059669" };

// results directory name -> [RL_algorithm, strategy] (report_data.json still names runs by directory)
const PREFIX = { lagrange: "lagrangian", mask: "maskable", repair: "repair" };
export function fromDir(d) {
  const p = (d ?? "").split("_");
  const rl = p[p.length - 1].toUpperCase();
  return [rl, p.length > 1 ? PREFIX[p[0]] ?? p[0] : "unconstraint"];
}

export const statusName = (s) => ({
  running: t("运行中", "Running"), finished: t("已完成", "Finished"), done: t("已完成", "Done"),
  stopped: t("已中断", "Stopped"), cancelled: t("已作废", "Cancelled"), queued: t("排队中", "Queued"),
  waiting: t("等待", "Waiting"), skipped: t("未运行", "Not run"),
}[s] ?? s);
export const STATUS_TONE = { running: "run", finished: "ok", done: "ok", stopped: "warn", cancelled: "off", queued: "off", waiting: "off", skipped: "off" };
export const purposeName = (p) => ({ main: t("正式", "Main"), smoke: "smoke", test: t("测试", "Test") }[p] ?? p);
export const roleName = (r) => ({ main: t("正式", "Main"), sweep: t("参数扫描", "Sweep"), reference: t("参照", "Reference") }[r] ?? r);
export const groupName = (g) => {
  const m = /^lambda_max_sweep_(\w+)$/.exec(g ?? "");
  return m ? `${t("λ_max 扫描", "λ_max sweep")} · ${m[1]}` : g ?? "";
};
