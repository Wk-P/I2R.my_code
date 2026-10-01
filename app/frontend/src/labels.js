// Display names shared by every page.

export const SCENARIOS = ["lt", "eq", "gt"];

export const SCENARIO_LABEL = {
  lt: "LT 资源紧缺",
  eq: "EQ 供需均衡",
  gt: "GT 资源充裕",
};

// 12 models = 3 learners x 4 constraint-handling mechanisms.
// The unconstrained learner is the control group.
export const LEARNERS = ["ppo", "dqn", "ddqn"];
export const MECHANISMS = ["none", "lagrange", "mask", "repair"];
export const MECH_LABEL = { none: "无约束", lagrange: "Lagrangian", mask: "Maskable", repair: "Repair" };
const OLD_PPO = { ppo_lagrangian: "lagrange", ppo_mask: "mask", ppo_opt: "repair" };  // pre-v4.3.1.3 dir names

export function learnerOf(a) {
  if (OLD_PPO[a]) return "ppo";
  const l = (a ?? "").split("_").pop();
  return LEARNERS.includes(l) ? l : a;
}
export function mechOf(a) {
  if (OLD_PPO[a]) return OLD_PPO[a];
  const parts = (a ?? "").split("_");
  return parts.length > 1 && MECHANISMS.includes(parts[0]) ? parts[0] : "none";
}
export const learnerLabel = (a) => learnerOf(a).toUpperCase();
export const mechLabel = (a) => MECH_LABEL[mechOf(a)];

// ordered by learner, then 无约束 -> Lagrangian -> Maskable -> Repair
export const ALGO_ORDER = LEARNERS.flatMap((l) => MECHANISMS.map((m) => (m === "none" ? l : `${m}_${l}`)));
const ALGO_LABEL = { ppo_lagrangian: "Lagrangian-PPO", ppo_mask: "Maskable-PPO", ppo_opt: "Repair-PPO" };
for (const a of ALGO_ORDER) {
  const m = mechOf(a), l = learnerOf(a).toUpperCase();
  ALGO_LABEL[a] = m === "none" ? `${l}（无约束）` : `${MECH_LABEL[m]}-${l}`;
}

export const algoLabel = (a) => ALGO_LABEL[a] ?? a;
export const algoIndex = (a) =>
  LEARNERS.includes(learnerOf(a))
    ? LEARNERS.indexOf(learnerOf(a)) * 10 + MECHANISMS.indexOf(mechOf(a))
    : 99;
export const scenarioIndex = (s) => {
  const i = SCENARIOS.indexOf(s);
  return i === -1 ? SCENARIOS.length : i;
};

// Batch / run status -> [label, css modifier]
export const STATUS = {
  running: ["运行中", "run"],
  finished: ["已完成", "ok"],
  done: ["已完成", "ok"],
  stopped: ["已中断", "warn"],
  cancelled: ["已作废", "off"],
  queued: ["排队中", "off"],
  skipped: ["未运行", "off"],
};
export const statusLabel = (s) => (STATUS[s] ?? [s])[0];
export const statusClass = (s) => "badge--" + ((STATUS[s] ?? [s, "off"])[1]);

// Repair-type algorithms never execute a violating placement; the stored
// violation counts are repair triggers, which are not reported.
export const isRepair = (a) => a === "ppo_opt" || /^repair_/.test(a ?? "");
const VIOL_KEYS = /viol_(rate|total)/;
export function zeroRepairViol(obj) {
  if (obj && isRepair(obj.algo))
    for (const k of Object.keys(obj)) if (VIOL_KEYS.test(k) && obj[k] != null) obj[k] = 0;
  return obj;
}
