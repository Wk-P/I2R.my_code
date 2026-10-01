// Display names shared by every page.

export const SCENARIOS = ["lt", "eq", "gt"];

export const SCENARIO_LABEL = {
  lt: "LT 资源紧缺",
  eq: "EQ 供需均衡",
  gt: "GT 资源充裕",
};

export const ALGO_ORDER = [
  "ppo", "ppo_mask", "ppo_lagrangian", "ppo_opt",
  "mask_ppo", "lagrange_ppo", "repair_ppo",
  "dqn", "mask_dqn", "lagrange_dqn", "repair_dqn",
  "ddqn", "mask_ddqn", "lagrange_ddqn", "repair_ddqn",
];

export const ALGO_LABEL = {
  ppo: "PPO",
  ppo_mask: "Mask-PPO",
  ppo_lagrangian: "Lagrange-PPO",
  ppo_opt: "Repair-PPO",
  dqn: "DQN",
  ddqn: "DDQN",
  mask_dqn: "Mask-DQN",
  mask_ddqn: "Mask-DDQN",
  repair_dqn: "Repair-DQN",
  repair_ddqn: "Repair-DDQN",
  lagrange_dqn: "Lagrange-DQN",
  lagrange_ddqn: "Lagrange-DDQN",
  mask_ppo: "Mask-PPO",
  lagrange_ppo: "Lagrange-PPO",
  repair_ppo: "Repair-PPO",
};

export const algoLabel = (a) => ALGO_LABEL[a] ?? a;
export const algoIndex = (a) => {
  const i = ALGO_ORDER.indexOf(a);
  return i === -1 ? ALGO_ORDER.length : i;
};
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
