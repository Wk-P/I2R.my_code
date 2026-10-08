#!/usr/bin/env python3
"""v4.3.1.2 reward pilot: 12 models x 3 rewards at 1M steps.

    REWARD_MODE  legacy | ar | directional   (see src/shared/reward_config.py)
    algos        {PPO, DQN, DDQN} x {unconstrained, Mask, Lagrange, Repair}
    scenarios    lt, eq, gt
    seed         1

108 jobs, all at the same 1M budget. `ratio` is dropped: in the v4.3.1 pilot
it was indistinguishable from `ar` and needs the ILP optimum at training time.
Purpose: pick the reward for the 5M x 3-seed campaign of the 12 models.

Results: results/final_paper_experiments/<scen>/<algo>/<exp_id>/; manifest
logs/v4.3.1.2_reward_pilot/manifest.json; report
paper_contents/v4.3.1/v4.3.1.2/reward_pilot_report.md.

Usage:
    nohup .venv/bin/python src/scripts/run_v4.3.1.2_reward_pilot.py > logs/run_v4.3.1.2_reward_pilot_driver.log 2>&1 &
    # resume after the driver was stopped (training processes keep running):
    nohup .venv/bin/python src/scripts/run_v4.3.1.2_reward_pilot.py --resume >> logs/run_v4.3.1.2_reward_pilot_driver.log 2>&1 &
"""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PY = str(PROJECT_ROOT / ".venv" / "bin" / "python")
LOG_DIR = PROJECT_ROOT / "logs" / "v4.3.1.2_reward_pilot"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = LOG_DIR / "manifest.json"
REPORT_PATH = PROJECT_ROOT / "paper_contents" / "v4.3.1" / "v4.3.1.2" / "reward_pilot_report.md"
RESULTS_ROOT = PROJECT_ROOT / "results" / "final_paper_experiments"

CAPACITY = int(56 * 0.93)
# Measured cores per job (ps %CPU): PPO / Mask-PPO ~6, DQN ~2 -- the old
# 10 / 5 reservations left ~15 of 56 cores idle.
WEIGHT = {a: (7 if a.startswith("ppo") else 3) for a in [
    "ppo", "ppo_mask", "ppo_lagrangian", "ppo_opt",
    "dqn", "mask_dqn", "lagrange_dqn", "repair_dqn",
    "ddqn", "mask_ddqn", "lagrange_ddqn", "repair_ddqn"]}
MODES = ["legacy", "ar", "directional"]
SCENARIOS = ["lt", "eq", "gt"]
ALGOS = list(WEIGHT)
MECH = {"": "无约束", "mask": "Mask", "lagrange": "Lagrange", "repair": "Repair"}
LEARNER = {"ppo": "PPO", "dqn": "DQN", "ddqn": "DDQN"}
PPO_MECH = {"ppo": "", "ppo_mask": "mask", "ppo_lagrangian": "lagrange", "ppo_opt": "repair"}


def split(algo):
    """algo dir -> (mechanism key, learner key)"""
    if algo in PPO_MECH:
        return PPO_MECH[algo], "ppo"
    mech, _, learner = algo.rpartition("_")
    return mech, learner


LABELS = {a: (f"{MECH[split(a)[0]]}-" if split(a)[0] else "") + LEARNER[split(a)[1]] for a in ALGOS}
SEED = 1
STEPS = 1_000_000


def new_exp_id() -> str:
    out = subprocess.run([PY, "-c", "from shared.paths import new_exp_id; print(new_exp_id())"],
                         cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def write_manifest(exp_ids: dict) -> None:
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True).stdout.strip()
    tmp.write_text(json.dumps({"version": "4.3.1.2-pilot", "commit": commit, "steps": STEPS, "seed": SEED, "modes": MODES,
                               "scenarios": SCENARIOS, "algos": ALGOS, "exp_ids": exp_ids}, indent=2))
    tmp.replace(MANIFEST_PATH)


def launch(mode, scen, algo, exp_ids):
    exp_id = new_exp_id()
    log_f = open(LOG_DIR / f"{mode}_{scen}_{algo}_{exp_id}.log", "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TRAIN_SEED": str(SEED), "EXP_ID": exp_id,
           "REWARD_MODE": mode, "PAPER_VERSION": f"4.3.1.2-pilot-{mode}"}
    proc = subprocess.Popen(
        [PY, "-u", f"scenarios/{scen}/{algo}/run_all.py", "--total-timesteps", str(STEPS)],
        cwd=PROJECT_ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    exp_ids.setdefault(mode, {}).setdefault(scen, {})[algo] = exp_id
    write_manifest(exp_ids)
    print(f"[{time.strftime('%H:%M:%S')}] launched {mode} {scen}/{algo} exp_id={exp_id} pid={proc.pid}",
          flush=True)
    return proc, log_f


def summary_row(scen, algo, exp_id):
    path = RESULTS_ROOT / scen / algo / exp_id / "summary.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    return rows[1] if len(rows) > 1 else None


def write_report(exp_ids: dict) -> None:
    def cell(mode, scen, algo, key, f):
        r = summary_row(scen, algo, exp_ids.get(mode, {}).get(scen, {}).get(algo, ""))
        return f(float(r[key])) if r else "—"

    pct = lambda v: f"{v:.3f}"
    lines = ["# v4.3.1.2 奖励试跑报告（12 个模型 × 3 种奖励，1M 步，种子 1）", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             "- 奖励：legacy = 无违规 M(2AR−1)；ar = 无违规 M·AR；有违规时两者均为 −M(1−valid/M)；"
             "directional = 逐步 ΔAR 方向奖励 + 完成奖励 B − 死局惩罚 C（见 README）。",
             "- 指标为训练脚本内置的测试集单次确定性评估（400 个实例）。success_rate = M 个服务全部合法放置且无违规的实例比例；"
             "AR = average resource utilization，全部测试 episode 的均值（违规 episode 的 AR 各环境口径不同，无约束 / Lagrange 方法的 AR 仅供参考）。",
             "- manifest：`logs/v4.3.1.2_reward_pilot/manifest.json`", ""]
    order = sorted(ALGOS, key=lambda a: (["", "mask", "lagrange", "repair"].index(split(a)[0]),
                                         ["ppo", "dqn", "ddqn"].index(split(a)[1])))
    for scen in SCENARIOS:
        lines += [f"## {scen.upper()}", "",
                  "| 约束机制 | 算法 | success_rate legacy | ar | directional | AR legacy | ar | directional | 冲突违规率 legacy | ar | directional |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for algo in order:
            mech, learner = split(algo)
            row = [MECH[mech], LEARNER[learner]]
            row += [cell(m, scen, algo, "success_rate", pct) for m in MODES]
            row += [cell(m, scen, algo, "ar_mean", lambda v: f"{v:.4f}") for m in MODES]
            row += [cell(m, scen, algo, "conflict_viol_rate", pct) for m in MODES]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"report -> {REPORT_PATH}", flush=True)


def main():
    for scen in SCENARIOS:
        p = RESULTS_ROOT / scen / "ilp" / "ar_star.json"
        pass  # ar / directional do not need AR*
    queue = [(m, s, a) for s in SCENARIOS for a in ALGOS for m in MODES]
    exp_ids: dict = {}
    adopted = []  # (pid, label, weight) of runs launched by a previous driver, still alive
    if "--resume" in sys.argv and MANIFEST_PATH.exists():
        exp_ids = json.loads(MANIFEST_PATH.read_text())["exp_ids"]
        launched = {(m, s, a): eid for m, d in exp_ids.items() for s, d2 in d.items() for a, eid in d2.items()}
        queue = [job for job in queue if job not in launched]
        live = {}
        for pd in Path("/proc").iterdir():
            if not pd.name.isdigit():
                continue
            try:
                env = dict(kv.split(b"=", 1) for kv in (pd / "environ").read_bytes().split(b"\0") if b"=" in kv)
                if b"run_all.py" in (pd / "cmdline").read_bytes():
                    live[env.get(b"EXP_ID", b"").decode()] = int(pd.name)
            except OSError:
                continue
        for (m, s, a), eid in launched.items():
            if eid in live:
                adopted.append((live[eid], f"{m} {s}/{a}", WEIGHT[a]))
        print(f"=== resume: {len(launched)} launched before, {len(adopted)} still running, "
              f"{len(queue)} left to launch ===", flush=True)
    print(f"=== v4.3.1.2 reward pilot start {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(queue)} jobs, "
          f"steps={STEPS:,} ===", flush=True)
    active, failed, done = [], [], 0

    def pid_alive(pid):
        try:
            os.kill(pid, 0)
            return Path(f"/proc/{pid}").exists() and b"run_all.py" in Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return False

    while queue or active or adopted:
        adopted = [x for x in adopted if pid_alive(x[0]) or print(
            f"[{time.strftime('%H:%M:%S')}] finished {x[1]} (adopted)", flush=True)]
        load = sum(w for *_, w in active) + sum(w for *_, w in adopted)
        i = 0
        while i < len(queue):
            m, s, a = queue[i]
            if load + WEIGHT[a] <= CAPACITY:
                queue.pop(i)
                proc, log_f = launch(m, s, a, exp_ids)
                active.append((proc, log_f, f"{m} {s}/{a}", WEIGHT[a]))
                load += WEIGHT[a]
            else:
                i += 1
        still = []
        for proc, log_f, label, w in active:
            ret = proc.poll()
            if ret is None:
                still.append((proc, log_f, label, w))
                continue
            log_f.close()
            done += 1
            print(f"[{time.strftime('%H:%M:%S')}] finished {label} ({'OK' if ret == 0 else f'EXIT={ret}'}) "
                  f"— {done} done, {len(queue)} queued", flush=True)
            if ret != 0:
                failed.append(label)
        active = still
        if queue or active or adopted:
            time.sleep(15)
    print(f"=== pilot done {time.strftime('%Y-%m-%d %H:%M:%S')} — {done} completed, {len(failed)} failed ===",
          flush=True)
    if failed:
        print("FAILED:", failed, flush=True)
    write_report(exp_ids)


if __name__ == "__main__":
    main()
