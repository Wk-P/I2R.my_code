#!/usr/bin/env python3
"""Campaign driver (2026-10-17): 4 strategies x 3 scenarios x 3 learners x 3 seeds under the ar_pen reward.

Every run: structure-aware network on GPU, 20M steps, checkpoints 5M / 10M / 15M, 50k ILP-verified training pool,
reward "ar_pen" (terminal AR if all M placements are legal, else M_v/M - 1; no factor M; Lagrangian - lambda c_t
per step), gamma 1, full episode, EXIT action for Maskable only (dead end -> EXIT, M_v/M - 1)
descending demand; DQN / DDQN epsilon 1 -> 0 over a fixed 2M steps; Lagrangian = the v5.4.0 setting
(lambda init 0.001, Adam lr 0.035 on -lambda (J_C - d), d = 0, lambda >= 0 uncapped, J_C of the behaviour policy;
PPO one update per rollout, DQN / DDQN every 1000 episodes). The same flags as v5.4.0 (run_v5.py) otherwise.

Groups (one strategy x one seed = 9 runs: 3 scenarios x PPO / DQN / DDQN) run strictly one after the other:
    G1 maskable s1   G2 repair s1                                   (batch 1)
    G3 maskable s2   G4 maskable s3   G5 repair s2   G6 repair s3   (batch 2)
    G7 lagrangian s2 G8 lagrangian s3 G9 unconstraint s2 G10 unconstraint s3   (batch 3)
Inside a group: scenario -> GPU (EQ 0, GT 1, LT 2), at most --per-gpu runs per GPU (default 3), MPS on. A run is
started on a GPU only if that GPU's utilisation (mean of the last 60 s) and memory are both below 97 %, and at least
STAGGER seconds after the previous start there, so the load of the new run is seen before the next one starts.
Above 97 % while running: email (at most every 30 min per GPU), no further start on that GPU until it is below.

Supervision of every run: exit code; stall (no CPU time used for STALL seconds -> killed); after it ends, checks
(results.json: steps / reward / exit_action / mechanism / seed / pool; model + every checkpoint (+ ckpt_lambda.json,
lambda_updates.csv for Lagrangian); the eval line of the log) and a reload check (the saved model, re-evaluated on
the test split with the run's own settings, must reproduce the logged numbers, and on every violation-free instance
AR <= ILP AR). A failed run is emailed at once; when the group's runs have ended, rerun_campaign.retry() re-trains
the failed ones (new exp_id, up to MAX_RETRY times). Still failing -> email and stop (exit 1, the next group does
not start). Group done / batch done -> summary email. Driver killed or crashed -> email, its runs are stopped,
`--resume` continues (verified runs are kept).

    nohup .venv/bin/python src/scripts/run_campaign.py --batch <batch> --version <version> --groups G1,G2 \\
        > logs/<batch>/driver.log 2>&1 &
    .venv/bin/python src/scripts/run_campaign.py --smoke --batch campaign_smoke      # end-to-end test (see SMOKE)
    .venv/bin/python src/scripts/run_campaign.py --batch <batch> --version <version> --groups G1,G2 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))
PY, PY_GPU = str(ROOT / ".venv" / "bin" / "python"), str(ROOT / ".venv-gpu" / "bin" / "python")
DATA, POOL = "v4.3.1.4", "results/sup_diag/{scen}.json"

LEARNERS = ("ppo", "dqn", "ddqn")
SCENARIOS = ("eq", "gt", "lt")
GPU_OF = {"eq": 0, "gt": 1, "lt": 2}
ALGO = {"maskable": "mask_{l}", "repair": "repair_{l}", "lagrangian": "lagrange_{l}", "unconstraint": "{l}"}
GROUPS = {"G1": ("maskable", 1), "G2": ("repair", 1),
          "G3": ("maskable", 2), "G4": ("maskable", 3), "G5": ("repair", 2), "G6": ("repair", 3),
          "G7": ("lagrangian", 2), "G8": ("lagrangian", 3), "G9": ("unconstraint", 2), "G10": ("unconstraint", 3),
          "G11": ("lagrangian", 1)}                     # v6.0.1-lagrangian: seed 1 of the standard Lagrangian
LAGRANGE = {"lambda_max": "none", "lambda_update": "adam", "lambda_lr": 0.035, "lambda_init": 0.001,
            "lambda_target": 0.0, "lambda_cost": "behavior"}
FLAG = {"lambda_max": "--lambda-max", "lambda_update": "--lambda-update", "lambda_lr": "--lambda-lr",
        "lambda_cost": "--lambda-cost", "lambda_init": "--lambda-init", "lambda_target": "--lambda-target",
        "lambda_window": "--lambda-window", "eps_steps": "--eps-steps", "n_epochs": "--n-epochs", "target_kl": "--target-kl"}
EPS_STEPS = 2_000_000
GPU_LIMIT = 97.0
MAX_RETRY = 2
EVAL_RX = re.compile(r"eval: violation-free=([\d.]+) \| violation-free instances: AR=([\d.na]+) ILP AR=([\d.na]+) "
                     r"AR gap=([-\d.na]+) \| cap viol=([\d.]+) \| privacy viol=([\d.]+) \| EXIT=([\d.]+) \| train ([\d.]+) min")

# SMOKE: every strategy once (each with a different seed), real concurrency (3 per GPU), tiny budget, two injected
# faults on the first attempt (a killed run and a frozen run) that must be caught, emailed and re-trained.
SMOKE_GROUPS = {"S1": ("maskable", 1), "S2": ("repair", 2), "S3": ("lagrangian", 3), "S4": ("unconstraint", 1)}
SMOKE_INJECT = {"seed1 eq ppo_maskable": "kill", "seed2 gt dqn_repair": "freeze"}


class Cfg:
    def __init__(self, a):
        self.smoke = a.smoke
        self.batch, self.version = a.batch, a.version or a.batch
        self.groups_def = SMOKE_GROUPS if a.smoke else GROUPS
        self.groups = a.groups.split(",") if a.groups else list(self.groups_def)    # smoke: --groups picks a subset (S1-S4)
        bad = [g for g in self.groups if g not in self.groups_def]
        if bad:
            sys.exit(f"unknown group(s) {bad}; choose from {list(self.groups_def)}")
        self.steps = 61_440 if a.smoke else 20_000_000           # smoke: 3 PPO rollouts (40 envs x 512)
        self.ckpts = [20_480, 40_960] if a.smoke else [5_000_000, 10_000_000, 15_000_000]
        self.per_gpu = a.per_gpu
        self.stagger = 10 if a.smoke else 90
        self.poll = 5 if a.smoke else 30
        self.stall = 120 if a.smoke else 1200
        self.space = "smoke" if a.smoke else os.environ.get("RESULTS_SPACE", "unified")
        self.results = ROOT / "results" / self.space
        self.log_dir = ROOT / "logs" / self.batch
        self.inject = dict(SMOKE_INJECT) if a.smoke else {}
        self.mps = not a.no_mps
        self.label = a.label
        self.test_set = a.test_set                        # v6.0.4: fixed test set (None: the seed's 400-instance split)
        self.learners = tuple(a.learners.split(","))      # v6.0.0.1: PPO-only batches
        self.ppo_params = {k: (float(v) if "." in v else int(v)) for k, v in (x.split("=") for x in a.ppo_params.split(",") if x)}
        # v6.0.1-lagrangian: --lag-reward ar_obj --lag-target 0.001 = the standard Lagrangian (objective AR_obj, the
        # constraint only through -lambda c_t, cost limit d > 0); the defaults keep v6.0.0 (ar_pen - lambda c_t, d = 0)
        self.lag_reward = a.lag_reward
        self.lag = {**LAGRANGE, "lambda_target": a.lag_target}


def reward_of(C, strategy):
    return C.lag_reward if strategy == "lagrangian" else "ar_pen"


# ── plan ─────────────────────────────────────────────────────────────────────
def key_of(l, strategy):
    return f"{l}_{strategy}"


def own_params(C, l, strategy):
    p = dict(C.ppo_params) if l == "ppo" else {"eps_steps": EPS_STEPS}
    if strategy == "lagrangian":
        p.update(C.lag, **({"lambda_window": "rollout"} if l == "ppo" else {}))
    return p


def plan_runs(C, gid):
    strategy, seed = C.groups_def[gid]
    return [{"label": f"seed{seed} {s} {key_of(l, strategy)}", "group": gid, "seed": seed, "scenario": s, "learner": l,
             "strategy": strategy, "key": key_of(l, strategy), "algo": ALGO[strategy].format(l=l), "params": own_params(C, l, strategy)}
            for s in SCENARIOS for l in C.learners]


def train_cmd(C, r):
    cmd = [PY_GPU, "-u", "-m", "paper_rl.train", "--scen", r["scenario"], "--algo", r["algo"], "--net", "graph",
           "--reward", reward_of(C, r["strategy"]), "--reward-norm", "none", "--obs", "base", "--gamma", "1.0", "--full-episode",
           "--steps", str(C.steps), "--checkpoints", ",".join(map(str, C.ckpts)), "--seed", str(r["seed"]),
           "--device", "cuda", "--train-pool", POOL]
    if C.test_set:
        cmd += ["--test-set", C.test_set]
    if r["strategy"] == "maskable":                     # EXIT is Mask's own mechanism (as v5.4.0)
        cmd.append("--exit-action")
    for k, v in r["params"].items():
        cmd += [FLAG[k], f"{v:g}" if isinstance(v, float) else str(v)]
    return cmd


def commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def title(C, strategies, runs):
    """Manifest title from what the batch actually runs (strategies, learners, seeds, PPO flags, test set)."""
    seeds = sorted({x["seed"] for x in runs})
    word = "strategy" if len(strategies) == 1 else "strategies"
    flags = ", ".join(f"{k}={v}" for k, v in C.ppo_params.items())
    zh = [f"{len(strategies)} {word}（{' / '.join(strategies)}）× 3 场景 × {' / '.join(l.upper() for l in C.learners)}，种子 {', '.join(map(str, seeds))}"]
    en = [f"{len(strategies)} {word} ({' / '.join(strategies)}) x 3 scenarios x {' / '.join(l.upper() for l in C.learners)}, seeds {', '.join(map(str, seeds))}"]
    if flags:
        zh.append(f"PPO {flags}（其余同 v6.0.0）")
        en.append(f"PPO {flags} (otherwise as v6.0.0)")
    if C.test_set:
        zh.append(f"测试集 {C.test_set}")
        en.append(f"test set {C.test_set}")
    if strategies == ["lagrangian"] and C.lag_reward == "ar_obj":     # v6.0.1-lagrangian
        d = C.lag["lambda_target"]
        zh.append(f"标准 Lagrangian：终局 AR_obj（含违约放置、单 ECU 利用率截到 1）− 每步 λ·c_t，d = {d:g}")
        en.append(f"standard Lagrangian: terminal AR_obj (violating placements included, per-ECU utilisation clipped to 1) − λ·c_t per step, d = {d:g}")
    else:
        zh.append("ar_pen 奖励（终局：全部合法给 AR，否则 M_v/M−1，不乘 M" + ("；Lagrangian 再减 λ·c_t" if "lagrangian" in strategies else "") + "）")
        en.append("ar_pen reward (terminal AR if all legal, else M_v/M-1, no factor M" + ("; Lagrangian -λc_t" if "lagrangian" in strategies else "") + ")")
    if "maskable" in strategies:
        zh.append("EXIT 仅 Maskable")
        en.append("EXIT for Maskable only")
    if "unconstraint" in strategies:
        zh.append("unconstraint（v6.0.1 起）不罚分，终局给目标 AR（违约放置也计入，ECU 利用率截断到 1），违约照记，观测中无约束 / 违约信息")
        en.append("unconstraint (from v6.0.1) is not penalised: terminal objective AR (violating placements included, ECU utilisation clipped at 1), violations recorded, no constraint / violation information in its observation")
    zh.append("组间串行")
    en.append("groups in series")
    return {"zh": (f"{C.label}：" if C.label else "") + "；".join(zh), "en": (f"{C.label}: " if C.label else "") + "; ".join(en)}


def write_manifest(C, L):
    runs = []
    for gid in C.groups:
        for r in plan_runs(C, gid):
            st = L.state["runs"].get(r["label"], {})
            runs.append({"key": r["key"], "role": "main", "group": gid, "depends_on": [], "RL_algorithm": r["learner"].upper(),
                         "strategy": r["strategy"], "seed": r["seed"], "scenario": r["scenario"], "params": r["params"],
                         "dir": r["algo"], "exp_id": st.get("exp_id"), "attempts": [x.get("exp_id") for x in st.get("attempts", [])],
                         "status": st.get("status", "planned")})
    strategies = [s for s in ("unconstraint", "lagrangian", "maskable", "repair") if any(x["strategy"] == s for x in runs)]
    man = {"schema": 3, "batch": C.batch, "version": C.version, "purpose": "smoke" if C.smoke else "main", "commit": commit(),
           "results_space": C.space, "driver_pid": os.getpid(), "created_at": L.state.get("created_at"),
           "title": title(C, strategies, runs),
           "label": C.label,
           "RL_algorithm": [l.upper() for l in C.learners], "strategies": strategies,
           "seeds": sorted({x["seed"] for x in runs}), "scenarios": list(SCENARIOS),
           "groups": {g: {"strategy": C.groups_def[g][0], "seed": C.groups_def[g][1]} for g in C.groups},
           "params": {"steps": C.steps, "checkpoints": C.ckpts, "net": "graph", "reward": "ar_pen", "lagrangian_reward": C.lag_reward, "gamma": 1.0,
                      "full_episode": True, "exit_action": "maskable only", "train_pool": POOL, "test_set": C.test_set, "data": DATA, "eps_steps_dqn": EPS_STEPS,
                      "lagrangian": C.lag, "ppo_params": C.ppo_params, "per_gpu": C.per_gpu, "gpu_limit_pct": GPU_LIMIT, "gpu_of": GPU_OF},
           "runs": runs, "references": []}
    tmp = C.log_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    tmp.replace(C.log_dir / "manifest.json")


# ── GPU ──────────────────────────────────────────────────────────────────────
def gpu_stats():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        st = {}
        for line in out.strip().splitlines():
            m, u, mu, mt = [x.strip() for x in line.split(",")]
            st[int(m)] = {"util": float(u), "mem_used": float(mu), "mem_pct": 100.0 * float(mu) / float(mt)}
        return st or None
    except Exception:
        return None


def mps_env():
    d = ROOT / "logs" / "mps"
    return {"CUDA_MPS_PIPE_DIRECTORY": str(d / "pipe"), "CUDA_MPS_LOG_DIRECTORY": str(d / "log")}


def mps_start(L):
    env = mps_env()
    for v in env.values():
        Path(v).mkdir(parents=True, exist_ok=True)
    if (Path(env["CUDA_MPS_PIPE_DIRECTORY"]) / "control").exists() and subprocess.run(
            ["pgrep", "-u", str(os.getuid()), "-x", "nvidia-cuda-mps"], capture_output=True).returncode == 0:
        L.info("mps: daemon already running")
        return
    subprocess.run(["nvidia-cuda-mps-control", "-d"], env={**os.environ, **env}, check=True)
    L.info(f"mps: daemon started (pipe {env['CUDA_MPS_PIPE_DIRECTORY']})")


def mps_stop(L):
    subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True, env={**os.environ, **mps_env()})
    L.info("mps: daemon stopped")


def foreign_training():
    """Other paper_rl.train processes (not started by this driver) -- the campaign must not share the GPUs."""
    out = subprocess.run(["pgrep", "-f", "-a", "paper_rl.train"], capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


# ── one group ────────────────────────────────────────────────────────────────
class Supervisor:
    """Starts and watches runs; shared by the group pass and rerun_campaign.retry()."""

    def __init__(self, C, L):
        self.C, self.L = C, L
        self.active = {}                     # label -> dict(proc, fh, gpu, started, cpu, cpu_t, inject_done)
        self.samples = {g: deque() for g in GPU_OF.values()}   # (time, util)
        self.last_start = {}
        self.last_alert = {}
        self.peak = {}                       # gpu -> (max 60 s mean util, max mem %)

    # GPU bookkeeping
    def sample(self):
        st = gpu_stats()
        now = time.time()
        if st is None:
            self.L.event("gpu_error", "nvidia-smi 读取失败，暂停启动新 run")
            return None
        runs_on = {}
        for a in self.active.values():
            runs_on[a["gpu"]] = runs_on.get(a["gpu"], 0) + 1
        self.L.gpu_rows(st, runs_on)
        for g, s in st.items():
            q = self.samples.setdefault(g, deque())
            q.append((now, s["util"]))
            while q and now - q[0][0] > 60:
                q.popleft()
            mean = sum(u for _, u in q) / len(q)
            pk = self.peak.get(g, (0.0, 0.0))
            self.peak[g] = (max(pk[0], mean if len(q) >= 3 else 0.0), max(pk[1], s["mem_pct"]))
            over = (len(q) >= 3 and mean >= GPU_LIMIT) or s["mem_pct"] >= GPU_LIMIT
            if over and runs_on.get(g) and now - self.last_alert.get(g, 0) > 1800:
                self.last_alert[g] = now
                self.L.event("gpu_alert", f"GPU {g} 超过 {GPU_LIMIT:g}%：60 s 平均利用率 {mean:.0f}%、显存 {s['mem_pct']:.1f}%（{runs_on[g]} 个 run）",
                             gpu=g, util60=mean, mem_pct=s["mem_pct"])
                self.L.mail(f"GPU {g} 超过 {GPU_LIMIT:g}% 告警", [
                    f"GPU {g}：最近 60 秒平均利用率 {mean:.0f}%，显存占用 {s['mem_pct']:.1f}%，上面有 {runs_on[g]} 个 run。",
                    "这块 GPU 上不再启动新 run，直到降到阈值以下；已在跑的 run 不受影响。",
                    f"逐条记录：logs/{self.C.batch}/gpu_monitor.csv"])
        return st

    def gate(self, g, st):
        """May a new run start on GPU g now?"""
        if st is None or g not in st:
            return False, "nvidia-smi 不可用"
        if sum(1 for a in self.active.values() if a["gpu"] == g) >= self.C.per_gpu:
            return False, "已达每卡并发上限"
        if time.time() - self.last_start.get(g, 0) < self.C.stagger:
            return False, "等待上一个 run 的负载稳定"
        q = self.samples.get(g) or deque()
        mean = sum(u for _, u in q) / len(q) if q else 0.0
        if mean >= GPU_LIMIT or st[g]["mem_pct"] >= GPU_LIMIT:
            return False, f"GPU {g} 利用率 {mean:.0f}% / 显存 {st[g]['mem_pct']:.1f}% ≥ {GPU_LIMIT:g}%"
        return True, ""

    # one run
    def launch(self, r):
        from shared.paths import new_exp_id
        C, L = self.C, self.L
        exp_id = new_exp_id()
        log = C.log_dir / f"seed{r['seed']}_{r['scenario']}_{r['key']}_{exp_id}.log"
        g = GPU_OF[r["scenario"]]
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "EXP_ID": exp_id, "TRAIN_SEED": str(r["seed"]),
               "PAPER_VERSION": C.version, "DATA_VERSION": DATA, "RESULTS_SPACE": C.space,
               "RUN_BATCH": C.batch, "RUN_KEY": r["key"], "RUN_ROLE": "main", "RUN_GROUP": r["group"],
               "RUN_PARAMS": json.dumps(r["params"]), "CUDA_VISIBLE_DEVICES": str(g)}
        if C.mps:
            env.update(mps_env())
        fh = open(log, "w")
        proc = subprocess.Popen(train_cmd(C, r), cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        now = time.time()
        self.active[r["label"]] = {"r": r, "proc": proc, "fh": fh, "gpu": g, "started": now, "cpu": 0.0, "cpu_t": now,
                                   "log": log, "exp_id": exp_id, "inject": C.inject.pop(r["label"], None), "reason": None}
        self.last_start[g] = now
        st = L.run(r["label"])
        st.update(status="running", exp_id=exp_id, pid=proc.pid, gpu=g, log=str(log.relative_to(ROOT)), started=time.strftime("%Y-%m-%d %H:%M:%S"))
        st["attempts"].append({"exp_id": exp_id, "started": st["started"]})
        L.save()
        write_manifest(C, L)
        L.event("launch", f"{r['label']} exp_id={exp_id} pid={proc.pid} GPU {g}{' MPS' if C.mps else ''} (attempt {len(st['attempts'])})",
                label=r["label"], exp_id=exp_id, pid=proc.pid, gpu=g, attempt=len(st["attempts"]))

    def kill(self, a, why):
        try:
            os.killpg(a["proc"].pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        a["reason"] = why

    def watch(self):
        """Poll the active runs once; returns the labels that ended (verified or failed)."""
        from campaign_log import pid_cpu_seconds
        ended = []
        now = time.time()
        for label, a in list(self.active.items()):
            p = a["proc"]
            if a["inject"] and now - a["started"] > 30 and p.poll() is None:      # smoke: injected fault
                if a["inject"] == "kill":
                    self.L.event("inject", f"[smoke 故障注入] SIGKILL {label}", label=label)
                    os.killpg(p.pid, signal.SIGKILL)
                else:
                    self.L.event("inject", f"[smoke 故障注入] SIGSTOP {label}（模拟卡死）", label=label)
                    os.killpg(p.pid, signal.SIGSTOP)
                a["inject"] = None
            if p.poll() is None:
                cpu = pid_cpu_seconds(p.pid)
                if cpu is not None and cpu > a["cpu"] + 0.5:
                    a["cpu"], a["cpu_t"] = cpu, now
                elif now - a["cpu_t"] > self.C.stall:
                    self.L.event("stall", f"{label}：{self.C.stall} 秒没有使用 CPU，判定卡死并终止", label=label)
                    self.kill(a, f"卡死（{self.C.stall} 秒无 CPU 进展）")
                continue
            a["fh"].close()
            del self.active[label]
            self.finish(label, a, p.returncode)
            ended.append(label)
        return ended

    def finish(self, label, a, rc):
        L = self.L
        st = L.run(label)
        mins = (time.time() - a["started"]) / 60
        problems = [] if rc == 0 else [a["reason"] or f"进程退出码 {rc}"]
        ev = None
        if rc == 0:
            problems, ev = check_run(self.C, a["r"], a["exp_id"], a["log"])
        st["attempts"][-1].update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), rc=rc, minutes=round(mins, 1), problems=problems)
        st.update(status="ok" if not problems else "failed", rc=rc, minutes=round(mins, 1), problems=problems, eval=ev)
        L.save()
        write_manifest(self.C, L)
        if not problems:
            L.event("finish", f"{label} OK（{mins:.1f} min）", label=label, exp_id=a["exp_id"], eval=ev)
            return
        L.event("finish", f"{label} 失败：{'；'.join(problems)}", label=label, exp_id=a["exp_id"], rc=rc, problems=problems)
        from campaign_log import tail
        L.mail(f"run 失败：{label}", [f"run：{label}（exp_id {a['exp_id']}，第 {len(st['attempts'])} 次）",
                                       f"原因：{'；'.join(problems)}", f"已运行 {mins:.1f} 分钟，退出码 {rc}",
                                       f"日志：{a['log'].relative_to(ROOT)}",
                                       "本组其余 run 继续；组内全部结束后会自动重跑失败的 run（最多 "
                                       f"{MAX_RETRY} 次）。"], pre=tail(a["log"], 40))

    def drive(self, runs):
        """Start `runs` (respecting the gate) and wait until all of them have ended."""
        pending = list(runs)
        waits = {}
        while pending or self.active:
            st = self.sample()
            for g in sorted({GPU_OF[r["scenario"]] for r in pending}):
                ok, why = self.gate(g, st)
                nxt = next((r for r in pending if GPU_OF[r["scenario"]] == g), None)
                if ok and nxt:
                    pending.remove(nxt)
                    self.launch(nxt)
                elif nxt and why.startswith("GPU") and waits.get(g) != why:
                    waits[g] = why
                    self.L.event("gate_wait", f"{nxt['label']} 暂不启动：{why}", gpu=g)
            self.watch()
            time.sleep(self.C.poll)

    def stop_all(self, why):
        for label, a in list(self.active.items()):
            self.kill(a, why)
            st = self.L.run(label)
            st.update(status="interrupted", problems=[why])
            self.L.event("interrupted", f"{label} 已终止：{why}", label=label)
        self.L.save()


# ── checks ───────────────────────────────────────────────────────────────────
def parse_eval(log: Path):
    m = None
    for m in EVAL_RX.finditer(log.read_text(errors="replace")):
        pass
    if not m:
        return None
    f = lambda x: float("nan") if x == "nan" else float(x)
    vf, ar, ilp, gap, cap, priv, ex, mins = (f(x) for x in m.groups())
    return {"viol_free": vf, "ar": ar, "ilp_ar": ilp, "gap": gap, "cap": cap, "priv": priv, "exit": ex, "train_min": mins}


def check_run(C, r, exp_id, log: Path):
    """[] if the finished run is complete and consistent, else the list of problems; plus the eval numbers."""
    probs = []
    d = C.results / r["scenario"] / r["algo"] / exp_id
    try:
        res = json.loads((d / "results.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"results.json 缺失或损坏（{e.__class__.__name__}）"], None
    want = {"reward_mode": reward_of(C, r["strategy"]), "exit_action": r["strategy"] == "maskable", "full_episode": True, "seed": r["seed"], "algo": r["algo"],
            "scenario": r["scenario"], "net": "graph", "train_pool": POOL, "gamma": 1.0,
            "test_set": C.test_set}
    probs += [f"results.json {k}={res.get(k)!r}（应为 {v!r}）" for k, v in want.items() if res.get(k) != v]
    tr = res.get("training") or {}
    if tr.get("total_steps", 0) < C.steps:
        probs.append(f"训练步数 {tr.get('total_steps')} < {C.steps}")
    if r["strategy"] == "lagrangian":
        for k, v in C.lag.items():
            got = tr.get(k)
            if k == "lambda_max":
                ok = got == "none"
            else:
                ok = got == v
            if not ok:
                probs.append(f"Lagrangian 参数 {k}={got!r}（应为 {v!r}）")
    probs += [f"results.json {k}={res.get(k)!r}（应为 {v!r}）" for k, v in r["params"].items()
              if r["learner"] == "ppo" and k in C.ppo_params and res.get(k) != v]
    if r["learner"] != "ppo" and tr.get("eps_steps") != EPS_STEPS:
        probs.append(f"eps_steps={tr.get('eps_steps')}（应为 {EPS_STEPS}）")
    if not list(d.glob("model_*")):
        probs.append("缺少最终模型 model_*")
    for c in C.ckpts:
        if not (d / f"ckpt_{c}_{exp_id}.zip").exists():
            probs.append(f"缺少 checkpoint {c}")
    if r["strategy"] == "lagrangian":
        try:
            lam = json.loads((d / "ckpt_lambda.json").read_text())
            probs += [f"ckpt_lambda.json 缺 {c}" for c in C.ckpts if str(c) not in lam]
        except (OSError, json.JSONDecodeError):
            probs.append("缺少 ckpt_lambda.json")
        if not (d / "lambda_updates.csv").exists() or len((d / "lambda_updates.csv").read_text().splitlines()) < 2:
            probs.append("lambda_updates.csv 缺失或为空（λ 没有更新过）")
    ev = parse_eval(log)
    if ev is None:
        probs.append("日志里没有 eval 行")
        return probs, None
    if ev["exit"] != 0 and r["strategy"] != "maskable":
        probs.append(f"EXIT={ev['exit']}（只有 Maskable 能有 EXIT）")
    if ev["viol_free"] > 0 and ev["ar"] == ev["ar"] and ev["ar"] > ev["ilp_ar"] + 1e-6:
        probs.append(f"AR {ev['ar']} > ILP AR {ev['ilp_ar']}")
    if not probs:
        probs += reload_check(C, r, d, res, ev)
    return probs, ev


def reload_check(C, r, d, res, ev):
    """Reload the saved model on CPU, evaluate it with the run's own settings and compare with the logged eval."""
    cmd = [PY_GPU, "-c", RELOAD_SNIPPET, str(d), r["scenario"], str(r["seed"]), r["algo"], json.dumps(res.get("exit_action")),
           C.test_set or ""]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, timeout=3600)
    if p.returncode != 0:
        return [f"重载评估失败：{(p.stderr or p.stdout).strip().splitlines()[-1] if (p.stderr or p.stdout).strip() else p.returncode}"]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    probs = []
    if out["ar_over_ilp"]:
        probs.append(f"{out['ar_over_ilp']} 个无违约实例 AR > ILP AR")
    tol = 1.5 / out["n"]                             # one instance may flip (CPU vs GPU float rounding)
    for k in ("viol_free", "cap", "priv"):
        if abs(out[k] - ev[k]) > tol:
            probs.append(f"重载评估 {k}={out[k]:.4f} 与训练日志 {ev[k]:.4f} 不一致")
    if out["viol_free"] > 0 and ev["viol_free"] > 0 and abs(out["gap"] - ev["gap"]) > 0.002:
        probs.append(f"重载评估 AR gap={out['gap']:.4f} 与训练日志 {ev['gap']:.4f} 不一致")
    ev["reload"] = out
    return probs


RELOAD_SNIPPET = r"""
import json, sys
import numpy as np, torch
torch.set_num_threads(4)
sys.path.insert(0, "src")
from pathlib import Path
from paper_rl.train import evaluate, model_class, split_algo, split_instances
d, scen, seed, algo, exit_action = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4], json.loads(sys.argv[5])
mech, learner = split_algo(algo)
res = json.loads((d / "results.json").read_text())
tr = res["training"]
lam = tr.get("final_lambda", 0.0) if mech == "lagrange" else 0.0
lm = tr.get("lambda_max", 50.0)
lm = float("inf") if lm == "none" else float(lm)
model = model_class(learner, mech).load(str(next(d.glob("model_*"))), device="cpu")
test = json.loads(Path(sys.argv[6].format(scen=scen)).read_text()) if sys.argv[6] else split_instances(scen, seed)[2]
ev = evaluate(model, test, mech, res["reward_mode"], lam, learner, "raw", True, exit_action, lam_max=lm)
ok = [e for e in ev if e["success"]]
ar = float(np.mean([e["ar"] for e in ok])) if ok else float("nan")
ilp = float(np.mean([e["ar_star"] for e in ok])) if ok else float("nan")
print(json.dumps({"n": len(ev), "viol_free": len(ok) / len(ev), "gap": ilp - ar if ok else float("nan"),
                  "cap": float(np.mean([e["cap_v"] > 0 for e in ev])), "priv": float(np.mean([e["conf_v"] > 0 for e in ev])),
                  "ar_over_ilp": int(sum(e["ar"] > e["ar_star"] + 1e-6 for e in ok))}))
"""


# ── summaries ────────────────────────────────────────────────────────────────
def pc(x):
    return "—" if x is None or x != x else f"{100 * x:.2f}%"


def group_table(C, L, gid):
    rows = []
    for r in plan_runs(C, gid):
        st = L.state["runs"].get(r["label"], {})
        ev = st.get("eval") or {}
        rows.append([r["scenario"].upper(), r["learner"].upper(), st.get("status", "—"), pc(ev.get("viol_free")),
                     f"{ev['ar']:.4f}" if ev.get("ar") == ev.get("ar") and ev.get("ar") is not None else "—",
                     f"{ev['ilp_ar']:.4f}" if ev.get("ilp_ar") == ev.get("ilp_ar") and ev.get("ilp_ar") is not None else "—",
                     pc(ev.get("gap")), pc(ev.get("cap")), pc(ev.get("priv")), f"{st.get('minutes', 0):.0f}",
                     len(st.get("attempts", [])), st.get("exp_id", "—")])
    return ["场景", "模型", "状态", "无违约实例比例", "AR", "ILP AR", "AR gap", "cap 违约率", "privacy 违约率", "分钟", "尝试次数", "exp_id"], rows


def run_groups(C, L, resume, sup_ref):
    from rerun_campaign import retry
    sup = sup_ref["sup"] = Supervisor(C, L)
    if C.mps:
        mps_start(L)
    try:
        for gid in C.groups:
            gst = L.group(gid)
            runs = [r for r in plan_runs(C, gid) if not (resume and L.state["runs"].get(r["label"], {}).get("status") == "ok")]
            if not runs:
                L.info(f"=== {gid}：全部 run 已通过检查，跳过 ===")
                continue
            strategy, seed = C.groups_def[gid]
            t0 = time.time()
            sup.peak = {}
            gst.update(status="running", started=time.strftime("%Y-%m-%d %H:%M:%S"))
            L.save()
            L.event("group_start", f"=== {gid}（{strategy}、种子 {seed}）：{len(runs)} 个 run，每卡最多 {C.per_gpu} 个 ===", group=gid)
            L.mail(f"{gid} 开始（{strategy}，种子 {seed}）", [
                f"{gid}：{strategy} × 3 场景 × PPO / DQN / DDQN，种子 {seed}，{len(runs)} 个 run。",
                f"每卡最多 {C.per_gpu} 个（EQ→GPU0、GT→GPU1、LT→GPU2），启动间隔 {C.stagger} 秒，GPU 利用率或显存 ≥ {GPU_LIMIT:g}% 时不启动新 run。",
                f"步数 {C.steps:,}，checkpoint {', '.join(f'{c:,}' for c in C.ckpts)}；ar_pen 奖励（不乘 M），EXIT 仅 Maskable。"])
            sup.drive(runs)
            failed = [r for r in plan_runs(C, gid) if L.state["runs"].get(r["label"], {}).get("status") != "ok"]
            if failed:
                failed = retry(C, L, sup, failed)
            mins = (time.time() - t0) / 60
            h, rows = group_table(C, L, gid)
            peak = [[g, f"{u:.0f}%", f"{m:.1f}%"] for g, (u, m) in sorted(sup.peak.items())]
            gst.update(status="failed" if failed else "done", finished=time.strftime("%Y-%m-%d %H:%M:%S"), minutes=round(mins, 1),
                       gpu_peak={str(g): {"util60": u, "mem_pct": m} for g, (u, m) in sup.peak.items()})
            L.save()
            if failed:
                L.event("group_failed", f"{gid}：{len(failed)} 个 run 重跑后仍失败，停止（后续组不启动）", group=gid)
                L.mail(f"{gid} 有 run 重跑 {MAX_RETRY} 次仍失败，已停止", [
                    f"仍失败：{', '.join(r['label'] for r in failed)}。后续组不会启动。",
                    f"处理后执行：run_campaign.py ... --groups {','.join(C.groups[C.groups.index(gid):])} --resume",
                    "（或手动：rerun_campaign.py --batch ... --label \"seed1 eq ppo_maskable\"）"],
                       tables=[(f"{gid} 结果", h, rows), ("GPU 峰值（60 秒平均利用率 / 显存）", ["GPU", "利用率", "显存"], peak)])
                return False
            L.event("group_done", f"=== {gid} 完成（{mins:.0f} 分钟）===", group=gid)
            L.mail(f"{gid} 完成（{strategy}，种子 {seed}，{mins:.0f} 分钟）", [f"{gid} 的 {len(rows)} 个 run 全部通过检查。"],
                   tables=[(f"{gid} 结果", h, rows), ("GPU 峰值（60 秒平均利用率 / 显存）", ["GPU", "利用率", "显存"], peak)])
        return True
    finally:
        if C.mps:
            mps_stop(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="batch name = logs/<batch>/ (a smoke batch ends with _smoke)")
    ap.add_argument("--version", default=None, help="PAPER_VERSION written into every run (default: the batch name)")
    ap.add_argument("--groups", default=None, help="comma-separated, run in this order, e.g. G1,G2")
    ap.add_argument("--per-gpu", type=int, default=3)
    ap.add_argument("--label", default=None, help="short batch label, prefixed to the manifest title")
    ap.add_argument("--resume", action="store_true", help="keep runs that already passed the checks")
    ap.add_argument("--test-set", default=None, help="v6.0.4: fixed test set, e.g. data/testset_v6.0.4/{scen}.json")
    ap.add_argument("--learners", default=",".join(LEARNERS), help="v6.0.0.1: comma-separated subset of ppo,dqn,ddqn")
    ap.add_argument("--ppo-params", default="", help="v6.0.0.1: extra PPO flags, e.g. target_kl=0.02 or n_epochs=4")
    ap.add_argument("--lag-reward", default="ar_pen", choices=["ar_pen", "ar_obj"],
                    help="v6.0.1-lagrangian: reward of the Lagrangian runs (ar_obj = standard Lagrangian, objective AR_obj)")
    ap.add_argument("--lag-target", type=float, default=0.0, help="v6.0.1-lagrangian: cost limit d of the Lagrangian runs")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-mps", action="store_true")
    ap.add_argument("--allow-busy", action="store_true", help="start although other paper_rl.train processes exist")
    a = ap.parse_args()
    if a.smoke != a.batch.endswith("_smoke"):
        sys.exit("a smoke batch must be named *_smoke (and only a smoke batch)")
    if not a.smoke and not a.groups:
        sys.exit("--groups is required")
    C = Cfg(a)
    os.environ["RESULTS_SPACE"] = C.space            # before shared.paths is imported (new_exp_id, results_dir)
    if C.log_dir.exists() and (C.log_dir / "state.json").exists() and not a.resume:
        sys.exit(f"logs/{C.batch}/ already has a state.json: use --resume, or a new batch name")
    from campaign_log import CampaignLog
    L = CampaignLog(C.log_dir, C.batch, C.smoke)
    busy = foreign_training()
    if busy and not a.allow_busy:
        L.event("refused", f"已有 {len(busy)} 个 paper_rl.train 进程在运行，按排队规则不启动：\n" + "\n".join(busy))
        sys.exit(2)
    L.state.update(version=C.version, groups=L.state.get("groups", {}), driver_pid=os.getpid(), commit=commit(),
                   plan=C.groups, per_gpu=C.per_gpu, steps=C.steps)
    L.save()
    write_manifest(C, L)
    L.event("start", f"=== {C.batch} 开始 | 组 {', '.join(C.groups)} | 每卡 {C.per_gpu} | commit {commit()} | "
                     f"results/{C.space}/ | {'resume' if a.resume else 'new'} ===")
    sup_ref = {}

    def on_signal(sig, _frm):
        L.event("signal", f"驱动收到信号 {signal.Signals(sig).name}，终止本组 run 并退出")
        if "sup" in sup_ref:
            sup_ref["sup"].stop_all(f"驱动收到 {signal.Signals(sig).name}")
        L.mail(f"驱动被终止（{signal.Signals(sig).name}）", ["正在运行的 run 已终止并记为 interrupted；",
                                                             f"继续：run_campaign.py --batch {C.batch} --version {C.version} --groups {','.join(C.groups)} --resume"])
        sys.exit(3)
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, on_signal)
    t0 = time.time()
    try:
        ok = run_groups(C, L, a.resume, sup_ref)
    except Exception:
        tb = traceback.format_exc()
        L.event("error", "驱动异常：\n" + tb)
        if "sup" in sup_ref:
            sup_ref["sup"].stop_all("驱动异常")
        L.mail("驱动异常退出", ["驱动抛出异常，正在运行的 run 已终止。修复后用 --resume 继续。"], pre=tb)
        sys.exit(4)
    hours = (time.time() - t0) / 3600
    tables = [(f"{g}（{C.groups_def[g][0]}，种子 {C.groups_def[g][1]}）", *group_table(C, L, g)) for g in C.groups]
    if ok:
        L.event("done", f"=== {C.batch} 全部完成（{hours:.1f} 小时）===")
        if C.smoke:
            smoke_summary(C, L, hours, tables)
        else:
            L.mail(f"批次完成：{', '.join(C.groups)}（{hours:.1f} 小时）", ["所有组的 run 都通过检查。"], tables=tables)
    sys.exit(0 if ok else 1)


def smoke_summary(C, L, hours, tables):
    evs = [json.loads(x) for x in (C.log_dir / "events.jsonl").read_text().splitlines()]
    inj = [e for e in evs if e["kind"] == "inject"]
    caught = {e["label"]: e for e in evs if e["kind"] == "finish" and e.get("problems")}
    retried_ok = [lab for lab in SMOKE_INJECT if L.state["runs"].get(lab, {}).get("status") == "ok"
                  and len(L.state["runs"][lab]["attempts"]) >= 2]
    lines = [f"smoke 结束：{len(C.groups)} 组 × 9 个 run（4 种 strategy × 3 场景 × PPO / DQN / DDQN），{C.steps:,} 步，用时 {hours * 60:.0f} 分钟。",
             f"故障注入：{len(inj)} 次（{', '.join(e['label'] for e in inj)}）；被检出 {sum(1 for k in SMOKE_INJECT if k in caught)} 个；"
             f"自动重跑后通过 {len(retried_ok)} 个。",
             "所有 run 都通过了：退出码、results.json 设置（ar_pen / EXIT 仅 Maskable / 步数 / 种子 / 训练池 / Lagrangian 参数 / ε）、"
             "最终模型与全部 checkpoint、eval 行、重载评估一致性、无违约实例 AR ≤ ILP AR。"]
    peaks = []
    for g in C.groups:
        for gpu, v in (L.state["groups"].get(g, {}).get("gpu_peak") or {}).items():
            peaks.append([g, gpu, f"{v['util60']:.0f}%", f"{v['mem_pct']:.1f}%"])
    tables = tables + [("各组 GPU 峰值（60 秒平均利用率 / 显存，每卡 3 个 run）", ["组", "GPU", "利用率", "显存"], peaks)]
    L.mail("smoke 结束：全部通过", lines, tables=tables)


if __name__ == "__main__":
    main()
