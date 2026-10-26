#!/usr/bin/env python3
"""v6.1.0: Optuna hyper-parameter search, one study per learner x strategy (12), shared by the three scenarios.

Trial: the sampled constructor arguments (train.py --hp) are trained on LT / EQ / GT at once (scenario -> GPU as in
run_campaign), --steps each, with the campaign's settings (graph net, ar_pen, gamma 1, full episode, 50k pool, EXIT for
Maskable, Lagrangian dual settings of v6.0.0). Every --val-every steps each run evaluates on the validation set
data/valset_v6.1.0 (never the test set) and the driver reports
    value = mean over scenarios of (relative AR gap on violation-free instances + fail rate)
to Optuna (TPE sampler, median pruner); the final value is the same quantity at the end of training. Every study gets
the same budget (--trials) and the same pruner. Lower is better.

    nohup .venv/bin/python src/scripts/tune_v6_1_0.py > logs/v6.1.0/driver.log 2>&1 &
    .venv/bin/python src/scripts/tune_v6_1_0.py --smoke                          # 2 trials x 2 studies, tiny budget
    .venv/bin/optuna-dashboard sqlite:///results/tune/v6.1.0.db --host 0.0.0.0 --port 8082
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))
from run_campaign import ALGO, FLAG, GPU_OF, LAGRANGE, POOL, PY_GPU, SCENARIOS, mps_env   # noqa: E402
from shared.paths import new_exp_id                                                    # noqa: E402

VERSION = "v6.1.0"
VAL_SET = "data/valset_v6.1.0/{scen}.json"
STRATEGIES = ("maskable", "repair", "lagrangian", "unconstraint")
LEARNERS = ("ppo", "dqn", "ddqn")


def sample(trial: optuna.Trial, learner: str) -> tuple[dict, dict]:
    """(constructor arguments for --hp, extra train.py flags)."""
    if learner == "ppo":
        return {"learning_rate": trial.suggest_float("learning_rate", 3e-5, 1e-3, log=True),
                "n_epochs": trial.suggest_int("n_epochs", 2, 10),
                "clip_range": trial.suggest_float("clip_range", 0.05, 0.3),
                "ent_coef": trial.suggest_float("ent_coef", 1e-4, 2e-2, log=True),
                "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
                "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
                "n_steps": trial.suggest_categorical("n_steps", [256, 512, 1024])}, {}
    return {"learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
            "buffer_size": trial.suggest_categorical("buffer_size", [100_000, 300_000, 1_000_000]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "target_update_interval": trial.suggest_categorical("target_update_interval", [250, 500, 1000, 2000, 5000]),
            "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
            "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 2])}, \
        {"eps_steps": trial.suggest_int("eps_steps", 200_000, 3_000_000, step=100_000)}


def cmd(scen, learner, strategy, hp, extra, steps, val_every, seed):
    c = [PY_GPU, "-u", "-m", "paper_rl.train", "--scen", scen, "--algo", ALGO[strategy].format(l=learner), "--net", "graph",
         "--reward", "ar_pen", "--reward-norm", "none", "--obs", "base", "--gamma", "1.0", "--full-episode",
         "--steps", str(steps), "--seed", str(seed), "--device", "cuda", "--train-pool", POOL,
         "--test-set", VAL_SET, "--val-every", str(val_every), "--hp", json.dumps(hp)]
    if strategy == "maskable":
        c.append("--exit-action")
    params = dict(extra)
    if strategy == "lagrangian":
        params.update(LAGRANGE, **({"lambda_window": "rollout"} if learner == "ppo" else {}))
    for k, v in params.items():
        c += [FLAG[k], f"{v:g}" if isinstance(v, float) else str(v)]
    return c


def value(rows: list[dict]) -> float:
    return sum(r["rel_gap"] + r["fail"] for r in rows) / len(rows)


def read_jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(e) for e in path.read_text().splitlines() if e.strip()]
    except OSError:
        return []


def make_objective(a, learner, strategy, log_dir: Path):
    algo = ALGO[strategy].format(l=learner)

    def objective(trial: optuna.Trial) -> float:
        hp, extra = sample(trial, learner)
        procs, dirs = {}, {}
        for scen in SCENARIOS:
            eid = new_exp_id()
            dirs[scen] = ROOT / "results" / a.space / scen / algo / eid
            env = {**os.environ, **mps_env(), "EXP_ID": eid, "RESULTS_SPACE": a.space, "PAPER_VERSION": VERSION.lstrip("v"),
                   "CUDA_VISIBLE_DEVICES": str(GPU_OF[scen]), "RUN_BATCH": VERSION, "RUN_ROLE": "tune",
                   "RUN_KEY": f"{learner}_{strategy}", "RUN_GROUP": f"trial{trial.number}",
                   "RUN_PARAMS": json.dumps({"trial": trial.number, **hp, **extra})}
            log = open(log_dir / f"{algo}_t{trial.number}_{scen}_{eid}.log", "w")
            procs[scen] = subprocess.Popen(cmd(scen, learner, strategy, hp, extra, a.steps, a.val_every, a.seed),
                                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        trial.set_user_attr("exp_ids", {s: d.name for s, d in dirs.items()})
        reported = 0
        try:
            while any(p.poll() is None for p in procs.values()):
                time.sleep(a.poll)
                curves = {s: read_jsonl(d / "val_curve.jsonl") for s, d in dirs.items()}
                while all(len(c) > reported for c in curves.values()):
                    rows = [c[reported] for c in curves.values()]
                    trial.report(value(rows), rows[0]["step"])
                    reported += 1
                    if trial.should_prune():
                        trial.set_user_attr("pruned_at", rows[0]["step"])
                        raise optuna.TrialPruned()
            bad = [s for s, p in procs.items() if p.returncode != 0]
            if bad:
                raise RuntimeError(f"train.py failed on {bad} (logs in {log_dir})")
            res = {s: json.loads((d / "results.json").read_text())["val"] for s, d in dirs.items()}
            for s, r in res.items():
                trial.set_user_attr(s, r)
            return value(list(res.values()))
        finally:
            for p in procs.values():
                if p.poll() is None:
                    p.kill()
                    p.wait()

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--learners", default=",".join(LEARNERS))
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--trials", type=int, default=20, help="same budget for every study")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--val-every", type=int, default=500_000)
    ap.add_argument("--parallel", type=int, default=3, help="concurrent trials (each uses all three GPUs)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    name = VERSION + ("-smoke" if a.smoke else "")
    if a.smoke:
        a.learners, a.strategies, a.trials, a.steps, a.val_every, a.parallel = "ppo,dqn", "maskable", 2, 61_440, 20_480, 2
    a.space, a.poll = ("tune_smoke" if a.smoke else "tune"), (5 if a.smoke else 30)
    log_dir = ROOT / "logs" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    for v in mps_env().values():
        Path(v).mkdir(parents=True, exist_ok=True)
    if subprocess.run(["pgrep", "-u", str(os.getuid()), "-x", "nvidia-cuda-mps"], capture_output=True).returncode != 0:
        subprocess.run(["nvidia-cuda-mps-control", "-d"], env={**os.environ, **mps_env()}, check=True)
    storage = f"sqlite:///{ROOT / 'results' / 'tune' / (name + '.db')}"
    for learner in a.learners.split(","):
        for strategy in a.strategies.split(","):
            study = optuna.create_study(study_name=f"{name}/{learner}_{strategy}", storage=storage, load_if_exists=True,
                                        direction="minimize", sampler=optuna.samplers.TPESampler(seed=a.seed, multivariate=True),
                                        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=0))
            study.set_user_attr("search", {"steps": a.steps, "val_every": a.val_every, "trials": a.trials,
                                           "objective": "mean_scen(rel_gap + fail) on data/valset_v6.1.0"})
            left = a.trials - len([t for t in study.trials if t.state.is_finished()])
            print(f"=== {study.study_name}: {left} trials to run ===", flush=True)
            if left > 0:
                study.optimize(make_objective(a, learner, strategy, log_dir), n_trials=left, n_jobs=a.parallel,
                               catch=(RuntimeError,), gc_after_trial=True)
            done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if done:
                print(f"    best {study.best_value:.4f}: {study.best_params}", flush=True)


TITLE = {"zh": "v6.1.0：Optuna 调参，12 个 study（PPO / DQN / DDQN × 4 strategies），三场景共用一套超参，每个 study 20 次试验 × 2M 步，"
                "目标 = 三场景验证集（data/valset_v6.1.0）相对 gap + 失败率的平均",
         "en": "v6.1.0: Optuna search, 12 studies (PPO / DQN / DDQN x 4 strategies), one setting shared by the three scenarios, "
               "20 trials x 2M steps per study, objective = mean over scenarios of relative gap + fail rate on data/valset_v6.1.0"}
LINKS = [{"zh": "Optuna 调参看板", "en": "Optuna dashboard", "url": "/optuna/dashboard/"}]


if __name__ == "__main__":
    from shared.batch_manifest import nontrain_batch      # docs/data_spec.md §6 (purpose tune, v6.1.0)
    smoke = "--smoke" in sys.argv
    try:
        with nontrain_batch(VERSION + ("-smoke" if smoke else ""), "tune", __file__, TITLE, depends_on=["v6.1.0-valset"],
                            report_dir=None, version=VERSION, inputs=["results/sup_diag/{scen}.json", VAL_SET],
                            outputs=["results/tune/" + VERSION + ("-smoke" if smoke else "") + ".db"],
                            logs=["logs/" + VERSION + ("-smoke" if smoke else "") + "/driver.log"], links=LINKS):
            main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
