"""重新加载已训练模型跑一次 eval，把 run_episodes() 已经在算的 "ars" 和
"success" 两个数组（每个测试场景一条）单独导出，算"只看成功场景"的AR均值。

不用重新训练——直接复用 run_all.py 里现成的 run_episodes()，只是把它返回
的逐场景数组存下来，而不是像 main() 那样立刻塌缩成一个 ar_mean。

用法: .venv/bin/python src/scripts/reeval_success_only_ar.py <scenario> <exp_id> [exp_id2 ...]
  例:  .venv/bin/python src/scripts/reeval_success_only_ar.py lt 4dd9ab33 9c3dcc30 ...
"""
import sys
from pathlib import Path

ROOT = Path("/home/soar009/github/my_code")


def reeval(scenario: str, exp_id: str):
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scenarios" / scenario))
    import importlib
    C = importlib.import_module(f"{scenario}.ppo_mask" if False else "ppo_mask.config")
    run_all = importlib.import_module("ppo_mask.run_all")
    from sb3_contrib import MaskablePPO

    model_dir = ROOT / f"results/add_states/{scenario}/ppo_mask/{exp_id}"
    model_files = list(model_dir.glob("model_*"))
    if not model_files:
        print(f"[skip] {exp_id}: no model .zip found in {model_dir}")
        return None
    model = MaskablePPO.load(str(model_files[0]))

    ecus, services, sc_name, _ = run_all.load_scenario(C.YAML_CONFIG, C.SCENARIO_IDX, C.SCENARIOS)

    def ppo_policy(obs, mask):
        action, _ = model.predict(obs, deterministic=False, action_masks=mask)
        return int(action)

    res = run_all.run_episodes(ecus, services, ppo_policy, n_samples=C.EVAL_BEST_OF_N)
    ars = res["ars"]
    success = res["success"]

    all_ar_mean = float(ars.mean())
    if success.sum() > 0:
        success_only_ar_mean = float(ars[success].mean())
    else:
        success_only_ar_mean = float("nan")
    fail_only_ar_mean = float(ars[~success].mean()) if (~success).sum() > 0 else float("nan")

    print(f"{exp_id}: success_rate={success.mean():.4f}  "
          f"all_ar_mean={all_ar_mean:.4f}  "
          f"success_only_ar_mean={success_only_ar_mean:.4f}  "
          f"fail_only_ar_mean={fail_only_ar_mean:.4f}  "
          f"n_success={int(success.sum())}/{len(success)}")
    return success.mean(), all_ar_mean, success_only_ar_mean


if __name__ == "__main__":
    scenario = sys.argv[1]
    exp_ids = sys.argv[2:]
    for exp_id in exp_ids:
        reeval(scenario, exp_id)
