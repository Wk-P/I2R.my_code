#!/usr/bin/env python3
"""paper_contents/v6.0.0/: the full experiment description of the 12 strategy x RL_algorithm variants of the summary
page (no training, no evaluation). Every number below is read from the code, the runs or the installed libraries and
checked with assert, not typed in:
  - the runs: src/scripts/merge_2d_table.py:load() (the same 108 runs = 12 cells x 3 scenarios x 3 seeds as /summary);
    every setting in results.json / run.json is asserted to be the same within a cell (commit / batch excepted);
  - the hyper-parameters: src/paper_rl/config.py and the run's own results.json / run.json params;
  - the library defaults: inspect.signature of the installed stable-baselines3 / sb3-contrib (same version as .venv-gpu,
    which trained every run);
  - the network: parameter counts of paper_rl.graph_net.GraphPolicyNet instantiated with the training arguments;
  - the environment: observation / action spaces of paper_rl.env.PlacementEnv built with each run's settings.
Files (zh + en each):
  README            index, sources, what was checked
  01_algorithms     the 12 variants: strategy (constraint handling) x RL_algorithm, reward, sources
  02_network        the relation-biased Transformer encoder: tokens, features, layers, heads, parameter counts
  03_training       libraries + versions, PPO / DQN / Double DQN hyper-parameters, Lagrangian dual, our callbacks
  04_environment    environment variables, observation (= state), action space, per-variant differences

    .venv/bin/python src/scripts/report_v6_0_0_info.py [--no-mail]
"""
import argparse
import inspect
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))
OUT = ROOT / "paper_contents" / "v6.0.0"
GPU_SITE = next((ROOT / ".venv-gpu/lib").glob("python*/site-packages"))

from merge_2d_table import LEARNERS, MECHS, SCENS, load   # noqa: E402
from paper_rl import config as C                          # noqa: E402

STRAT = {"mask": "maskable", "repair": "repair", "lagrange": "lagrangian", "none": "unconstraint"}
PAPER = {"mask": "Mask", "repair": "Repair", "lagrange": "Lagrangian", "none": "Unconstraint"}
ALGO = {"ppo": "PPO", "dqn": "DQN", "ddqn": "DDQN (Double DQN)"}
NM = {"lt": (10, 15), "eq": (10, 10), "gt": (15, 10)}


# ── facts ───────────────────────────────────────────────────────────────────
def runs():
    """Per cell: the settings every run of the cell shares (asserted), and the per-run sources."""
    keys = ["reward_mode", "reward_norm", "obs", "gamma", "full_episode", "exit_action", "action_mode", "net", "device", "bc",
            "glob_std", "separate_critic", "pair_head", "group_k", "train_pool", "checkpoints", "plan", "plan_decay", "glob3",
            "order", "order_seed", "gae_lambda", "lr", "ent_coef", "target_kl", "hp", "train_count", "data", "resumed"]
    rows = load()
    assert len(rows) == 108, len(rows)
    cell, src = {}, defaultdict(list)
    for r in rows:
        res = json.loads((Path(r["dir"]) / "results.json").read_text())
        run = json.loads((Path(r["dir"]) / "run.json").read_text())
        c = (r["mech"], r["learner"])
        tr = {k: v for k, v in (res.get("training") or {}).items() if k not in ("n_episodes", "train_seconds", "ar_last50", "final_lambda")}
        v = {k: res.get(k) for k in keys} | {"steps": run["steps"], "training": tr,
                                               "n_epochs": res.get("n_epochs") or (C.PPO_N_EPOCHS if r["learner"] == "ppo" else None)}
        if c in cell:
            assert cell[c] == v, (c, r["exp_id"])
        cell[c] = v
        assert (res["N"], res["M"]) == NM[r["scen"]]
        src[c].append(dict(seed=r["seed"], scen=r["scen"], exp_id=r["exp_id"], batch=run["batch"], commit=run["commit"],
                           final_lambda=(res.get("training") or {}).get("final_lambda")))
    assert set(cell) == {(m, l) for m in MECHS for l in LEARNERS}
    for c, v in cell.items():                        # what the text states, for every cell
        assert v["net"] == "graph" and v["obs"] == "raw" and v["gamma"] == 1.0 and v["full_episode"] and v["action_mode"] == "ecu"
        assert v["order"] == "desc" and v["reward_norm"] == "none" and v["device"] == "cuda" and v["steps"] == 20_000_000
        assert not any(v[k] for k in ("bc", "glob_std", "separate_critic", "pair_head", "group_k", "plan", "plan_decay", "glob3", "hp", "resumed"))
        assert v["train_pool"] == "results/sup_diag/{scen}.json" and v["train_count"] == 50000
        assert v["checkpoints"] == "5000000,10000000,15000000" and v["target_kl"] is None
        assert v["data"] == {"conflict_pair_prob": 0.6, "k_sets": 10}
        assert v["exit_action"] == (c[0] == "mask")
        assert v["reward_mode"] == ("ar_obj" if c[0] == "lagrange" else "ar_pen")
        if c[1] == "ppo":
            assert (v["lr"], v["gae_lambda"], v["ent_coef"]) == (C.PPO_LR, C.PPO_GAE_LAMBDA, C.PPO_ENT_COEF)
            assert v["n_epochs"] == (6 if c[0] == "mask" else C.PPO_N_EPOCHS)
        else:
            assert v["training"]["eps_steps"] == 2_000_000
        if c[0] == "lagrange":
            t = v["training"]
            assert (t["lambda_update"], t["lambda_lr"], t["lambda_init"], t["lambda_target"], t["lambda_max"], t["lambda_cost"]) == \
                ("adam", 0.035, 0.001, 0.001, "none", "behavior")
            assert t["lambda_window"] == ("rollout" if c[1] == "ppo" else C.LAMBDA_UPDATE_WINDOW)
    return cell, src


def versions():
    """Package versions of .venv-gpu (the training environment), from its dist-info folders."""
    out = {}
    for p in GPU_SITE.glob("*.dist-info"):
        name, ver = p.name[:-len(".dist-info")].rsplit("-", 1)
        out[name.lower().replace("_", "-")] = ver
    py = (ROOT / ".venv-gpu/pyvenv.cfg").read_text()
    out["python"] = re.search(r"version(?:_info)?\s*=\s*([\d.]+)", py).group(1)
    return out


def lib_defaults():
    """Defaults of the installed SB3 / sb3-contrib constructors that the runs do not set."""
    import sb3_contrib
    import stable_baselines3
    from sb3_contrib import MaskablePPO
    from stable_baselines3 import DQN, PPO
    v = versions()
    assert stable_baselines3.__version__ == v["stable-baselines3"] and sb3_contrib.__version__ == v["sb3-contrib"]
    d = lambda cls, k: inspect.signature(cls.__init__).parameters[k].default
    ppo = {k: d(PPO, k) for k in ("vf_coef", "max_grad_norm", "normalize_advantage", "clip_range_vf")}
    assert ppo == {k: d(MaskablePPO, k) for k in ppo}
    dqn = {k: d(DQN, k) for k in ("max_grad_norm", "exploration_initial_eps", "n_steps", "optimize_memory_usage")}
    return ppo, dqn


def net_facts():
    import torch
    from paper_rl.graph_net import GraphPolicyNet
    from paper_rl.graph_policy import GraphMaskablePolicy
    from paper_rl.graph_dqn import GraphDQNPolicy
    sig = lambda f, k: inspect.signature(f).parameters[k].default
    d, h, layers = (sig(GraphMaskablePolicy.__init__, k) for k in ("d", "heads", "layers"))
    assert (d, h, layers) == tuple(sig(GraphDQNPolicy.__init__, k) for k in ("d", "heads", "layers"))
    n0, n1 = GraphPolicyNet(d=d, heads=h, layers=layers), GraphPolicyNet(d=d, heads=h, layers=layers, lam_in=True)
    cnt = lambda m: sum(p.numel() for p in m.parameters())
    parts = {k: cnt(getattr(n0, k)) for k in ("inp", "blocks", "ln", "ecu_head", "exit_head", "value_head")}
    parts["type_emb"] = n0.type_emb.numel()
    blk = n0.blocks[0]
    bparts = {"ln1 + ln2": cnt(blk.ln1) + cnt(blk.ln2), "qkv": cnt(blk.qkv), "o": cnt(blk.o), "ff": cnt(blk.ff), "rel_w": blk.rel_w.numel()}
    assert sum(parts.values()) == cnt(n0) and sum(bparts.values()) == cnt(blk)
    assert [m.in_features for m in n0.inp] == [4, 6, 5] and n1.inp[0].in_features == 5
    assert isinstance(blk.ff[1], torch.nn.GELU) and blk.ff[0].out_features == 2 * d
    return dict(d=d, heads=h, layers=layers, total=cnt(n0), total_lam=cnt(n1), parts=parts, block=bparts, per_block=cnt(blk))


def env_facts():
    from paper_rl.env import PlacementEnv
    out = {}
    for s in SCENS:
        x = json.loads((ROOT / f"data/testset_v6.0.4/{s}.json").read_text())[:1]
        for m in MECHS:
            e = PlacementEnv(x, m, "ar_obj" if m == "lagrange" else "ar_pen", obs_mode="raw", full_episode=True,
                             exit_action=m == "mask", lam_max=float("inf"))
            o, _ = e.reset()
            assert o.shape == e.observation_space.shape
            out[s, m] = dict(obs=e.observation_space.shape[0], act=int(e.action_space.n), low=float(e.observation_space.low[0]),
                             high=float(e.observation_space.high[0]), N=e.N, M=e.M)
    return out


# ── writing ─────────────────────────────────────────────────────────────────
def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def readme(T, cell, src):
    L = [T("# v6.0.0 实验信息总览（总图 12 个变种）", "# v6.0.0 experiment description (the 12 variants of the summary page)"), "",
         T("由 `src/scripts/report_v6_0_0_info.py` 生成；所有数值从代码、run 的 results.json / run.json、已安装的库读取并用 assert 核对，"
           "没有手写数字。不训练、不评估。",
           "Generated by `src/scripts/report_v6_0_0_info.py`; every value is read from the code, the runs' results.json / run.json and the "
           "installed libraries and checked with assert -- no number is typed in. No training, no evaluation."), "",
         T("## 文件", "## Files"), "",
         T("1. [01_algorithms](01_algorithms.md)：12 个变种（strategy × RL_algorithm）讲解、奖励、来源",
           "1. [01_algorithms](01_algorithms.en.md): the 12 variants (strategy × RL_algorithm), rewards, sources"),
         T("2. [02_network](02_network.md)：Transformer（关系偏置注意力编码器）的结构、层数、维度、参数量",
           "2. [02_network](02_network.en.md): the Transformer (relation-biased attention encoder): structure, layers, sizes, parameter counts"),
         T("3. [03_training](03_training.md)：库与版本、PPO / DQN / DDQN 超参数、Lagrangian 对偶更新、自定义回调函数",
           "3. [03_training](03_training.en.md): libraries and versions, PPO / DQN / DDQN hyper-parameters, Lagrangian dual update, our callbacks"),
         T("4. [04_environment](04_environment.md)：环境变量、观测（= 状态）、动作空间，以及各变种的区别",
           "4. [04_environment](04_environment.en.md): environment variables, observation (= state), action space, differences between the variants"), "",
         T("## 范围与核对", "## Scope and checks"), "",
         T("- 范围：总图（/summary，`src/scripts/merge_2d_table.py`）使用的 108 个 run = 12 格 × 3 场景（LT / EQ / GT）× 3 种子（1, 2, 3）。",
           "- Scope: the 108 runs of the summary page (/summary, `src/scripts/merge_2d_table.py`) = 12 cells × 3 scenarios (LT / EQ / GT) × 3 seeds (1, 2, 3)."),
         T("- 同一格内所有 run 的设置（奖励、网络、超参数、数据、步数等）逐项断言一致，只有代码提交与批次不同（见下表）。",
           "- Within a cell every setting (reward, network, hyper-parameters, data, steps, ...) is asserted identical across runs; only the commit and batch differ (table below)."),
         T("- 代码一致性：这些 run 来自 5 个提交。用 `git diff <提交> HEAD` 核对了训练相关文件（env.py、graph_net.py、graph_policy.py、graph_dqn.py、"
           "dqn_variants.py、train.py、config.py、vec_env.py）：差异只有变量改名（cap/req → c/e，i/j → m/n）、新增但这些 run 未使用的 ar_obj 奖励与 "
           "feas_in 开关（这些 run 全部等价于 feas_in=True）、新增未使用的命令行参数。对这些 run 的设置而言，训练行为与当前代码相同，本文按当前代码描述。",
           "- Code consistency: the runs come from 5 commits. `git diff <commit> HEAD` over the training files (env.py, graph_net.py, graph_policy.py, graph_dqn.py, "
           "dqn_variants.py, train.py, config.py, vec_env.py) shows only renames (cap/req → c/e, i/j → m/n), the ar_obj reward and the feas_in switch added "
           "but unused by these runs (all of them are equivalent to feas_in=True), and new unused flags. For these runs' settings training behaves as the current code, "
           "which this description follows."),
         T("- 记号：服务 m（共 M 个）、ECU n（共 N 个）、需求 e_m、容量 c_n、放置变量 x_mn、启用变量 y_n。DDQN = Double DQN。",
           "- Notation: service m (M of them), ECU n (N of them), demand e_m, capacity c_n, placement x_mn, ECU-in-use y_n. DDQN = Double DQN."), ""]
    L += [T("### 每格的来源", "### Sources per cell"), ""]
    rows = []
    for m in MECHS:
        for l in LEARNERS:
            by = defaultdict(lambda: defaultdict(set))
            for s in src[m, l]:
                by[(s["batch"], s["commit"])][s["seed"]].add(s["scen"])
            def fmt(sd):
                full = [str(k) for k, v in sorted(sd.items()) if len(v) == 3]
                part = [f"{k} ({', '.join(x.upper() for x in sorted(v))})" for k, v in sorted(sd.items()) if len(v) < 3]
                return T("种子 ", "seed ") + ", ".join(full + part)
            rows.append([STRAT[m], ALGO[l], "<br>".join(f"{b} @ `{c}`: {fmt(sd)}"
                                                        for (b, c), sd in sorted(by.items(), key=lambda kv: (min(kv[1]), kv[0][1])))])
    L += table(["strategy", "RL_algorithm", T("批次 @ 提交：种子（只列不全的场景）", "batch @ commit: seeds (scenarios listed when not all three)")], rows) + [""]
    return "\n".join(L)


def algorithms(T, cell, src):
    L = [T("# 1. 12 个变种：strategy × RL_algorithm", "# 1. The 12 variants: strategy × RL_algorithm"), "",
         T("## 1.1 共同的问题设定", "## 1.1 Common setting"), "",
         T("- 任务：把 M 个服务放到 N 个 ECU 上。每个实例给定容量 c_n、需求 e_m 和隐私冲突集（K = 10 个）。",
           "- Task: place M services on N ECUs. An instance gives the capacities c_n, the demands e_m and the privacy conflict sets (K = 10)."),
         T("- 约束：容量 —— 放在 ECU n 上的服务需求之和 ≤ c_n；隐私 —— 同一冲突集中的两个服务不能放在同一个 ECU 上。",
           "- Constraints: capacity -- the demands placed on ECU n sum to at most c_n; privacy -- two services of the same conflict set may not share an ECU."),
         T("- 目标 AR（平均资源利用率）= Σ_{合法放置 (m,n)} e_m / c_n ÷ 至少承载一个合法放置的 ECU 数。违约的放置不计入 AR。",
           "- Objective AR (average resource utilisation) = Σ_{legal placements (m,n)} e_m / c_n ÷ the number of ECUs hosting at least one legal placement. A violating placement adds nothing to AR."),
         T("- 序贯决策：服务按需求降序排列（`order = desc`），第 t 步为第 t 个服务选择一个 ECU；一个 episode 最多 M 步。",
           "- Sequential decision: services are sorted by descending demand (`order = desc`); step t chooses an ECU for the t-th service; an episode has at most M steps."),
         T("- `full_episode = True`（所有 run）：不提前结束；出现死局（当前服务没有可行 ECU）时：maskable 只能选 EXIT；其余 strategy 照常执行并记录违约。唯一的提前结束是 maskable 的 EXIT 动作。",
           "- `full_episode = True` (every run): no early stop; at a dead end (no feasible ECU for the current service) maskable can only choose EXIT, the other strategies execute the placement and record the violation. The only early end is maskable's EXIT action."),
         T(f"- 场景：LT（N={NM['lt'][0]}, M={NM['lt'][1]}，资源紧张）、EQ（N={NM['eq'][0]}, M={NM['eq'][1]}）、GT（N={NM['gt'][0]}, M={NM['gt'][1]}，资源充足）。",
           f"- Scenarios: LT (N={NM['lt'][0]}, M={NM['lt'][1]}, scarce), EQ (N={NM['eq'][0]}, M={NM['eq'][1]}), GT (N={NM['gt'][0]}, M={NM['gt'][1]}, ample)."),
         T("- 折扣 γ = 1.0（所有 run），所以回报 = 一个 episode 的奖励总和。",
           "- Discount γ = 1.0 (every run), so the return is the sum of an episode's rewards."), "",
         T("## 1.2 四种 strategy（约束处理方式）", "## 1.2 The four strategies (constraint handling)"), "",
         T("**maskable（论文名 Mask）**", "**maskable (paper name Mask)**"), "",
         T("- 动作掩码 = 对当前服务 m，同时满足容量（剩余容量 ≥ e_m）和隐私（ECU 上没有与 m 冲突的服务）的 ECU。",
           "- Action mask = the ECUs that satisfy both capacity (remaining capacity ≥ e_m) and privacy (no service conflicting with m on the ECU) for the current service m."),
         T("- 额外的 EXIT 动作（下标 N）：仅当没有任何可行 ECU 时它才有效，此时它也是唯一有效动作。选择 EXIT 会以失败奖励立即结束 episode。因此 Mask 在构造上永不违约。",
           "- An extra EXIT action (index N): valid only when no ECU is feasible, and then the only valid action. Choosing EXIT ends the episode at once with the failure reward. Mask therefore never violates by construction."),
         T("- 奖励：`ar_pen`（见 1.3）。", "- Reward: `ar_pen` (see 1.3)."), "",
         T("**repair**", "**repair**"), "",
         T("- 智能体选择的动作若不可行，环境把它替换为可行 ECU 中 e_m / c_n 最大者（即容量最小的可行 ECU，并列取下标最小），并计入 `repairs`。",
           "- If the chosen action is infeasible, the environment replaces it by the feasible ECU with the largest e_m / c_n (i.e. the feasible ECU of smallest capacity, lowest index on ties) and counts it in `repairs`."),
         T("- 若没有任何可行 ECU，则执行智能体自己的动作并记录违约（full_episode）。所以 repair 只在死局时违约。",
           "- If no ECU is feasible, the agent's own action is executed and the violation recorded (full_episode). Repair therefore violates only at a dead end."),
         T("- 替换发生在环境内部，回放缓冲 / rollout 中存的是智能体原本选的动作。奖励：`ar_pen`。",
           "- The replacement happens inside the environment; the replay buffer / rollout stores the action the agent chose. Reward: `ar_pen`."), "",
         T("**lagrangian（标准 Lagrangian）**", "**lagrangian (standard Lagrangian)**"), "",
         T("- 动作按选择执行，违约照常记录。每步代价 c_t = 1[容量违约] + 1[隐私违约] ∈ {0, 1, 2}。",
           "- Actions are executed as chosen and violations recorded. Per-step cost c_t = 1[capacity violated] + 1[privacy violated] ∈ {0, 1, 2}."),
         T("- 每步奖励 r_t = r_t^obj − λ·c_t，其中 r_t^obj 是 `ar_obj`（见 1.3）；λ 是对偶变量，用 Adam 更新（参数见 03_training），在一个 episode 内固定，"
           "并作为观测的最后一维（λ / (1 + λ)）输入网络。",
           "- Per-step reward r_t = r_t^obj − λ·c_t with r_t^obj the `ar_obj` reward (see 1.3); λ is the dual variable, updated by Adam (settings in 03_training), fixed within an episode "
           "and fed to the network as the last observation entry (λ / (1 + λ))."), "",
         T("**unconstraint（论文名 Unconstraint）**", "**unconstraint (paper name Unconstraint)**"), "",
         T("- 不做任何约束处理：动作按选择执行，违约只被记录。奖励：完整的 `ar_pen`（v6.0.0 定义），违约导致 episode 失败、得到负奖励。",
           "- No constraint handling: actions are executed as chosen and violations only recorded. Reward: the full `ar_pen` (v6.0.0 definition); a violation makes the episode a failure with a negative reward."),
         T("- 状态中包含可行性特征（与其他 strategy 相同，feas_in = True），但没有任何机制强制可行。",
           "- The state contains the feasibility features (as for the other strategies, feas_in = True), but nothing enforces feasibility."), "",
         T("## 1.3 奖励", "## 1.3 Rewards"), "",
         T("记 valid = 合法放置的服务数。所有中间步奖励为 0，只在最后一步给出：",
           "Let valid = the number of legally placed services. Every non-terminal step pays 0; the terminal step pays:"), "",
         T("- `ar_pen`（maskable / repair / unconstraint）：全部 M 个服务都合法放置时为 AR ∈ (0, 1]；否则（含 EXIT）为 −(1 − valid / M) ∈ [−1, 0)。",
           "- `ar_pen` (maskable / repair / unconstraint): AR ∈ (0, 1] if all M services are placed legally; otherwise (EXIT included) −(1 − valid / M) ∈ [−1, 0)."),
         T("- `ar_obj`（lagrangian）：AR_obj = 在所有执行过放置（含违约放置）的 ECU 上，取 min(负载_n / c_n, 1) 的平均；没有失败惩罚，违约只通过每步的 −λ·c_t 体现。",
           "- `ar_obj` (lagrangian): AR_obj = the mean of min(load_n / c_n, 1) over the ECUs with any executed placement (violating ones included); no failure penalty -- violations enter only through −λ·c_t every step."), "",
         T("## 1.4 三种 RL_algorithm", "## 1.4 The three RL_algorithms"), "",
         T("- **PPO**：on-policy 的 actor-critic（裁剪的概率比目标 + GAE）。maskable 用 sb3-contrib 的 `MaskablePPO`，其余用 stable-baselines3 的 `PPO`；"
           "策略网络都是我们的 `GraphMaskablePolicy`（02_network），actor 与 critic 共用一个编码器。",
           "- **PPO**: on-policy actor-critic (clipped probability-ratio objective + GAE). maskable uses sb3-contrib's `MaskablePPO`, the others stable-baselines3's `PPO`; "
           "the policy is always our `GraphMaskablePolicy` (02_network), actor and critic sharing one encoder."),
         T("- **DQN**：off-policy 的值学习，经验回放 + 目标网络，TD 目标 r + γ·max_a' Q_target(s', a')。maskable 用我们的 `MaskableDQN`，其余用 stable-baselines3 的 `DQN`。",
           "- **DQN**: off-policy value learning with experience replay and a target network, TD target r + γ·max_a' Q_target(s', a'). maskable uses our `MaskableDQN`, the others stable-baselines3's `DQN`."),
         T("- **DDQN = Double DQN**（van Hasselt et al., 2016）：在线网络选出下一状态的动作 a* = argmax_a' Q_online(s', a')，目标网络给它估值，TD 目标 r + γ·Q_target(s', a*)，"
           "以减小 DQN 的 Q 值高估。maskable 用我们的 `MaskableDDQN`，其余用我们的 `DoubleDQN`（`src/shared/dqn_variants.py`，继承 SB3 `DQN`，只改 `train()`）。",
           "- **DDQN = Double DQN** (van Hasselt et al., 2016): the online network picks the next-state action a* = argmax_a' Q_online(s', a'), the target network evaluates it, TD target r + γ·Q_target(s', a*), "
           "reducing DQN's Q-value overestimation. maskable uses our `MaskableDDQN`, the others our `DoubleDQN` (`src/shared/dqn_variants.py`, subclasses of SB3 `DQN` that only override `train()`)."),
         T("- Q 网络与 PPO 策略是同一个结构（02_network）：每个 ECU token 给出 Q(s, ECU n)；maskable 另由全局 token 给出 Q(s, EXIT)。",
           "- The Q-network has the same structure as the PPO policy (02_network): every ECU token gives Q(s, ECU n); for maskable the global token also gives Q(s, EXIT)."), "",
         T("## 1.5 12 个变种一览", "## 1.5 The 12 variants"), ""]
    cls = {("mask", "ppo"): "sb3_contrib.MaskablePPO", ("mask", "dqn"): "MaskableDQN", ("mask", "ddqn"): "MaskableDDQN"}
    mask_how = {"ppo": T("环境 `action_masks()` 每步给出，存入 rollout 并用于更新；网络内部也按同样的可行性把不可行 logit 置为 −1e9",
                         "from the environment's `action_masks()` every step, stored in the rollout and used in the update; the network also sets infeasible logits to −1e9 by the same feasibility"),
                "dqn": T("由网络从原始观测计算（`graph_net.build`，与环境同一可行性）；ε-贪心只在可行动作中随机，贪心与 TD 目标的 max 只在可行动作上取",
                         "computed by the network from the raw observation (`graph_net.build`, the same feasibility as the environment); ε-greedy samples among valid actions only, the greedy action and the TD-target max range over valid actions only"),
                "ddqn": T("同 DQN；TD 目标中在线网络的 argmax 只在下一状态的可行动作上取", "as DQN; the online network's argmax in the TD target ranges over the next state's valid actions only")}
    rows = []
    for m in MECHS:
        for l in LEARNERS:
            v = cell[m, l]
            c = cls.get((m, l)) or {"ppo": "stable_baselines3.PPO", "dqn": "stable_baselines3.DQN", "ddqn": "DoubleDQN"}[l]
            act = "N + 1 (EXIT)" if m == "mask" else "N"
            how = mask_how[l] if m == "mask" else {"repair": T("无掩码；环境替换不可行动作", "no mask; the environment replaces infeasible actions"),
                                                   "lagrange": T("无掩码；−λ·c_t 惩罚", "no mask; −λ·c_t penalty"),
                                                   "none": T("无", "none")}[m]
            rows.append([STRAT[m], ALGO[l], f"`{c}`", f"`{v['reward_mode']}`" + (" − λ·c_t" if m == "lagrange" else ""), act, how])
    L += table(["strategy", "RL_algorithm", T("类", "class"), T("奖励", "reward"), T("动作数", "actions"), T("约束处理", "constraint handling")], rows) + [""]
    L += [T("## 1.6 训练与评估", "## 1.6 Training and evaluation"), "",
          T("- 每个 run 训练 20,000,000 步（环境交互步数），训练实例为 `results/sup_diag/<场景>.json` 的 50,000 个 ILP 验证实例（p = 0.6，K = 10），每次 reset 随机抽一个。",
            "- Every run trains for 20,000,000 environment steps on the 50,000 ILP-verified instances of `results/sup_diag/<scenario>.json` (p = 0.6, K = 10), one drawn at random at every reset."),
          T("- 总图的评估：最终模型在 v6.0.4 固定测试集（`data/testset_v6.0.4/<场景>.json`，每场景 5000 个 ILP 最优实例，与训练池不重叠）上确定性评估（`src/scripts/eval_testset_v6_0_4.py`），"
            "设置与该 run 训练时相同（Lagrangian 用训练结束时的 λ）。",
            "- Summary evaluation: the final model, deterministic, on the fixed v6.0.4 test set (`data/testset_v6.0.4/<scenario>.json`, 5000 ILP-optimal instances per scenario, disjoint from the training pool; "
            "`src/scripts/eval_testset_v6_0_4.py`), with the run's own settings (Lagrangian: the λ at the end of training)."),
          T("- gap = 1 − mean(AR) / mean(ILP AR)；ILP AR 由 Dinkelbach 方法求得（`shared/ilp_utils.solve_ilp_max_ar`）。",
            "- gap = 1 − mean(AR) / mean(ILP AR); ILP AR from Dinkelbach's method (`shared/ilp_utils.solve_ilp_max_ar`)."), ""]
    L += [T("### Lagrangian 训练结束时的 λ", "### Lagrangian: λ at the end of training"), ""]
    rows = []
    for l in LEARNERS:
        for s in SCENS:
            lam = sorted((x["seed"], x["final_lambda"]) for x in src["lagrange", l] if x["scen"] == s)
            rows.append([ALGO[l], s.upper()] + [f"{v:.4f}" for _, v in lam])
    L += table(["RL_algorithm", T("场景", "scenario"), T("种子 1", "seed 1"), T("种子 2", "seed 2"), T("种子 3", "seed 3")], rows) + [""]
    return "\n".join(L)


def network(T, nf):
    d, h, ly = nf["d"], nf["heads"], nf["layers"]
    p, b = nf["parts"], nf["block"]
    L = [T("# 2. 网络：关系偏置的 Transformer 编码器", "# 2. Network: relation-biased Transformer encoder"), "",
         T("代码：`src/paper_rl/graph_net.py`（`GraphPolicyNet`、`build`、`Block`），PPO 包装 `src/paper_rl/graph_policy.py`，DQN 包装 `src/paper_rl/graph_dqn.py`。"
           "12 个变种全部使用同一个结构（`--net graph`），没有使用 MLP。",
           "Code: `src/paper_rl/graph_net.py` (`GraphPolicyNet`, `build`, `Block`), PPO wrapper `src/paper_rl/graph_policy.py`, DQN wrapper `src/paper_rl/graph_dqn.py`. "
           "All 12 variants use this one structure (`--net graph`); no MLP is used."), "",
         T("## 2.1 结构参数", "## 2.1 Sizes"), ""]
    L += table([T("项", "item"), T("值", "value")], [
        [T("隐藏维度 d", "hidden size d"), d], [T("注意力头数", "attention heads"), T(f"{h}（d_k = {d // h}）", f"{h} (d_k = {d // h})")],
        [T("Transformer 层数", "Transformer layers"), ly], [T("前馈层", "feed-forward"), f"{d} → {2 * d} → {d}, GELU"],
        [T("归一化", "normalisation"), T("Pre-LN（每个子层前 LayerNorm）+ 编码器末尾 LayerNorm", "pre-LN (LayerNorm before each sub-layer) + a final LayerNorm")],
        [T("位置编码", "positional encoding"), T("无（对 ECU、服务的排列等变）；当前服务由特征「是否当前」标出", "none (permutation-equivariant in ECUs and services); the current service is marked by an \"is current\" feature")],
        [T("token 数", "tokens"), "1 + N + M"],
        [T("参数量（PPO 一个网络；DQN 在线 / 目标网络各一个）", "parameters (PPO: one network; DQN: online and target network, one each)"),
         T(f"{nf['total']:,}（lagrangian：{nf['total_lam']:,}，全局输入多 1 维 λ）", f"{nf['total']:,} (lagrangian: {nf['total_lam']:,}, one extra global input λ)")]]) + [""]
    L += [T("## 2.2 输入：token 与特征", "## 2.2 Input: tokens and features"), "",
          T("网络读原始观测（04_environment），在 `build()` 里计算全部特征，输入只是这些特征（不读其他东西）：",
            "The network reads the raw observation (04_environment) and derives every feature in `build()`:"), "",
          T("- **全局 token（4 维）**：当前 AR、未放置服务占比、未放置需求 / 总容量、剩余容量 / 总容量。lagrangian 再加 1 维 λ / (1 + λ)。",
            "- **global token (4)**: AR so far, share of unplaced services, unplaced demand / total capacity, free capacity / total capacity. lagrangian adds λ / (1 + λ)."),
          T("- **ECU token（每个 6 维）**：c_n / c_max、max(剩余容量, 0) / c_max、是否已启用、对当前服务是否可行、承载服务数 / M、负载 / c_n。",
            "- **ECU token (6 each)**: c_n / c_max, max(free capacity, 0) / c_max, in use, feasible for the current service, hosted services / M, load / c_n."),
          T("- **服务 token（每个 5 维）**：e_m / c_max、是否已放置、是否当前服务、可行 ECU 数 / N、与其他未放置服务的冲突数 / M（已放置的服务为 0）。c_max = 该实例最大容量。",
            "- **service token (5 each)**: e_m / c_max, placed, is current, feasible ECUs / N, conflicts with other unplaced services / M (0 once placed). c_max = the instance's largest capacity."),
          T("- 每种 token 先经各自的线性层投影到 d 维（4→128、6→128、5→128；lagrangian 全局 5→128），再加上该类型的可学习类型嵌入（3 × d，零初始化）。",
            "- Each token type is projected to d by its own linear layer (4→128, 6→128, 5→128; lagrangian global 5→128), plus a learned type embedding (3 × d, zero-initialised)."), "",
          T("## 2.3 关系偏置注意力", "## 2.3 Relation-biased attention"), "",
          T("每层的注意力 logit = QKᵀ / √d_k + Σ_r R_r · w_{r,h}，R_r 是 4 种 0/1 关系矩阵，w 是每层每头可学习的权重（4 × heads，零初始化）：",
            "In every layer the attention logits are QKᵀ / √d_k + Σ_r R_r · w_{r,h}, with R_r four 0/1 relation matrices and w a learned weight per layer and head (4 × heads, zero-initialised):"), "",
          T("1. 服务–服务冲突（两者都未放置）", "1. service–service conflict (both unplaced)"),
          T("2. 服务承载于 ECU（对称）", "2. service hosted on ECU (symmetric)"),
          T("3. 未放置服务与 ECU 上的某个服务冲突（对称）", "3. unplaced service conflicts with a service on the ECU (symmetric)"),
          T("4. 未放置服务的需求 ≤ ECU 剩余容量（对称）", "4. unplaced service fits the ECU's free capacity (symmetric)"), "",
          T("## 2.4 输出头", "## 2.4 Heads"), ""]
    L += table([T("头", "head"), T("结构", "structure"), T("用途", "use")], [
        ["ecu_head", f"Linear({d}→{d}) → GELU → Linear({d}→1)", T("每个 ECU token 一个分数，所有 ECU 共用同一个头", "one score per ECU token, the same head for every ECU")],
        ["exit_head", f"Linear({d}→1)", T("由全局 token 给 EXIT 打分（只有 maskable 用到）", "scores EXIT from the global token (used by maskable only)")],
        ["value_head", f"Linear({2 * d}→{d}) → GELU → Linear({d}→1)", T("输入 [全局 token, 其余 token 的均值]，给出 V(s)（只有 PPO 用到）",
                                                                  "input [global token, mean of the other tokens], gives V(s) (PPO only)")]]) + [""]
    L += [T("## 2.5 在不同算法中的用法", "## 2.5 Use per algorithm"), "",
          T("- PPO：actor 与 critic 共用编码器（`separate_critic = False`）。maskable 用 N + 1 个 logit（含 EXIT），并在网络内按可行性把不可行 logit 置为 −1e9；"
            "其余 strategy 用不加掩码的分数，截取前 N 个（无 EXIT）。动作分布为分类分布（Categorical）。",
            "- PPO: actor and critic share the encoder (`separate_critic = False`). maskable uses N + 1 logits (EXIT included) and sets infeasible logits to −1e9 inside the network; "
            "the other strategies use the unmasked scores cut to the first N (no EXIT). The action distribution is categorical."),
          T("- DQN / DDQN：不加掩码的分数直接作为 Q 值（maskable 为 N + 1 个，其余为 N 个）；value_head 不参与，不产生梯度。在线网络和目标网络各是一个完整的 `GraphPolicyNet`。",
            "- DQN / DDQN: the unmasked scores are the Q-values (N + 1 for maskable, N otherwise); value_head is not used and gets no gradient. The online and the target network are each a full `GraphPolicyNet`."),
          T("- 关闭的可选结构（所有 run）：glob_std、glob3、pair_head、plan、plan_decay、separate_critic 全部为 False。",
            "- Optional parts switched off (every run): glob_std, glob3, pair_head, plan, plan_decay, separate_critic all False."),
          T("- 初始化：PyTorch 默认初始化（`_build` 被覆盖，SB3 的正交初始化不起作用）；关系权重 w 和类型嵌入为零初始化。",
            "- Initialisation: PyTorch defaults (`_build` is overridden, so SB3's orthogonal initialisation does not apply); relation weights w and type embeddings start at zero."),
          T("- config.py 中的 `PPO_NET_ARCH`（256-256）与 `DQN_NET_ARCH`（128-128）只用于 MLP 网络，这些 run 不使用。",
            "- `PPO_NET_ARCH` (256-256) and `DQN_NET_ARCH` (128-128) in config.py are for the MLP network only and unused by these runs."), "",
          T("## 2.6 参数量明细（不含 λ 输入）", "## 2.6 Parameter count (without the λ input)"), ""]
    L += table([T("部分", "part"), T("参数", "parameters")],
               [[k, f"{v:,}"] for k, v in p.items()] + [[T("合计", "total"), f"{nf['total']:,}"]]) + [""]
    L += [T(f"每个 Transformer 层 {nf['per_block']:,} 个参数：", f"Each Transformer layer has {nf['per_block']:,} parameters:"), ""]
    L += table([T("部分", "part"), T("参数", "parameters")], [[k, f"{v:,}"] for k, v in b.items()]) + [""]
    return "\n".join(L)


def training(T, cell, ver, ppo_def, dqn_def):
    v = lambda c: cell[c]
    L = [T("# 3. 库、超参数与回调函数", "# 3. Libraries, hyper-parameters and callbacks"), "",
         T("## 3.1 库与版本（训练环境 `.venv-gpu`）", "## 3.1 Libraries and versions (training environment `.venv-gpu`)"), ""]
    L += table([T("库", "library"), T("版本", "version"), T("用途", "use")], [
        ["Python", ver["python"], ""],
        ["torch", ver["torch"], T("网络与训练（CUDA）", "networks and training (CUDA)")],
        ["stable-baselines3", ver["stable-baselines3"], T("`PPO`、`DQN`（DoubleDQN / MaskableDQN / MaskableDDQN 的基类）、`Monitor`、`DummyVecEnv`、回调基类",
                                                      "`PPO`, `DQN` (base of DoubleDQN / MaskableDQN / MaskableDDQN), `Monitor`, `DummyVecEnv`, callback base")],
        ["sb3-contrib", ver["sb3-contrib"], T("`MaskablePPO`、`MaskableActorCriticPolicy`（GraphMaskablePolicy 的基类）",
                                          "`MaskablePPO`, `MaskableActorCriticPolicy` (base of GraphMaskablePolicy)")],
        ["gymnasium", ver["gymnasium"], T("环境接口（`PlacementEnv`）", "environment interface (`PlacementEnv`)")],
        ["numpy", ver["numpy"], ""]]) + [""]
    L += table(["RL_algorithm", "strategy", T("类（来源）", "class (source)")], [
        ["PPO", "maskable", "`MaskablePPO` (sb3-contrib)"],
        ["PPO", "repair / lagrangian / unconstraint", "`PPO` (stable-baselines3)"],
        ["DQN", "maskable", T("`MaskableDQN`（本项目，继承 SB3 `DQN`）", "`MaskableDQN` (ours, subclass of SB3 `DQN`)")],
        ["DQN", "repair / lagrangian / unconstraint", "`DQN` (stable-baselines3)"],
        ["DDQN", "maskable", T("`MaskableDDQN`（本项目，继承 MaskableDQN）", "`MaskableDDQN` (ours, subclass of MaskableDQN)")],
        ["DDQN", "repair / lagrangian / unconstraint", T("`DoubleDQN`（本项目，继承 SB3 `DQN`）", "`DoubleDQN` (ours, subclass of SB3 `DQN`)")]]) + [""]
    L += [T("策略 / Q 网络：PPO 全部为 `GraphMaskablePolicy`（本项目），DQN / DDQN 全部为 `GraphDQNPolicy`（本项目），见 02_network。",
            "Policy / Q-network: every PPO uses `GraphMaskablePolicy` (ours), every DQN / DDQN `GraphDQNPolicy` (ours); see 02_network."), "",
          T("## 3.2 PPO 超参数", "## 3.2 PPO hyper-parameters"), ""]
    np_ = C.PPO_N_ENVS * C.PPO_N_STEPS
    L += table([T("参数", "parameter"), T("值", "value"), T("来源", "source")], [
        [T("并行环境数 n_envs", "parallel envs n_envs"), C.PPO_N_ENVS, "config.PPO_N_ENVS"],
        ["n_steps", T(f"{C.PPO_N_STEPS}（每次更新 {np_:,} 条转移）", f"{C.PPO_N_STEPS} ({np_:,} transitions per update)"), "config.PPO_N_STEPS"],
        ["batch_size", T(f"{C.PPO_BATCH_SIZE}（每轮 {np_ // C.PPO_BATCH_SIZE} 个小批量）", f"{C.PPO_BATCH_SIZE} ({np_ // C.PPO_BATCH_SIZE} minibatches per epoch)"), "config.PPO_BATCH_SIZE"],
        ["n_epochs", T(f"maskable：{v(('mask', 'ppo'))['n_epochs']}；其余：{v(('repair', 'ppo'))['n_epochs']}",
                       f"maskable: {v(('mask', 'ppo'))['n_epochs']}; others: {v(('repair', 'ppo'))['n_epochs']}"),
         T("maskable：`--n-epochs 6`（v6.0.5-PPO-hyparam）；其余：config.PPO_N_EPOCHS", "maskable: `--n-epochs 6` (v6.0.5-PPO-hyparam); others: config.PPO_N_EPOCHS")],
        ["learning_rate", T(f"{C.PPO_LR}（常数）", f"{C.PPO_LR} (constant)"), "config.PPO_LR"],
        ["gamma", v(("mask", "ppo"))["gamma"], "`--gamma 1.0`"],
        ["gae_lambda", C.PPO_GAE_LAMBDA, "config.PPO_GAE_LAMBDA"],
        ["clip_range", C.PPO_CLIP_RANGE, "config.PPO_CLIP_RANGE"],
        ["ent_coef", T(f"{C.PPO_ENT_COEF}（常数）", f"{C.PPO_ENT_COEF} (constant)"), "config.PPO_ENT_COEF"],
        ["vf_coef", ppo_def["vf_coef"], T("库默认", "library default")],
        ["max_grad_norm", ppo_def["max_grad_norm"], T("库默认", "library default")],
        ["normalize_advantage", ppo_def["normalize_advantage"], T("库默认", "library default")],
        ["clip_range_vf", ppo_def["clip_range_vf"], T("库默认（不裁剪价值）", "library default (no value clipping)")],
        ["target_kl", "None", T("未设置", "not set")],
        [T("优化器", "optimizer"), "Adam, eps = 1e-5", T("MaskableActorCriticPolicy 默认", "MaskableActorCriticPolicy default")],
        [T("torch 线程 / 设备", "torch threads / device"), f"{C.PPO_TORCH_THREADS} / cuda", "config.PPO_TORCH_THREADS, `--device cuda`"]]) + [""]
    L += [T("## 3.3 DQN / DDQN 超参数（两者相同）", "## 3.3 DQN / DDQN hyper-parameters (identical)"), ""]
    tu = C.DQN_TARGET_UPDATE // C.DQN_N_ENVS
    L += table([T("参数", "parameter"), T("值", "value"), T("来源", "source")], [
        [T("并行环境数 n_envs", "parallel envs n_envs"), C.DQN_N_ENVS, "config.DQN_N_ENVS"],
        ["learning_rate", T(f"{C.DQN_LR}（常数）", f"{C.DQN_LR} (constant)"), "config.DQN_LR"],
        ["buffer_size", f"{C.DQN_BUFFER_SIZE:,}", "config.DQN_BUFFER_SIZE"],
        ["learning_starts", T(f"{C.DQN_LEARNING_STARTS:,} 条转移（之前随机动作，maskable 只在可行动作中随机）",
                              f"{C.DQN_LEARNING_STARTS:,} transitions (random actions before; maskable: among valid actions only)"), "config.DQN_LEARNING_STARTS"],
        ["batch_size", C.DQN_BATCH_SIZE, "config.DQN_BATCH_SIZE"],
        ["gamma", v(("mask", "dqn"))["gamma"], "`--gamma 1.0`"],
        ["train_freq / gradient_steps", T(f"每 {C.DQN_TRAIN_FREQ} 次向量步（{C.DQN_TRAIN_FREQ * C.DQN_N_ENVS} 条转移）做 {C.DQN_GRADIENT_STEPS} 次梯度更新",
                                          f"{C.DQN_GRADIENT_STEPS} gradient step every {C.DQN_TRAIN_FREQ} vector steps ({C.DQN_TRAIN_FREQ * C.DQN_N_ENVS} transitions)"),
         "config.DQN_TRAIN_FREQ / DQN_GRADIENT_STEPS"],
        ["target_update_interval / tau", T(f"{C.DQN_TARGET_UPDATE} / {C.DQN_TAU}：每 {tu} 次向量步（{tu * C.DQN_N_ENVS} 条转移）把在线网络整体复制到目标网络",
                                           f"{C.DQN_TARGET_UPDATE} / {C.DQN_TAU}: the online network is copied to the target every {tu} vector steps ({tu * C.DQN_N_ENVS} transitions)"),
         T("config；SB3 按 interval // n_envs 计次", "config; SB3 counts interval // n_envs calls")],
        [T("探索 ε", "exploration ε"), T(f"{dqn_def['exploration_initial_eps']} → {C.DQN_EXPLORATION_FINAL_EPS}，在前 {v(('mask', 'dqn'))['training']['eps_steps']:,} 步内线性下降，之后为 0",
                                          f"{dqn_def['exploration_initial_eps']} → {C.DQN_EXPLORATION_FINAL_EPS}, linear over the first {v(('mask', 'dqn'))['training']['eps_steps']:,} steps, 0 afterwards"),
         T("`--eps-steps 2000000`（exploration_fraction = 0.1）", "`--eps-steps 2000000` (exploration_fraction = 0.1)")],
        [T("损失", "loss"), "Smooth L1 (Huber)", T("SB3 `DQN.train` / 本项目 `_td_update`，相同", "SB3 `DQN.train` / our `_td_update`, the same")],
        ["max_grad_norm", dqn_def["max_grad_norm"], T("库默认", "library default")],
        ["n_steps (TD)", dqn_def["n_steps"], T("库默认（单步 TD）", "library default (one-step TD)")],
        [T("优化器", "optimizer"), "Adam (torch default, eps = 1e-8)", T("DQNPolicy 默认（optimizer_kwargs 为空）", "DQNPolicy default (empty optimizer_kwargs)")],
        [T("torch 线程 / 设备", "torch threads / device"), f"{C.DQN_TORCH_THREADS} / cuda", "config.DQN_TORCH_THREADS, `--device cuda`"]]) + [""]
    L += [T("DQN 与 DDQN 的唯一区别是 TD 目标（01_algorithms §1.4）。", "DQN and DDQN differ only in the TD target (01_algorithms §1.4)."), "",
          T("## 3.4 Lagrangian 对偶更新（只用于 lagrangian）", "## 3.4 Lagrangian dual update (lagrangian only)"), ""]
    t = v(("lagrange", "ppo"))["training"]
    L += table([T("参数", "parameter"), T("值", "value")], [
        [T("更新方式", "update"), T("把 λ 当作参数，损失 −λ·(J_C − d)，Adam（默认 β），更新后截断到 λ ≥ 0",
                                   "λ as a parameter, loss −λ·(J_C − d), Adam (default betas), clamped to λ ≥ 0 after each step")],
        [T("学习率", "learning rate"), t["lambda_lr"]], [T("初值 λ_0", "initial λ_0"), t["lambda_init"]],
        [T("目标 d", "target d"), t["lambda_target"]], [T("上限", "upper bound"), T("无（`--lambda-max none`）", "none (`--lambda-max none`)")],
        ["J_C", T("每个 episode 的违约次数 Σ_t c_t 的平均（行为策略的训练 episode）", "mean over episodes of the episode's violations Σ_t c_t (training episodes of the behaviour policy)")],
        [T("更新频率", "frequency"), T(f"PPO：每个 rollout（{np_:,} 条转移）一次，用该 rollout 内结束的 episode；DQN / DDQN：每 {C.LAMBDA_UPDATE_WINDOW} 个 episode 一次（不重叠的窗口）",
                                       f"PPO: once per rollout ({np_:,} transitions) from the episodes that ended in it; DQN / DDQN: every {C.LAMBDA_UPDATE_WINDOW} episodes (non-overlapping windows)")],
        [T("生效时机", "takes effect"), T("下一次 reset（episode 内 λ 固定）", "at the next reset (λ fixed within an episode)")],
        [T("观测中的 λ", "λ in the observation"), T("λ / (1 + λ)（无上限时）", "λ / (1 + λ) (no upper bound)")]]) + [""]
    L += [T("## 3.5 共同的训练设置", "## 3.5 Common training settings"), ""]
    L += table([T("项", "item"), T("值", "value")], [
        [T("总步数", "total steps"), "20,000,000"], [T("种子", "seeds"), "1, 2, 3"],
        [T("训练实例", "training instances"), T("`results/sup_diag/<场景>.json`，每场景 50,000 个 ILP 验证实例", "`results/sup_diag/<scenario>.json`, 50,000 ILP-verified instances per scenario")],
        [T("检查点", "checkpoints"), T("5M、10M、15M（另存 20M 最终模型）", "5M, 10M, 15M (+ the 20M final model)")],
        [T("奖励缩放", "reward scaling"), T("无（`--reward-norm none`）", "none (`--reward-norm none`)")],
        [T("CUDA Graphs", "CUDA Graphs"), T("开启（默认 `V5_CUDA_GRAPHS=1`）；只影响速度，数值与普通执行一致（src/scripts/check_v5.py 逐位核对）",
                                          "on (default `V5_CUDA_GRAPHS=1`); speed only, the same values as eager execution (checked bit for bit by src/scripts/check_v5.py)")]]) + [""]
    L += [T("## 3.6 本项目的回调函数（`src/paper_rl/train.py`）", "## 3.6 Our callbacks (`src/paper_rl/train.py`)"), "",
          T("训练时用 `CallbackList([主回调, 检查点回调])`：", "Training uses `CallbackList([main callback, checkpoint callback])`:"), "",
          T("**1. 主回调 `make_callback` → `Cb`（继承 SB3 `BaseCallback`）**", "**1. Main callback `make_callback` → `Cb` (subclass of SB3 `BaseCallback`)**"), "",
          T("- `_on_training_start`：lagrangian 时把初始 λ 写入所有训练环境（`set_lambda`）。",
            "- `_on_training_start`: for lagrangian, writes the initial λ into every training environment (`set_lambda`)."),
          T("- `_on_step`：对每个结束的 episode 记录一行 (timestep, episode 奖励, AR, 是否无违约完成, 合法放置数, 是否容量违约, 是否隐私违约, λ, 是否 EXIT)，"
            "训练结束写入 `training_curve.csv`（episode 奖励来自 SB3 `Monitor`，是原始奖励之和，lagrangian 含 −λ·c_t）；"
            "lagrangian + DQN / DDQN 时累计 episode 违约数，每 1000 个 episode 更新一次 λ；每 100,000 步打印进度并写 `.progress.json`（看板用）。",
            "- `_on_step`: for every finished episode records (timestep, episode reward, AR, completed without violation, legal placements, capacity violated, privacy violated, λ, EXIT), "
            "written to `training_curve.csv` at the end (the episode reward comes from SB3 `Monitor`: the raw reward sum, −λ·c_t included for lagrangian); "
            "for lagrangian + DQN / DDQN accumulates the episode violations and updates λ every 1000 episodes; every 100,000 steps prints progress and writes `.progress.json` (for the panel)."),
          T("- `_on_rollout_end`：lagrangian + PPO 时，用本 rollout 内结束的 episode 的平均违约数更新一次 λ。",
            "- `_on_rollout_end`: for lagrangian + PPO, updates λ once from the mean violations of the episodes that ended in this rollout."),
          T("- λ 的每次更新记录在 `lambda_updates.csv`（列：episode, timestep, jc_behavior, jc_used, lambda；这些 run 中 jc_used = jc_behavior）。",
            "- Every λ update is logged in `lambda_updates.csv` (columns episode, timestep, jc_behavior, jc_used, lambda; jc_used = jc_behavior in these runs)."), "",
          T("**2. 检查点回调 `checkpoint_callback` → `Ckpt`**", "**2. Checkpoint callback `checkpoint_callback` → `Ckpt`**"), "",
          T("- 步数第一次达到 5M / 10M / 15M 时保存 `ckpt_<步数>_<exp_id>.zip`；lagrangian 同时把当时的 λ 写入 `ckpt_lambda.json`（评估检查点时要用同一个 λ）。",
            "- Saves `ckpt_<step>_<exp_id>.zip` the first time the step count reaches 5M / 10M / 15M; for lagrangian also writes the λ at that moment to `ckpt_lambda.json` (a checkpoint is evaluated with the same λ)."), "",
          T("这些 run 未使用的回调：`val_callback`（v6.1.0 调参用）、`GradDiagCallback`（梯度诊断）。",
            "Callbacks not used by these runs: `val_callback` (v6.1.0 tuning), `GradDiagCallback` (gradient diagnostic)."), "",
          T("**其他自定义组件（不是回调）**", "**Other custom components (not callbacks)**"), "",
          T("- `FastDummyVecEnv`（`src/paper_rl/vec_env.py`）：SB3 `DummyVecEnv` 去掉每步对 info 的深拷贝（改为一层拷贝，内容相同），只为提速。",
            "- `FastDummyVecEnv` (`src/paper_rl/vec_env.py`): SB3 `DummyVecEnv` without the per-step deep copy of the infos (a one-level copy with the same contents), for speed only."),
          T("- 每个训练环境的包装：`PlacementEnv` → SB3 `Monitor`（记录 episode 奖励）→ `TransformReward`（×1，即不缩放）。",
            "- Each training environment is wrapped `PlacementEnv` → SB3 `Monitor` (episode reward) → `TransformReward` (×1, i.e. no scaling)."),
          T("- `DoubleDQN` / `MaskableDQN` / `MaskableDDQN`：覆盖 `train()`（TD 目标）；Maskable 版本还覆盖 `predict()` 与 `_sample_action()`（掩码下的 ε-贪心）。",
            "- `DoubleDQN` / `MaskableDQN` / `MaskableDDQN`: override `train()` (TD target); the Maskable ones also override `predict()` and `_sample_action()` (masked ε-greedy)."), ""]
    return "\n".join(L)


def environment(T, ef):
    L = [T("# 4. 环境：变量、观测（状态）、动作空间", "# 4. Environment: variables, observation (state), action space"), "",
         T("代码：`src/paper_rl/env.py` 的 `PlacementEnv`（gymnasium.Env），12 个变种共用。", "Code: `PlacementEnv` (gymnasium.Env) in `src/paper_rl/env.py`, shared by all 12 variants."), "",
         T("## 4.1 实例（每次 reset 抽一个）", "## 4.1 Instance (one drawn at every reset)"), ""]
    L += table([T("变量", "variable"), T("含义", "meaning"), T("分布（数据生成）", "distribution (data generation)")], [
        ["c_n", T("ECU n 的容量（容器数），N 个", "capacity of ECU n (containers), N of them"), "uniform{50, 55, …, 195}"],
        ["e_m", T("服务 m 的需求（容器数），M 个", "demand of service m (containers), M of them"), "uniform{10, 15, …, 95}"],
        [T("冲突集", "conflict sets"), T("K = 10 个；同一集合中的两个服务不能同 ECU", "K = 10; two services of the same set may not share an ECU"),
         T("每个服务以概率 q 独立加入每个集合，少于 2 个成员的集合丢弃；两服务冲突概率 p = 1 − (1 − q²)^K = 0.6",
           "each service joins each set independently with probability q, sets with < 2 members dropped; two services conflict with probability p = 1 − (1 − q²)^K = 0.6")],
        ["ar_star", T("该实例的 ILP AR（Dinkelbach），只用于记录与评估，不进入观测", "the instance's ILP AR (Dinkelbach), for logging and evaluation only, not in the observation"), ""]]) + [""]
    L += [T("所有实例都经 ILP 解出最优（Optimal）才保留。", "Every instance is kept only if the ILP solves it to optimality."), "",
          T("## 4.2 环境内部变量（每个 episode）", "## 4.2 Internal variables (per episode)"), ""]
    L += table([T("变量", "variable"), T("含义", "meaning")], [
        ["t", T("当前步 = 当前要放置的服务（按需求降序后的位置），0 … M", "current step = the service to place (position in descending-demand order), 0 … M")],
        ["remaining_n", T("ECU n 的剩余容量 c_n − 负载（违约时可为负）", "free capacity c_n − load of ECU n (negative after a capacity violation)")],
        ["hosted_n", T("ECU n 上已执行放置的服务集合（含违约放置）", "services executed on ECU n (violating ones included)")],
        ["partners_m", T("与服务 m 至少共处一个冲突集的服务", "services sharing at least one conflict set with m")],
        ["legal_ru / legal_ecus", T("合法放置的 e_m / c_n 之和、承载合法放置的 ECU 集合 → AR", "sum of e_m / c_n over legal placements, ECUs hosting a legal placement → AR")],
        ["exec_ru / exec_ecus", T("所有执行放置（含违约）→ AR_obj（lagrangian 奖励）", "every executed placement (violations included) → AR_obj (lagrangian reward)")],
        ["valid_placed", T("合法放置的服务数", "number of legally placed services")],
        ["cap_violations / conflict_violations", T("容量 / 隐私违约次数", "capacity / privacy violation counts")],
        ["repairs", T("repair 替换动作的次数", "number of repair replacements")],
        ["dead_end", T("是否出现过「当前服务没有可行 ECU」", "whether a step with no feasible ECU occurred")],
        ["exited", T("是否以 EXIT 结束（只有 maskable）", "whether the episode ended by EXIT (maskable only)")],
        ["λ", T("对偶变量（只有 lagrangian），episode 内固定", "dual variable (lagrangian only), fixed within an episode")]]) + [""]
    L += [T("## 4.3 观测（observation）与状态（state）", "## 4.3 Observation and state"), "",
          T("- 环境输出的观测就是完整状态（完全可观测、马尔可夫）：`obs_mode = raw`（命令行虽写 `--obs base`，`--net graph` 会强制改为 raw，results.json 记录为 raw）。",
            "- The observation the environment emits is the full state (fully observable, Markov): `obs_mode = raw` (the command line says `--obs base`, but `--net graph` forces raw; results.json records raw)."),
          T("- 观测向量 = [c (N) | e，按放置顺序 (M) | 冲突图邻接矩阵 (M × M) | 每个服务的 ECU 编号 + 1，0 = 未放置 (M) | t (1)]，lagrangian 再在末尾加 λ / (1 + λ) (1)。",
            "- Observation vector = [c (N) | e in placement order (M) | conflict-graph adjacency (M × M) | ECU index + 1 of every service, 0 = unplaced (M) | t (1)], lagrangian appends λ / (1 + λ) (1)."),
          T("- 观测空间：`gymnasium.spaces.Box(low=-1, high=1e4, float32)`，数值未归一化（容量与需求是原始容器数）；归一化在网络内部做（02_network §2.2）。",
            "- Observation space: `gymnasium.spaces.Box(low=-1, high=1e4, float32)`, not normalised (capacities and demands are raw container counts); normalisation happens inside the network (02_network §2.2)."),
          T("- 关于「observer」：代码里没有单独的 observer 类。观测由 `PlacementEnv._raw_obs()` 生成；特征提取（可行性、负载、关系矩阵等）由网络的 `build()` 完成。"
            "PPO 与 DQN / DDQN 收到的观测和做的特征提取完全相同。",
            "- About an \"observer\": there is no separate observer class. The observation is produced by `PlacementEnv._raw_obs()`; feature extraction (feasibility, loads, relation matrices, ...) is done by the network's `build()`. "
            "PPO and DQN / DDQN receive the same observation and run the same feature extraction."), ""]
    L += [T("## 4.4 动作空间", "## 4.4 Action space"), "",
          T("- 离散动作空间 `gymnasium.spaces.Discrete`（不是连续的）：动作 a ∈ {0, …, N − 1} = 把当前服务放到 ECU a；maskable 多一个 EXIT 动作 a = N。",
            "- Discrete action space `gymnasium.spaces.Discrete` (not continuous): action a ∈ {0, …, N − 1} = place the current service on ECU a; maskable has one extra EXIT action a = N."),
          T("- 每步只决定一个服务放哪里（`action_mode = ecu`），服务顺序固定为需求降序。",
            "- Each step decides one service's ECU (`action_mode = ecu`); the service order is fixed (descending demand)."), ""]
    L += [T("## 4.5 各场景、各 strategy 的维度（由环境实例化得到）", "## 4.5 Dimensions per scenario and strategy (from instantiating the environment)"), ""]
    rows = [[s.upper(), STRAT[m], ef[s, m]["N"], ef[s, m]["M"], ef[s, m]["obs"], f"Discrete({ef[s, m]['act']})"] for s in SCENS for m in MECHS]
    L += table([T("场景", "scenario"), "strategy", "N", "M", T("观测维度", "observation size"), T("动作空间", "action space")], rows) + [""]
    L += [T("## 4.6 一步的执行顺序（`step`）", "## 4.6 One step (`step`)"), "",
          T("1. maskable 选 EXIT：标记 exited，给失败奖励，episode 结束。", "1. maskable chooses EXIT: mark exited, failure reward, the episode ends."),
          T("2. repair：动作不可行且存在可行 ECU 时替换为 e_m / c_n 最大的可行 ECU；没有可行 ECU 时保留原动作（记 dead_end）。",
            "2. repair: an infeasible action is replaced by the feasible ECU with the largest e_m / c_n when one exists; otherwise the action is kept (dead_end)."),
          T("3. 判定违约：容量（remaining_n < e_m）、隐私（ECU 上有冲突伙伴）；执行放置并更新 remaining、hosted、AR 相关量。",
            "3. Violations: capacity (remaining_n < e_m), privacy (a conflict partner on the ECU); the placement is executed and remaining, hosted and the AR terms are updated."),
          T("4. t ← t + 1；t = M 时 episode 结束并给终局奖励（01_algorithms §1.3）。lagrangian 每步再减 λ·c_t。",
            "4. t ← t + 1; at t = M the episode ends with the terminal reward (01_algorithms §1.3). lagrangian subtracts λ·c_t at every step."), ""]
    L += [T("## 4.7 12 个变种的区别", "## 4.7 Differences between the 12 variants"), ""]
    L += table([T("方面", "aspect"), "maskable", "repair", "lagrangian", "unconstraint"], [
        [T("观测", "observation"), T("原始状态", "raw state"), T("原始状态", "raw state"), T("原始状态 + λ / (1 + λ)", "raw state + λ / (1 + λ)"), T("原始状态", "raw state")],
        [T("动作空间", "action space"), "Discrete(N + 1)", "Discrete(N)", "Discrete(N)", "Discrete(N)"],
        [T("不可行动作", "infeasible action"), T("被掩码；无可行时只能 EXIT", "masked; EXIT only when none is feasible"),
         T("环境替换；无可行时执行并违约", "replaced by the env; executed (violation) when none is feasible"), T("执行，违约受 −λ·c_t 惩罚", "executed, violation charged −λ·c_t"),
         T("执行，违约只记录", "executed, violation only recorded")],
        [T("奖励", "reward"), "ar_pen", "ar_pen", "ar_obj − λ·c_t", "ar_pen"],
        [T("可能违约", "can violate"), T("否", "no"), T("仅死局", "dead ends only"), T("是", "yes"), T("是", "yes")]]) + [""]
    L += table([T("方面", "aspect"), "PPO", "DQN", "DDQN (Double DQN)"], [
        [T("类型", "type"), "on-policy actor-critic", "off-policy, replay + target network", "off-policy, replay + target network"],
        [T("输出", "output"), T("动作 logit（分类分布）+ V(s)", "action logits (categorical) + V(s)"), "Q(s, a)", "Q(s, a)"],
        [T("训练时探索", "exploration in training"), T("从分类分布采样（熵系数 0.005）", "sampling from the categorical distribution (entropy coefficient 0.005)"),
         T("ε-贪心，ε 在前 2M 步从 1 线性降到 0", "ε-greedy, ε linear from 1 to 0 over the first 2M steps"), T("同 DQN", "as DQN")],
        [T("评估时", "evaluation"), T("确定性：取最大 logit", "deterministic: argmax logit"), "argmax Q", "argmax Q"],
        [T("maskable 的掩码来源", "mask source for maskable"), T("环境 `action_masks()`（+ 网络内同一掩码）", "environment `action_masks()` (+ the same mask inside the network)"),
         T("网络从观测计算", "computed by the network from the observation"), T("网络从观测计算", "computed by the network from the observation")],
        [T("并行环境数", "parallel envs"), C.PPO_N_ENVS, C.DQN_N_ENVS, C.DQN_N_ENVS]]) + [""]
    L += [T("## 4.8 info 字段（每步返回，用于日志与评估）", "## 4.8 info fields (returned every step, for logging and evaluation)"), "",
          "`ar`, `ar_obj`, `ar_star`, `valid_placed`, `services_placed`, `capacity_violations`, `conflict_violations`, `repairs`, `cap_violated`, "
          "`conflict_violated`, `dead_end`, `exited`, `ecus_used`, `inst_idx`, `viol_rate_ep`, `cost_ep`", ""]
    return "\n".join(L)


def main():
    cell, src = runs()
    ver = versions()
    ppo_def, dqn_def = lib_defaults()
    nf, ef = net_facts(), env_facts()
    OUT.mkdir(parents=True, exist_ok=True)
    for lang, suf in (("zh", ".md"), ("en", ".en.md")):
        T = (lambda z, e: z) if lang == "zh" else (lambda z, e: e)
        docs = {"README": readme(T, cell, src), "01_algorithms": algorithms(T, cell, src), "02_network": network(T, nf),
                "03_training": training(T, cell, ver, ppo_def, dqn_def), "04_environment": environment(T, ef)}
        for name, text in docs.items():
            (OUT / f"{name}{suf}").write_text(text)
    print(OUT)
    if not ARGS.no_mail:
        from shared.report_mail import send_report
        send_report("v6.0.0", OUT / "README.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mail", action="store_true")
    ARGS = ap.parse_args()
    main()
