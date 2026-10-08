"""量化训练曲线的形状（不只是看头尾数值）。

读 training_curve.csv（scenarios/*/ppo_mask/run_all.py 训练完会自动存），
把训练进度切成 10 等分（deciles），分别看：
  - 全部 episode 的 AR 均值和 success rate 均值（原始的、混合的趋势）
  - 只看 success==1 的 episode 的 AR 均值（排除"失败拉低均值"这个构成效应，
    单独看奖励公式 M*(2ar-1) 实际在成功样本里有没有把AR往上推）

用法: python src/scripts/analyze_curve_shape.py <training_curve.csv> [<csv2> ...]
"""
import csv
import sys
from pathlib import Path


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "timestep": int(r["timestep"]),
                "ar": float(r["episode_ar"]),
                "success": int(r["episode_success"]),
            })
    return rows


def bin_means(rows, n_bins, key, filt=None):
    data = [r for r in rows if (filt is None or filt(r))]
    n = len(data)
    if n == 0:
        return [None] * n_bins
    bin_size = max(1, n // n_bins)
    means = []
    for b in range(n_bins):
        chunk = data[b * bin_size: (b + 1) * bin_size if b < n_bins - 1 else n]
        if not chunk:
            means.append(None)
            continue
        means.append(sum(r[key] for r in chunk) / len(chunk))
    return means


def analyze(path):
    rows = load(path)
    n = len(rows)
    if n < 20:
        print(f"{path}: too few episodes ({n}), skipping")
        return

    n_bins = 10
    all_ar = bin_means(rows, n_bins, "ar")
    succ_rate = bin_means(rows, n_bins, "success")
    succ_only_ar = bin_means(rows, n_bins, "ar", filt=lambda r: r["success"] == 1)

    print(f"\n{path}  (n={n} episodes)")
    print(f"{'decile':<8}{'success_rate':<14}{'all_ep_AR':<14}{'success_only_AR':<16}")
    for i in range(n_bins):
        sr = succ_rate[i]
        a = all_ar[i]
        sa = succ_only_ar[i]
        print(f"{i+1:<8}{(sr if sr is not None else float('nan')):<14.4f}"
              f"{(a if a is not None else float('nan')):<14.4f}"
              f"{(sa if sa is not None else float('nan')):<16.4f}")

    valid_sa = [v for v in succ_only_ar if v is not None]
    if len(valid_sa) >= 4:
        early = sum(valid_sa[:len(valid_sa)//2]) / (len(valid_sa)//2)
        late = sum(valid_sa[len(valid_sa)//2:]) / (len(valid_sa) - len(valid_sa)//2)
        print(f"  success-only AR: early-half avg={early:.4f}  late-half avg={late:.4f}  delta={late-early:+.4f}")
        if late > early + 0.005:
            print("  -> within successful episodes, AR IS trending up over training (reward's ar-term is working).")
        elif late < early - 0.005:
            print("  -> within successful episodes, AR is trending DOWN -- the mixing-effect theory is wrong, something else is going on.")
        else:
            print("  -> within successful episodes, AR is flat -- reward's ar-term isn't visibly pushing quality up either.")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(Path(p))
