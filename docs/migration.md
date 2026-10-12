# 迁移手册：把实验环境搬到另一台 Linux 机器

本手册把 `my_code`（ILP vs RL 服务部署实验）迁到一台新的 Linux + NVIDIA GPU 机器上，迁完后可以直接接着跑 v5.x 实验。按"必需 → 验证 → 可选"三段走；每一步都写了预计耗时。

> 依据：2026-10-12 现机环境（Ubuntu 24.04.2、NVIDIA 驱动 535.309.01、Python 3.12.3、3 × GTX 1080 Ti、双路 E5-2660 v4）。

## 0. 预计总耗时

| 场景 | 内容 | 预计耗时 |
|---|---|---|
| **最小可训练**（新机已装好 NVIDIA 驱动） | 第 1–5 步 + 第 6 步验证 | **约 1–1.5 小时**（其中约 30 分钟是验证训练） |
| 加上面板、队列服务、外网访问 | 再做第 7–9 步 | **约 2–3 小时** |
| 新机还没装 NVIDIA 驱动 | 第 1 步要装驱动并重启 | **再加 30–60 分钟** |
| 拷全部历史结果（约 26 GB） | 第 5 步的可选部分 | 千兆局域网约 5–10 分钟；走公网可能数小时 |

时间主要花在：下载 GPU 版 PyTorch（约 2.5 GB）、拷贝数据、验证训练。

## 1. 新机器的前提（约 0–60 分钟）

- **系统**：Ubuntu 22.04 / 24.04（其他发行版也可以，命令要相应调整）。不建议 WSL2：不支持 MPS，而 PPO 阶段依赖 MPS 提速（实测 1.56×）。
- **GPU**：NVIDIA，驱动 **≥ 535**（支持 CUDA 12.x）。GPU 版 PyTorch 自带 CUDA 运行库，**不需要另装 CUDA Toolkit**。
  ```bash
  nvidia-smi                      # 能看到 GPU 和驱动版本即可；看不到就先装驱动：
  sudo ubuntu-drivers install     # 装推荐驱动，装完重启
  ```
- **内存**：每个训练进程约 2–3 GB。同时跑 27–36 个需要约 90 GB；内存少就按比例减少并发（`V5_PARALLEL`，见第 6 步）。
- **磁盘**：代码 + 环境约 10 GB；实验结果每 100 个 run 约 15 GB。

## 2. 系统软件包（约 5 分钟，需要 sudo）

```bash
sudo apt update
sudo apt install -y git python3.12 python3.12-venv build-essential numactl
curl -LsSf https://astral.sh/uv/install.sh | sh      # uv：建 GPU 环境用
```
ILP 求解器 CBC 由 PuLP 自带，不用另装。

## 3. 取代码（约 5 分钟）

新机需要能访问 GitHub 仓库（把新机的 SSH 公钥加到 GitHub 账号，或用 HTTPS + token）。

```bash
mkdir -p ~/github && cd ~/github
git clone git@github.com:Wk-P/I2R.my_code.git my_code
cd my_code
git checkout main && git log --oneline -1          # 应与现机一致（迁移前先在现机 push）
```
**迁移前在现机执行 `git push`（含 tag：`git push --tags`）**，否则新机拿不到最新代码和 v5.0.0 / v5.0.2 等 tag。

## 4. 两个 Python 环境（约 15–30 分钟，主要是下载）

项目用两个环境：`.venv`（CPU：面板、队列、报告、MLP 训练）和 `.venv-gpu`（GPU：结构感知网络训练）。

```bash
cd ~/github/my_code

# 4.1 CPU 环境
python3.12 -m venv .venv
# requirements.txt 固定了 torch==2.10.0+cpu，它只在 PyTorch 自己的源上，所以要加 --extra-index-url
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e .                          # 让 src/ 下的包可导入

# 4.2 GPU 环境（PyTorch 2.5.1 + CUDA 12.1 运行库）
uv venv .venv-gpu --python 3.12
uv pip install --python .venv-gpu/bin/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
grep -v -i "^torch==" requirements.txt > /tmp/req-gpu.txt
uv pip install --python .venv-gpu/bin/python -r /tmp/req-gpu.txt
uv pip install --python .venv-gpu/bin/python -e .

# 4.3 检查
.venv-gpu/bin/python -c "import torch, stable_baselines3 as s, sb3_contrib as c; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), s.__version__, c.__version__)"
# 期望：2.5.1+cu121 True <GPU 数> 2.7.1 2.7.1
```

版本必须和现机一致（stable-baselines3 / sb3-contrib 2.7.1、gymnasium 1.2.3、PyTorch 2.5.1），否则训练结果不可比。

## 5. 拷贝不在 git 里的数据（约 5–15 分钟）

| 内容 | 大小 | 是否必需 | 用途 |
|---|---|---|---|
| `results/sup_diag/{lt,eq,gt}.json` | 约 50 MB | **必需** | 5 万实例训练池（经 ILP 验证） |
| `logs/v4.5.2/`、`logs/v4.4.15`–`v4.4.17/` 等 manifest | 几 MB | **必需** | 报告读参照数据靠它们 |
| v4.5.2 的 18 个结构感知 Maskable DQN/DDQN 结果目录 | 约 2.4 GB | **v5.0.2 必需** | v5.0.2 直接复用它们作 Maskable DQN/DDQN |
| `.env.email` | 几百字节 | 要发邮件就必需 | 报告完成 / 任务失败的邮件凭据（**勿提交到 git**） |
| `results/unified/` 其余部分 | 约 18 GB | 可选 | 面板上的历史结果、旧版本复现 |

在**现机**上执行（把 `NEW` 换成新机的 `用户@地址`）：

```bash
cd ~/github/my_code
NEW=user@new-host
rsync -avP results/sup_diag  $NEW:~/github/my_code/results/
rsync -avP logs/             $NEW:~/github/my_code/logs/ --exclude 'mps/'
rsync -avP .env.email        $NEW:~/github/my_code/
# v4.5.2 的 18 个参照 run（v5.0.2 需要）：
python3 - <<'EOF' > /tmp/ref_dirs.txt
import json
m = json.load(open("logs/v4.5.2/manifest.json"))
for sd in m["exp_ids"].values():
    for s, d in sd.items():
        for k, v in d.items():
            if k.endswith("graph"):
                print(f"results/unified/{s}/{'mask_dqn' if k.startswith('dqn') else 'mask_ddqn'}/{v}")
EOF
rsync -avP --files-from=/tmp/ref_dirs.txt -r . $NEW:~/github/my_code/
# 可选：全部历史结果
# rsync -avP results/unified $NEW:~/github/my_code/results/
```

## 6. 验证（约 30 分钟，**必做**）

新机的 GPU 型号或驱动可能和现机不同，必须先确认环境能正确训练。

```bash
cd ~/github/my_code

# 6.1 一致性：v5 代码与 v5 之前的代码在新机上训练逐位一致（CPU + GPU，约 10 分钟）
git worktree add ../my_code-ref 9aa6bc7             # v5.0.0 之前的参照代码
.venv/bin/python src/scripts/check_v5.py identical --old-src ../my_code-ref/src
cat logs/check_v5/identical.md                      # 每一行都应为"是"
git worktree remove ../my_code-ref

# 6.2 冒烟：用极小步数把 v5.0.2 的两个阶段、中间报告、最终报告、ILP 计时完整走一遍（约 20 分钟）
RESULTS_SPACE=smoke_v502 V5_SMOKE=1 .venv/bin/python src/scripts/run_v5.py --version 5.0.2
ls results/smoke_v5.0.2/report.md                   # 生成即通过
```

两项都通过，新机就可以跑正式实验。

**关于跨机器的可比性**：同一台机器上，同一种子的结果逐位可复现；**换了 GPU 型号或驱动，同一种子的结果不再逐位相同，但统计上等价**。同一组对比实验（例如一个 v5.0.x 的全部 run）最好在同一台机器上跑完；确实分在两台机器上时，在报告里注明每个 run 用的机器。

**新机资源不同时**：`run_v5.py` 每个阶段的并发数默认 27 / 36，按新机调整：
```bash
V5_PARALLEL=12 .venv/bin/python src/scripts/run_v5.py --version 5.0.2   # 例如内存或 GPU 较少的机器
```
求解时间对比（模型 vs ILP）要全部在同一台机器上测，否则时间不可比。

## 7. 可选：机器调优（约 10 分钟，需要 root）

与现机相同的三项（GPU 持久模式、CPU performance 调频、开机设风扇）。风扇用 NVML 设置，需要先建一个小环境：

```bash
sudo python3 -m venv /root/nvfan && sudo /root/nvfan/bin/pip install -q nvidia-ml-py
sudo tee /etc/systemd/system/gpu-tune.service > /dev/null <<'EOF'
[Unit]
Description=GPU persistence mode + CPU performance governor
After=multi-user.target
[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
ExecStart=/bin/sh -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $g; done'
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now gpu-tune.service
```
风扇转速是否需要手动设定视新机散热而定（现机放在角落，GPU 0 会顶到 84 °C 降频，所以设成了固定转速）。

## 8. 可选：面板与实验队列（约 15 分钟）

面板（训练监控、图表、版本报告）和串行实验队列是用户级 systemd 服务。下面的路径按新机的用户名和目录修改。

```bash
cd ~/github/my_code/app/frontend && npm ci && npm run build && cd ~/github/my_code   # 需要 Node.js 18+
mkdir -p ~/.config/systemd/user
# 从现机拷贝 4 个 unit 文件，或按现机内容新建：
#   my-code-panel.service    面板（uvicorn，端口 8081）
#   my-code-queue.service    串行实验队列（KillMode=process：重启服务不会杀实验）
#   my-code-monitor.service  卡住 / 完成检测
#   cloudflared-my-code.service  外网访问（见第 9 步）
sed -i "s#/home/soar009#$HOME#g" ~/.config/systemd/user/my-code-*.service
loginctl enable-linger $USER                        # 退出登录后服务继续运行（需要 sudo 时加 sudo）
systemctl --user daemon-reload
systemctl --user enable --now my-code-panel my-code-queue my-code-monitor
curl -s localhost:8081/api/system | head -c 200     # 有输出即面板正常
```

## 9. 可选：外网访问（约 15–30 分钟）

现机通过 Cloudflare Tunnel 把面板发布在 `my-code.knettf.com`。新机二选一：
- **沿用同一个隧道**：拷贝 `~/.cloudflared/my-code.yml` 和它引用的凭据 json（`~/.cloudflared/<隧道 ID>.json`），安装 `cloudflared` 后启用 `cloudflared-my-code.service`。同一个隧道同时在两台机器上运行时，访问会被分到两台机器，**迁完后在旧机上停掉它**。
- **新建隧道**：`cloudflared tunnel login` → `cloudflared tunnel create` → 配置 DNS。需要 Cloudflare 账号权限。

## 10. 迁移后接着跑实验

```bash
.venv/bin/python src/scripts/queue_runner.py list                      # 看队列
.venv/bin/python src/scripts/queue_runner.py add v5.0.2 --log logs/run_v5.0.2_driver.log -- \
    .venv/bin/python src/scripts/run_v5.py --version 5.0.2
```
报告生成在 `paper_contents/v<版本>/report.md` 与 `report.en.md`，完成后自动发邮件（需要第 5 步的 `.env.email`）。

## 附：现机关键版本

| 项目 | 版本 |
|---|---|
| 系统 | Ubuntu 24.04.2 LTS，内核 6.17 |
| NVIDIA 驱动 | 535.309.01（CUDA 12.2） |
| Python | 3.12.3 |
| PyTorch（GPU 环境） | 2.5.1 + cu121 |
| stable-baselines3 / sb3-contrib | 2.7.1 / 2.7.1 |
| gymnasium | 1.2.3 |
| PuLP（CBC） | 3.3.0 |
| Node.js（面板前端构建） | 22.21.1 |
