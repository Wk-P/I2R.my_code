# Docker 一键部署 / 迁移

把整个实验环境（训练、面板、实验队列、可选的 Cloudflare 外网访问）打包成一个 Docker 镜像。换机器时：装好驱动和 Docker → 拿到代码 → 恢复数据 → 一条命令启动。

## 一键部署（新机器）

前提：NVIDIA 驱动 ≥ 535、Docker（含 compose 插件）、NVIDIA Container Toolkit（让容器用 GPU）。

```bash
git clone git@github.com:Wk-P/I2R.my_code.git my_code && cd my_code
deploy/install.sh --restore --verify --tunnel
```
- `--restore`：先从云端（`RCLONE_REMOTE`）拉回实验数据；
- `--verify`：启动后在容器里做训练逐位一致检查 + v5.0.2 冒烟测试（约 35 分钟，**新机器第一次必做**）；
- `--tunnel`：自动建立 / 复用 Cloudflare 隧道并启动外网访问。

第一次运行会生成 `deploy/.env`（从 `.env.example` 复制），按需填写后再运行一次即可。耗时：构建镜像约 10–20 分钟（主要是下载 PyTorch），恢复数据视数据量和网速而定，验证约 35 分钟。

## 目录与数据

| 位置 | 内容 |
|---|---|
| 镜像里 | 代码、两个 Python 环境（CPU / GPU，与现机版本完全一致）、构建好的面板前端 |
| `DATA_ROOT`（宿主机目录，挂载进容器） | `results/`（模型、checkpoint、结果）、`logs/`（manifest、日志、队列）、`paper_contents/`（报告）、`config/.env.email`（邮件凭据） |

镜像只放代码，数据永远在宿主机上：重建或升级镜像不会丢数据。

**镜像标签 = 代码版本**：在 `deploy/.env` 里设 `IMAGE_TAG=v5.0.2` 再构建，之后每个结果都能对应到生成它的镜像。

## 日常操作

```bash
docker exec mycode-lab .venv/bin/python src/scripts/queue_runner.py list          # 看队列
docker exec mycode-lab .venv/bin/python src/scripts/queue_runner.py add v5.0.2 --log logs/run_v5.0.2_driver.log -- \
    .venv/bin/python src/scripts/run_v5.py --version 5.0.2                       # 排实验
docker logs -f mycode-lab                                                         # 容器日志
deploy/data_sync.sh backup                                                        # 备份数据到云端
```
**注意**：停止、重启或重新创建 `mycode-lab` 容器会结束正在跑的全部实验。改代码后要更新镜像，请在没有实验运行时进行。

## 数据和日志放在哪里

| 位置 | 放什么 | 不放什么 |
|---|---|---|
| **GitHub** | 代码、报告（`paper_contents/` 的 md / json）、小的配置 | 模型、checkpoint、训练日志：体积太大（`results/` 约 26 GB；GitHub 单文件上限 100 MB，仓库建议 1 GB 以内） |
| **本机硬盘**（`DATA_ROOT`） | 全部实验数据的工作副本：训练直接读写，速度最快 | — |
| **云端对象存储**（推荐 Cloudflare R2） | 全部实验数据的备份；换机器时从这里恢复 | — |
| **（可选）第二块硬盘** | 本机数据的离线备份 | — |

推荐做法（3-2-1 备份原则：3 份数据、2 种介质、1 份异地）：本机硬盘做工作副本，Cloudflare R2 做异地备份和迁移中转，重要节点再拷一份到移动硬盘。

为什么推荐 R2：已经在用 Cloudflare；与 S3 兼容（rclone 直接支持）；**下载不收流量费**，换机器整包拉回不额外花钱；存储费约 0.015 美元 / GB / 月，26 GB 约每月 0.4 美元，前 10 GB 免费。Google Drive 等网盘也能用（rclone 同样支持），但大量小文件时速度较慢。

论文发表时，最终模型和 5 万实例数据集可以另外发布到 Zenodo（有 DOI，可引用），这是可复现性的最终存档，和上面的日常备份分开。

**配置 R2**（一次性）：Cloudflare 控制台 → R2 → 建桶（如 `mycode-data`）→ 管理 API 令牌（对象读写）→ 得到 Access Key ID / Secret / Endpoint，然后：
```bash
rclone config      # 新建 remote，名字 r2，类型 s3，provider Cloudflare，填上面三项
deploy/data_sync.sh backup ~/github/my_code      # 现机（非 Docker 布局）首次整包上传
```

## Cloudflare 外网访问的自动化

`deploy/cloudflare_setup.py` 用 Cloudflare API 自动完成：建立（或复用）远程管理的隧道 → 设置转发（主机名 → 面板）→ 建立 DNS 记录 → 把隧道令牌写进 `deploy/.env`。之后 `cloudflared` 容器用这个令牌连接，不需要在机器上做浏览器登录。

需要一个 API 令牌（Cloudflare 控制台 → 我的个人资料 → API 令牌），权限：Account › Cloudflare Tunnel › Edit、Zone › DNS › Edit。填进 `deploy/.env` 的 `CF_API_TOKEN`、`CF_HOSTNAME`。

```bash
python3 deploy/cloudflare_setup.py            # 先看会改什么（默认只预览）
python3 deploy/cloudflare_setup.py --apply    # 执行
```
脚本可以重复运行（复用同名隧道和 DNS 记录）。如果主机名已经指向别的隧道（例如旧机器），脚本会拒绝修改，除非加 `--yes`：那会把外网流量切到这台新机器。

## 宿主机调优（可选，需要 root）

GPU 持久模式、CPU performance 调频、风扇设定属于宿主机设置，不在容器里。见 `docs/migration.md` 第 7 步。
