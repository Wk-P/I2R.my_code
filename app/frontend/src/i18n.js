// 中英切换：不改模板，在 DOM 层把中文片段替换成英文（最长匹配优先）。
// 原文存在 WeakMap 里，切回中文时还原；Vue 更新文本后自动重新翻译。
// 论文草稿（article.paper）与后端返回的文档正文不翻译。
import { ref } from "vue";
import { DATA } from "./i18n_data.js";

const D = {
  ...DATA,
  // docs/data_spec.md (manifest v2): roles, params, waiting, test batches
  "正式": "main", "参数扫描": "sweep", "参照": "reference", "来源": "source", "参数": "params", "λ_max 扫描": "λ_max sweep",
  "等待依赖（例如等参数扫描结束）": "waiting for a dependency (e.g. a parameter sweep)", "等待依赖": "waiting", "等待 ": "waiting for ",
  "运行 / 排队 / 等待依赖 / 中断": "running / queued / waiting / interrupted",
  // navigation (sidebar groups), batches by stage, queue on the overview
  "当前": "Now", "实验": "Experiments", "分析": "Analysis", "论文": "Paper", "进程监控": "Processes", "批次": "Batches",
  "版本报告": "Version reports", "结果对比": "Compare results", "版本报告 →": "version report →", "结果对比 →": "compare results →",
  "批次 →": "batches →", "进程监控 →": "processes →", "在结果对比中查看 →": "view in Compare results →", "全部 run": "All runs",
  "正在进行的批次、实验队列与最新结果": "running batches, the experiment queue and the latest results",
  "每次启动的训练 / 评估为一个批次，按版本分组；点击批次查看结果矩阵与各 run 的状态":
    "each training / evaluation launch is a batch, grouped by version; click a batch for its result matrix and run status",
  "阶段": "Stage", "全部阶段": "all stages", "（当前）": " (current)", "版本 ": "version ",
  "实验队列": "Experiment queue", "待执行的项目 →": "pending items →", "已暂停": "paused", "等待前置任务": "waiting for prerequisite",
  "队列已暂停：排队的项目不会自动启动（": "Queue paused: queued items will not start on their own (", "恢复）": " resumes)",
  "队列中没有待执行的项目": "nothing pending in the queue", "项目": "item", "前置条件": "waits for", "数据空间": "data space",
  "结果所在的阶段（results/ 下的目录）": "the stage the results belong to (a directory under results/)",
  // v5.0.3: per-learner lambda_max sweep, adaptive lambda
  "λ 自适应 · 上限 ": "λ adaptive · cap ", "末 λ=": "final λ=", "待定": "pending", "选定": "chosen",
  "（每个 learner 各扫一次，": "(one sweep per learner, ", "λ_max 扫描（": "λ_max sweep (",
  "λ 在训练中按局对偶上升自适应，λ_max 只是它的上限。每个 learner 选三场景平均\"有效 gap\"最低的上限（有违约的实例按 AR = 0 计），该 learner 的 Lagrangian 格子随后开始训练；详见报告 lam_sweep_<learner>.md。":
    "λ adapts per episode by dual ascent during training; λ_max is only its cap. Each learner takes the cap with the lowest mean \"effective gap\" over the three scenarios (a violating instance counts as AR = 0); its Lagrangian cells start training then; see the report lam_sweep_<learner>.md.",
  // batch detail: result matrix / sweep / run details
  "结果矩阵": "Result matrix",
  "每格一个模型（场景 × 模型 × 约束处理）：相对 gap 与测试集违约率；未完成的格子显示当前状态":
    "one model per cell (scenario × model × constraint handling): relative gap and test violation rates; unfinished cells show their status",
  "相对 gap = (ILP AR − AR) / ILP AR，只在无违约的测试实例上计算；容量 / 隐私 = 有容量 / 隐私违约的测试实例比例。点击已完成的格子查看该 run。":
    "relative gap = (ILP AR − AR) / ILP AR, on violation-free test instances only; capacity / privacy = share of test instances with a capacity / privacy violation. Click a finished cell to open its run.",
  "参照 · 非本批次训练": "reference · not trained in this batch", "个种子": " seeds", "隐私 ": "privacy ", "容量 ": "capacity ",
  "参数扫描：λ_max": "Parameter sweep: λ_max", "选定 λ_max=": "chosen λ_max=", "完成 ": "done ", "步；": " steps; ",
  "· 扫描结束后自动选定，Lagrangian 各格随后开始训练": "· chosen automatically when the sweep ends; the Lagrangian cells start training then",
  "选择规则：三场景平均\"有效 gap\"最低（有违约的实例按 AR = 0 计），详见报告 lam_sweep.md。":
    "Selection: lowest mean \"effective gap\" over the three scenarios (a violating instance counts as AR = 0); see the report lam_sweep.md.",
  "（每个 run 的种子、状态与全部指标，共": "(seeds, status and all metrics of every run; ", "组）": " groups)", "类别": "kind",
  "区分同一模型 × 约束处理下不同 run 的参数（docs/data_spec.md）": "parameters that tell runs of the same model × constraint handling apart (docs/data_spec.md)",
  "显示测试批次（smoke 等": "show test batches (smoke etc., ", "个）": ")", "冒烟测试": "smoke test", "测试": "test", "步数=": "steps=",
  "次数": "count", "该版本没有独立文档": "this version has no separate doc", "顶栏切换": "switch in the top bar",
  "上一页": "Prev", "下一页": "Next", "不存在": "not found",
  "个任务完成": " jobs done", "个任务，调度器按空闲 CPU 依次启动": " jobs; the scheduler starts them as CPUs free up",
  "个并行进程 · 开始于": " parallel processes · started ", "个批次 →": " batches →", "个批次": " batches", "个版本": " versions",
  "个解中至少有一个成功的测试实例比例": " solutions: share of test instances with at least one success",
  "个解取最好：先选成功的，再选 AR 最高的；ILP AR 只用于计分。K = 1 即确定性输出（贪心为纯贪心规则）。AR 与 ILP AR 都只在成功的实例上平均，AR gap = ILP AR − AR。指标为各种子的均值 ± 标准差。":
    " solutions, keep the best: successful first, then highest AR; ILP AR is only used for scoring. K = 1 is the deterministic output (greedy = plain greedy rule). AR and ILP AR are averaged over successful instances only, AR gap = ILP AR − AR. Mean ± std over seeds.",
  "中位数": "median", "中断": "interrupted", "供需均衡": "balanced", "全部场景": "all scenarios", "全部实验": "all runs",
  "全部批次 →": "all batches →", "全部结果 →": "all results →", "全部完成后生成）。指标在整个评估结束、写出原始结果后显示；运行中只显示各任务的状态与单次耗时。":
    "generated when everything finishes). Metrics appear after the whole evaluation writes its raw results; while running only job status and per-run time are shown.",
  "全部": "all", "共": "total ", "其他项目进程": "other project processes", "冲突违规": "privacy violation", "分支": "branch",
  "切换要查看的数据空间（results/ 下的目录，不会执行 git checkout）": "Switch the data space to view (a directory under results/; no git checkout)",
  "加载中": "Loading", "单个 episode（一个测试实例、一个解）的平均耗时，单线程": "mean time per episode (one test instance, one solution), single thread",
  "单元格为筛选范围内所有运行的均值 ± 样本标准差（n 为运行数），绿色为该列最优。点击单元格进入对应明细。AR = average resource utilization（优化目标），AR gap = ILP AR − AR；success_rate = M 个服务全部合法放置且无违规的测试实例比例。跨批次汇总时请先在左侧限定批次，避免混入不同代码版本。":
    "Each cell is the mean ± sample std over all runs in the filter (n = runs); green is the best in its column. Click a cell for details. AR = average resource utilization (the objective), AR gap = ILP AR − AR; success_rate = share of test instances with all M services placed legally. When aggregating across batches, restrict the batch on the left first to avoid mixing code versions.",
  "单次耗时（逐个运行；合并成 batch 时更低）": "time per run (one by one; lower when batched)", "单次耗时 ms": "time per run ms",
  "只在成功的测试实例上平均": "averaged over successful test instances only", "只读面板，不影响任何训练进程": "read-only panel; does not affect any training process",
  "同一批成功实例上 ILP 最优解的 AR": "AR of the ILP optimum on the same successful instances", "同组其他种子": "other seeds in this group",
  "在实验结果中对比 →": "compare in Results →", "场景": "scenario", "基本信息": "basic info", "奖励模式": "reward mode",
  "完成于": "finished at", "完成时间": "finished", "完成": "done", "实验管理台": "Experiment Console", "实验结果": "Results",
  "容量 / 冲突违规": "capacity / privacy violation", "容量违规": "capacity violation",
  "小于 1× 时标红，表示比 ILP 还慢。ILP 耗时来自 v4.3.1.7，与本批次不是同一时间、同一负载下测得。":
    "Red below 1× means slower than ILP. ILP time is from v4.3.1.7, not measured at the same time or load as this batch.",
  "小时": " h", "层级：版本 → 批次 → 场景 → 算法 → 单次运行 · 先选范围（版本或批次），再按场景、算法筛选，可切换明细 / 汇总对比":
    "Hierarchy: version → batch → scenario → algorithm → run · pick a scope (version or batch), then filter by scenario and algorithm; switch between details / summary",
  "已中断": "interrupted", "已作废": "discarded", "已完成（可点击查看）": "done (click to view)", "已完成实验": "finished runs", "已完成": "done",
  "已用时": "elapsed", "已运行": "running for", "平均耗时 ÷ 本方法耗时": "mean time ÷ this method's time",
  "平均资源利用率，优化目标）": "average resource utilization, the objective)", "开始时间": "started", "开始": "start",
  "当前没有训练任务在运行": "no training job is running", "当前没有运行中的批次": "no batch is running", "总览": "Overview", "总进度": "overall progress",
  "成功率": "success rate", "手动启动": "manual", "批次 / 算法": "batch / algorithm", "批次管理": "Batches", "批次": "batch",
  "找不到批次": "batch not found", "找不到版本": "version not found", "找不到该运行（": "run not found (", "报告：": "Report: ",
  "指标为已完成种子的均值 ± 标准差；AR = average resource utilization；AR gap = ILP AR − AR；success_rate = M 个服务全部合法放置且无违规的测试实例比例":
    "Metrics are mean ± std over finished seeds; AR = average resource utilization; AR gap = ILP AR − AR; success_rate = share of test instances with all M services placed legally",
  "排队中": "queued", "排队": "queued", "搜索 exp_id / 批次名 / 算法，回车": "search exp_id / batch / algorithm, Enter",
  "搜索版本号或摘要": "search version or summary", "搜索": "search", "摘要": "summary", "数据空间": "data space", "文档": "doc",
  "新训练写入此处": "new runs write here", "新训练写入": "new runs write to", "方法": "method", "无约束": "unconstrained", "日期": "date",
  "时间": "time", "明细": "details", "更新": "updated", "最优）": "best)", "最后更新": "last update", "最近完成的实验": "recently finished runs",
  "最近结束的批次": "recently finished batches", "服务部署": "service placement", "未归属批次）": "no batch)", "未标注版本": "untagged",
  "未运行（批次已结束）": "not run (batch ended)", "未运行（评估已结束）": "not run (evaluation ended)", "未运行": "not run",
  "本机正在运行的项目进程，每 5 秒刷新": "project processes running on this machine, refreshed every 5 s", "条运行": " runs",
  "查看训练监控 →": "training monitor →", "查看该版本结果 →": "results of this version →", "查看这些运行": "view these runs",
  "核（共": " cores (of ", "步 · N/M =": " steps · N/M =", "步 · 开始于": " steps · started ", "步数": "steps", "模型": "model",
  "每个 git tag 一行，摘要来自 version/VERSION.md，有独立文档的可点击查看": "one row per git tag; summaries from version/VERSION.md; rows with their own doc are clickable",
  "每个测试实例取 K 个解中最好的一个；K = 1 即确定性输出": "best of K solutions per test instance; K = 1 is the deterministic output",
  "比 ILP 快": "faster than ILP", "汇总对比": "summary", "没有排队中的任务": "no queued jobs", "没有符合条件的批次": "no matching batch",
  "没有符合条件的运行": "no matching run", "测试实例中 M 个服务全部合法放置且无违规的比例": "share of test instances with all M services placed legally",
  "测试对比": "test comparison", "测试集）": "test set)", "清除筛选（": "clear filters (", "版本 / 批次": "version / batch", "版本记录": "Versions",
  "版本": "version", "状态": "status", "的批次与最新结果；训练任务、CPU 为本机全局状态": " batches and latest results; jobs and CPU are machine-wide",
  "的训练批次与评估批次，按版本分组；点击查看各任务的状态与汇总结果": " training and evaluation batches, grouped by version; click for job status and summaries",
  "种子": "seed", "算法": "algorithm", "类型": "type", "系统负载 1/5/15 分钟": "system load 1/5/15 min", "约束处理": "constraint handling",
  "结果目录": "result dir", "耗时 ms/实例": "time ms/instance", "范围：": "Scope: ", "菜单": "Menu", "训练中任务": "training jobs",
  "训练中": "Training", "图表": "Charts", "冻结": "frozen", "冻结中": "frozen", "最近约 30 万步的速度（由日志反推）；悬停单元格可看自启动以来的平均速度": "speed over the last ~300k steps (derived from the log); hover a cell for the average since start", "自启动以来平均": "average since start", "训练任务": "training jobs", "训练占用 CPU": "training CPU", "训练回合数": "training episodes", "训练曲线": "training curve",
  "训练末 50 回合 AR": "AR of last 50 training episodes", "训练 / 测试实例数": "train / test instances", "训练监控": "Monitor",
  "训练进度来自各任务日志中最近一次 [train] 记录（旧实现每 20 万步、paper_rl 每 10 万步写一次），刚启动或处于 ILP / 评估阶段时显示 0%。":
    "Progress comes from the latest [train] line in each job log (every 200k steps in the old code, every 100k in paper_rl); 0% right after start or during ILP / evaluation.",
  "训练进度": "training progress", "训练": "training", "论文草稿": "Paper Draft", "评估批次（不训练）· 版本": "evaluation batch (no training) · version",
  "评估": "evaluation", "负载": "load", "贪心": "greedy", "资源充裕": "abundant", "资源紧缺": "scarce", "运行中批次": "running batches",
  "运行中（数字为训练进度 %）": "running (number = training progress %)", "运行中": "running", "运行 / 排队 / 中断": "running / queued / interrupted",
  "运行 / 排队": "running / queued", "运行详情": "Run Detail", "运行": "run", "进度": "progress", "进程": "process", "选择范围：": "Scope: ",
  "阶段5 · 论文最终实验": "Stage 5 · final paper experiments", "选择": "select",
  // v4.4.3+ UI (GPU status, network / device, relative gap, EXIT) -- whole sentences first, so they are not split
  "指标为已完成种子的均值 ± 标准差；AR = average resource utilization；AR gap = ILP AR − AR；相对 gap = (ILP AR − AR) / ILP AR；success_rate = M 个服务全部合法放置且无违规的测试实例比例":
    "Metrics are mean ± std over finished seeds; AR = average resource utilization; AR gap = ILP AR − AR; relative gap = (ILP AR − AR) / ILP AR; success_rate = share of test instances with all M services placed legally",
  "奖励模式（早期试跑）或策略网络（v4.4.3：结构感知网络 / MLP）": "reward mode (early pilots) or policy network (v4.4.3: structure-aware network / MLP)",
  "奖励模式（早期试跑）或策略网络（v4.4.3）": "reward mode (early pilots) or policy network (v4.4.3)",
  "Maskable 选了 EXIT（无合法 ECU，回合结束）的测试实例比例；v4.3.8 之前没有 EXIT 动作":
    "share of test instances where Maskable chose EXIT (no legal ECU, the episode ends); there is no EXIT action before v4.3.8",
  "Maskable 选了 EXIT 的测试实例比例": "share of test instances where Maskable chose EXIT",
  "日志里最近一次 [train] 记录的平均速度（自启动以来）": "average speed in the latest [train] log line (since start)",
  "单个进程占用的 CPU，按单核计：100% = 占满 1 个核（整机共 56 核）。GPU 训练的进程同样会占满 1 个核：环境推进、动作掩码与 rollout 循环都在 CPU 上":
    "CPU of a single process, per core: 100% = one full core (56 cores in total). GPU training processes also keep one core busy: environment steps, action masks and the rollout loop run on the CPU",
  "CPU（单核）": "CPU (per core)",
  "同一命令下所有进程的 CPU 之和，按单核计：100% = 占满 1 个核（整机共 56 核）":
    "sum of the CPU of every process with this command, per core: 100% = one full core (56 cores in total)",
  "CPU（单核合计）": "CPU (per-core sum)", "生成报告（训练结束后）": "Report (after training)", "报告诊断占用 CPU": "report diagnostics CPU",
  "进程数": "processes", "阶段": "stage",
  // v4.4.14 panel redesign
  "每个版本一行：改了什么、新变体在三个场景的测试集相对 gap 和相对基线的变化；点击查看完整报告。目标：三场景 gap 都降到 5% 以下。":
    "One row per version: what changed, the new variant's test relative gap in the three scenarios and the change against its baseline; click for the full report. Target: all three gaps below 5%.",
  "个 tag，摘要来自 version/VERSION.md": " tags, summaries from version/VERSION.md", "全部 git 版本": "All git versions",
  "搜索版本号或改动…": "search version or change…", "训练或生成报告中 · 实验版本 →": "training or building the report · versions →",
  "测试集相对 gap 走势": "Test relative gap by version", "在运行明细中对比 →": "compare in Run details →",
  "最近完成的运行": "Recently finished runs", "进行中的批次": "Active batches", "该版本的批次": "Batches of this version",
  "全部版本 →": "all versions →", "运行明细 →": "run details →", "实验版本": "Versions", "运行明细": "Run details",
  "全部批次": "All batches", "三场景平均": "3-scenario mean", "相对基线": "vs baseline", "改动": "change", "任务": "jobs",
  "结构感知网络": "structure-aware network", "AR gap · 相对": "AR gap · relative", "相对 gap": "relative gap", "EXIT 率": "EXIT rate",
  "网络 / 设备": "network / device", "网络 · 设备": "network · device", "利用率": "utilization", "显存": "GPU memory",
  "步/秒": "steps/s", "变体": "variant", "网络": "network", "设备": "device",
};
const EXACT = { "有": "yes" };
// 数字 + 量词：只在紧跟数字时替换，避免拆坏其他词
const RULES = [[/第\s*(\d+)\s*\/\s*(\d+)\s*页/g, "page $1 / $2"], [/第\s*(\d+)\s*页/g, "page $1"], [/(\d+)\s*小时/g, "$1 h "], [/(\d+)\s*天/g, "$1 d "], [/(\d+)\s*分钟?/g, "$1 min"],
  [/(\d+)\s*秒/g, "$1 s"], [/(\d+)\s*次/g, "$1 runs"], [/(\d+)\s*个/g, "$1"], [/(\d+)\s*核/g, "$1 cores"], [/(\d+)\s*页/g, "$1 pages"]];
const PUNCT = [["（", " ("], ["）", ")"], ["：", ": "], ["，", ", "], ["、", ", "], ["。", ". "], ["…", "..."], ["；", "; "]];
const KEYS = Object.keys(D).sort((a, b) => b.length - a.length);
const HAN = /[一-鿿（）：，、。；]/;

export function tr(s) {
  if (!HAN.test(s)) return s;
  const t = s.trim();
  if (EXACT[t]) return s.replace(t, EXACT[t]);
  let out = s;
  for (const k of KEYS) if (out.includes(k)) out = out.split(k).join(D[k]);
  for (const [re, b] of RULES) out = out.replace(re, b);
  for (const [a, b] of PUNCT) out = out.split(a).join(b);
  return out.replace(/ {2,}/g, " ");
}

const KEY = "lang";
export let lang = (() => { try { return localStorage.getItem(KEY) || "zh"; } catch { return "zh"; } })();
export const langRef = ref(lang);       // reactive copy for components that load language-specific content
const orig = new WeakMap();      // text node -> original Chinese
const done = new WeakMap();      // text node -> what we wrote
const ATTRS = ["title", "placeholder", "aria-label"];
const skip = (n) => n.parentElement?.closest?.("article.paper, pre, code, textarea, input, [data-no-i18n]");

function doText(n) {
  if (skip(n)) return;
  if (done.get(n) !== n.data) orig.set(n, n.data);     // Vue changed it: new original
  const src = orig.get(n) ?? n.data;
  const v = lang === "en" ? tr(src) : src;
  if (v !== n.data) n.data = v;
  done.set(n, n.data);
}
function doEl(el) {
  for (const a of ATTRS) {
    if (!el.hasAttribute?.(a)) continue;
    const o = `data-zh-${a}`, cur = el.getAttribute(a);
    if (el.getAttribute(`data-en-${a}`) !== cur) el.setAttribute(o, cur);
    const src = el.getAttribute(o);
    const v = lang === "en" ? tr(src) : src;
    if (v !== cur) el.setAttribute(a, v);
    el.setAttribute(`data-en-${a}`, v);
  }
}
function walk(root) {
  if (root.nodeType === 3) return doText(root);
  if (root.nodeType !== 1) return;
  doEl(root);
  const it = document.createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
  for (let n = it.nextNode(); n; n = it.nextNode()) n.nodeType === 3 ? doText(n) : doEl(n);
}

let busy = false;
const obs = new MutationObserver((ms) => {
  if (busy) return;
  busy = true;
  for (const m of ms) {
    if (m.type === "characterData") doText(m.target);
    else if (m.type === "attributes") doEl(m.target);
    else m.addedNodes.forEach(walk);
  }
  obs.takeRecords();
  busy = false;
});

export function setLang(l) {
  lang = l;
  langRef.value = l;
  try { localStorage.setItem(KEY, l); } catch {}
  document.documentElement.lang = l === "en" ? "en" : "zh-CN";
  busy = true; walk(document.body); obs.takeRecords(); busy = false;
}

export function initI18n() {
  obs.observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ATTRS });
  setLang(lang);
}
