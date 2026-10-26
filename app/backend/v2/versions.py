"""Test relative gap by version for the v2 panel (GET /api/v2/version_trend). The numbers are parsed from each
version's primary report (paper_contents/v<version>/report.md, else seeds_report.md / pilot_report.md, + .en.md): the summary table's test relative gap per
scenario of the version's new variant (the last non-reference model) and of its baseline (the first model). Which
versions are shown, the anchors and the phase notes come from version_trend.json. Independent of the old backend."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .data import PAPER, read_json

CONFIG = Path(__file__).resolve().parent / "version_trend.json"
SCENS = ["LT", "EQ", "GT"]
GAP_HEAD = ("测试集相对 gap", "Test relative gap", "相对 gap", "Relative gap")
REPORT_NAMES = ("report", "seeds_report", "pilot_report")    # a version's primary report, in this order of preference


def _pct(cell: str):
    m = re.match(r"\s*(-?[\d.]+)%", cell)
    return float(m.group(1)) if m else None


def parse_report(text: str) -> dict:
    """Title, the "single change" sentence, and the test relative gap per (scenario, model) from the first tables
    that have a model and a gap column (a summary table, or one table per "## LT/EQ/GT" section; per-seed tables
    are skipped). New variant = the last model that is not a reference; baseline = the first model."""
    lines = text.splitlines()
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), "")
    change = ""
    for l in lines:
        m_ = re.match(r"^- (唯一改动[^：:]*|两个单变量实验[^：:]*|Single change[^:]*)[：:]\s*(.*)", l)
        if m_:
            change = re.split(r"(?<=[。.])\s", m_.group(2), 1)[0]
            break
    gaps, models, section, m = {}, [], None, 0
    while m < len(lines):
        l = lines[m]
        m_ = re.match(r"^##\s+(LT|EQ|GT)\b", l)
        if m_:
            section = m_.group(1)
        if l.startswith("|") and m + 1 < len(lines) and re.match(r"^\|[-| :]+\|?$", lines[m + 1].strip()):
            head = [c.strip() for c in l.strip().strip("|").split("|")]
            rows, n = [], m + 2
            while n < len(lines) and lines[n].startswith("|"):
                rows.append([c.strip() for c in lines[n].strip().strip("|").split("|")])
                n += 1
            gcol = next((head.index(h) for h in GAP_HEAD if h in head), None)
            mcol = next((head.index(h) for h in ("模型", "Model") if h in head), None)
            scol = next((head.index(h) for h in ("场景", "Scenario") if h in head), None)
            if gcol is not None and mcol is not None and not any(h in head for h in ("种子", "Seed")):
                for r in rows:
                    if len(r) <= max(gcol, mcol):
                        continue
                    sc, v = (r[scol] if scol is not None else section), _pct(r[gcol])
                    if sc in SCENS and v is not None and (sc, r[mcol]) not in gaps:
                        gaps[(sc, r[mcol])] = v
                        if r[mcol] not in models:
                            models.append(r[mcol])
            m = n
            continue
        m += 1
    out = {"title": title, "change": change, "new_model": None, "base_model": None, "new": None, "base": None}
    if models:
        refs = [m_ for m_ in models if re.search(r"参照|reference", m_, re.I)]
        variants = [m_ for m_ in models[1:] if m_ not in refs]
        new, base = (variants[-1] if variants else models[-1]), models[0]
        g = {s: gaps.get((s, new)) for s in SCENS}
        b = {s: gaps.get((s, base)) for s in SCENS}
        out.update(new_model=new, base_model=base,
                   new=g if all(v is not None for v in g.values()) else None,
                   base=b if base != new and all(v is not None for v in b.values()) else None)
    return out


def trend() -> dict:
    cfg = read_json(CONFIG) or {}
    points, skipped = [], []
    for v in cfg.get("versions", []):
        rep = next((PAPER / v / f"{n}.md" for n in REPORT_NAMES if (PAPER / v / f"{n}.md").is_file()), None)
        if rep is None:
            skipped.append({"version": v, "reason": "no report"})
            continue
        zh = parse_report(rep.read_text(encoding="utf-8"))
        en_p = rep.with_name(rep.stem + ".en.md")
        en = parse_report(en_p.read_text(encoding="utf-8")) if en_p.is_file() else {}
        if zh["new"] is None:
            skipped.append({"version": v, "reason": "no complete LT / EQ / GT gap in the report"})
            continue
        g = zh["new"]
        points.append({"version": v, "LT": g["LT"], "EQ": g["EQ"], "GT": g["GT"], "avg": round(sum(g.values()) / 3, 2),
                       "model": {"zh": zh["new_model"], "en": en.get("new_model") or zh["new_model"]},
                       "change": {"zh": zh["change"], "en": en.get("change") or ""},
                       "base_avg": round(sum(zh["base"].values()) / 3, 2) if zh["base"] else None, "report": rep.name,
                       "anchor": (cfg.get("anchors") or {}).get(v)})
    # the final configuration's baseline: the baseline mean most versions compare against
    count = {}
    for p in points:
        if p["base_avg"] is not None:
            count[round(p["base_avg"], 1)] = count.get(round(p["base_avg"], 1), 0) + 1
    baseline = max(count, key=count.get) if count else None
    return {"points": points, "skipped": skipped, "baseline": baseline, "target": 5.0,
            "mlp_era_before": cfg.get("mlp_era_before"), "phases": cfg.get("phases", [])}
