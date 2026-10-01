"""shared/version_config.py — single source of truth for the "current" paper
version tag.

Every place that used to hardcode a version string (shared/paths.py's
default VERSION, scripts/run_*.sh's PAPER_VERSION) should import
CURRENT_VERSION from here instead. Bump CURRENT_VERSION once per release;
$PAPER_VERSION still overrides it for ad-hoc/historical runs (e.g. replaying
an old ablation tag) without editing this file.
"""

CURRENT_VERSION = "4.3.1.2"

# results/<RESULTS_SPACE>/ is where every run writes (shared/paths.py). Until
# v4.3.1 this was the checked-out git branch name; since the old stage
# branches were archived as tags and work continues on main, it is a fixed
# name instead, so existing data and every doc path that cites
# results/final_paper_experiments/... stay valid. $RESULTS_SPACE overrides it.
RESULTS_SPACE = "final_paper_experiments"
