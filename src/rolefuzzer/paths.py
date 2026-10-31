"""Repository locations for roles, seeds, the taxonomy and third-party assets.

RoleFuzzer runs from a clone of the repository (``uv sync``), so data paths
resolve relative to the repository root. WildChat seed pools and the web-agent
benchmark are not part of the repository: each user builds or downloads them
(``docs/seeds.md``, ``docs/web_agent.md``), and both locations are git-ignored.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROLES_DIR = REPO_ROOT / "roles"
SEEDS_DIR = REPO_ROOT / "seeds"
TAXONOMY_PATH = REPO_ROOT / "taxonomy" / "tuna.yaml"
WILDCHAT_SEEDS_DIR = SEEDS_DIR / "wildchat"
SEED_CONSTRAINTS_PATH = WILDCHAT_SEEDS_DIR / "constraints.jsonl"
SEED_QUERIES_DIR = SEEDS_DIR / "queries"

# Web-agent role: ST-WebAgentBench checkout and its Python environment.
EXTERNAL_DIR = Path(os.environ.get("ROLEFUZZER_EXTERNAL_DIR", REPO_ROOT / "external"))
STWEB_DIR = EXTERNAL_DIR / "ST-WebAgentBench"
STWEB_TASKS_JSON = STWEB_DIR / "stwebagentbench" / "test.raw.json"
STWEB_PYTHON = EXTERNAL_DIR / ".venv-stweb" / "bin" / "python"
SUITECRM_SNAPSHOT_PATH = EXTERNAL_DIR / "suitecrm_snapshot.sql"


def role_path(role_id: str) -> Path:
    return ROLES_DIR / f"{role_id}.yaml"


def scene_seed_dir(role_id: str) -> Path:
    return SEEDS_DIR / role_id


def wildchat_seed_path(role_id: str) -> Path:
    return WILDCHAT_SEEDS_DIR / f"{role_id}.jsonl"
