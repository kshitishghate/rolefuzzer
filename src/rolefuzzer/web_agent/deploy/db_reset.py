"""Snapshot and restore the SuiteCRM database between web-agent episodes.

Why this exists
---------------
ST-WebAgentBench sets ``require_reset: False`` on all 375 tasks and never
resets between them: it assumes one pass over the suite. Fuzzing runs the
same task many times. Eleven of the twenty target tasks delete or export
records, so without a restore the second episode of task 275 finds no
Michael Scott to delete, the sensitive action never fires, and the
hierarchy policy is reported *dormant* rather than violated. That failure
is silent and produces a clean-looking, wrong result, so the restore is
mandatory rather than an optimisation.

Usage::

    uv run python -m rolefuzzer.web_agent.deploy.db_reset --snapshot
    uv run python -m rolefuzzer.web_agent.deploy.db_reset --restore
    uv run python -m rolefuzzer.web_agent.deploy.db_reset --verify
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path

from rolefuzzer.paths import SUITECRM_SNAPSHOT_PATH

logger = logging.getLogger(__name__)

DB_CONTAINER = "rolefuzzer-crm-mariadb-1"
DB_USER = "bn_suitecrm"
DB_PASSWORD = "bitnami123"
DB_NAME = "bitnami_suitecrm"

SNAPSHOT_PATH = SUITECRM_SNAPSHOT_PATH

# Sentinel rows the twenty target tasks act on. A restore that does not
# put these back has not worked, whatever mysqldump reported.
SENTINELS = {
    "contacts": "Michael Scott",
    "leads": "Bruce Wayne",
    "opportunities": "Data Analytics Implementation",
}


def _mysql(args: list[str], stdin_data: bytes | None = None, container: str | None = None) -> bytes:
    # Always -i and never -t: a pseudo-tty makes mysql pretty-print an
    # ASCII table even under -N, which breaks numeric parsing. --batch
    # forces tab-separated output.
    cmd = [
        "docker", "exec", "-i",
        container or DB_CONTAINER, "mysql", "--batch",
        f"-u{DB_USER}", f"-p{DB_PASSWORD}",
    ] + args
    proc = subprocess.run(cmd, input=stdin_data or b"", capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"mysql failed ({proc.returncode}): {proc.stderr.decode(errors='replace')[:500]}"
        )
    return proc.stdout


def snapshot(path: Path = SNAPSHOT_PATH) -> Path:
    """Dump the current database to ``path``. Call once, after loading the demo data."""
    cmd = [
        "docker", "exec", DB_CONTAINER, "mysqldump",
        f"-u{DB_USER}", f"-p{DB_PASSWORD}",
        "--single-transaction", "--skip-lock-tables",
        "--add-drop-table", "--routines", "--triggers",
        DB_NAME,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"mysqldump failed: {proc.stderr.decode(errors='replace')[:500]}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(proc.stdout)
    logger.info("Snapshot written to %s (%.1f MB)", path, len(proc.stdout) / 1e6)
    return path


def restore(path: Path = SNAPSHOT_PATH, verify: bool = True, container: str | None = None) -> float:
    """Restore the database from ``path``. Returns elapsed seconds.

    Raises if verification fails, because a silently-failed restore is the
    failure mode this module exists to prevent.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No snapshot at {path}. Load the benchmark's demo data, then run "
            "`uv run python -m rolefuzzer.web_agent.deploy.db_reset --snapshot` "
            "once (docs/web_agent.md)."
        )
    start = time.monotonic()
    _mysql([DB_NAME], stdin_data=path.read_bytes(), container=container)
    elapsed = time.monotonic() - start
    if verify:
        missing = verify_sentinels(container=container)
        if missing:
            raise RuntimeError(
                f"Restore completed but sentinel rows are missing: {missing}. "
                "The episode must not start against a dirty database."
            )
    logger.info("Restored in %.1fs", elapsed)
    return elapsed


def verify_sentinels(container: str | None = None) -> list[str]:
    """Return the names of sentinel rows that are absent or soft-deleted."""
    missing: list[str] = []
    for table, name in SENTINELS.items():
        if table in ("contacts", "leads"):
            first, last = name.split(" ", 1)
            where = f"first_name='{first}' AND last_name='{last}'"
        else:
            where = f"name='{name}'"
        out = _mysql([
            "-N", "-e",
            f"USE {DB_NAME}; SELECT COUNT(*) FROM {table} WHERE {where} AND deleted=0;",
        ], container=container)
        count = int(out.decode().strip().splitlines()[-1].strip())
        if count == 0:
            missing.append(f"{table}:{name}")
    return missing


# Tables the ten campaign workflows act on, plus the audit tables that log
# field changes with timestamps (they give the order of operations). Read at
# the start and end of each episode for the rule-based state check; the
# evaluated agent never sees them.
STATE_TABLES = (
    "accounts", "contacts", "leads", "opportunities", "cases", "meetings",
    "meetings_contacts", "meetings_users", "meetings_leads",
    "accounts_contacts", "accounts_opportunities", "notes",
    "contacts_audit", "leads_audit", "opportunities_audit", "cases_audit",
)
# Only these user columns are needed to resolve assignee IDs.
USER_COLUMNS = "id, user_name"


def _unescape(value: str) -> str | None:
    if value == "NULL":
        return None
    return (value.replace("\\n", "\n").replace("\\t", "\t")
            .replace("\\0", "\0").replace("\\\\", "\\"))


_MARK = "@@state_table@@"


def read_state(container: str | None = None) -> dict[str, list[dict]]:
    """Every row of the workflow tables, keyed by table name, in one query."""
    queries = [(table, f"SELECT * FROM {table}") for table in STATE_TABLES]
    queries.append(("users", f"SELECT {USER_COLUMNS} FROM users"))
    # One client call; a marker row precedes each table because the client
    # prints nothing, not even a header, for an empty result.
    sql = "".join(f"SELECT '{_MARK}{table}' AS m; {query};" for table, query in queries)
    lines = _mysql([DB_NAME, "-e", sql], container=container).decode().splitlines()
    state: dict[str, list[dict]] = {table: [] for table, _ in queries}
    table = header = None
    for line in lines:
        if line == "m":
            continue
        if line.startswith(_MARK):
            table, header = line[len(_MARK):], None
        elif table is not None and header is None:
            header = line.split("\t")
        elif table is not None:
            state[table].append(
                dict(zip(header, (_unescape(v) for v in line.split("\t")))))
    return state


def state_diff(before: dict[str, list[dict]], after: dict[str, list[dict]]) -> dict:
    """Rows added, removed, or changed per table, keyed by row ``id``."""
    diff: dict[str, dict] = {}
    for table in sorted(set(before) | set(after)):
        old = {row.get("id"): row for row in before.get(table, [])}
        new = {row.get("id"): row for row in after.get(table, [])}
        added = [new[k] for k in new if k not in old]
        removed = [old[k] for k in old if k not in new]
        changed = []
        for key in new.keys() & old.keys():
            cols = sorted(c for c in new[key] if new[key].get(c) != old[key].get(c))
            if cols:
                changed.append({
                    "id": key,
                    "before": {c: old[key].get(c) for c in cols},
                    "after": {c: new[key].get(c) for c in cols},
                    "row": new[key],
                })
        if added or removed or changed:
            diff[table] = {"added": added, "removed": removed, "changed": changed}
    return diff


def state_lookup(state: dict[str, list[dict]]) -> dict[str, dict[str, str]]:
    """ID-to-name maps for resolving relation fields in the state check."""
    def person(row: dict) -> str:
        return " ".join(p for p in (row.get("first_name"), row.get("last_name")) if p)
    return {
        "users": {r["id"]: r.get("user_name") or "" for r in state.get("users", [])},
        "accounts": {r["id"]: r.get("name") or "" for r in state.get("accounts", [])},
        "contacts": {r["id"]: person(r) for r in state.get("contacts", [])},
        "leads": {r["id"]: person(r) for r in state.get("leads", [])},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="store_true", help="dump current DB state")
    parser.add_argument("--restore", action="store_true", help="restore from snapshot")
    parser.add_argument("--verify", action="store_true", help="check sentinel rows only")
    parser.add_argument("--path", type=Path, default=SNAPSHOT_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.snapshot:
        p = snapshot(args.path)
        print(f"snapshot: {p} ({p.stat().st_size / 1e6:.1f} MB)")
    if args.restore:
        elapsed = restore(args.path)
        print(f"restored in {elapsed:.1f}s")
    if args.verify or not (args.snapshot or args.restore):
        missing = verify_sentinels()
        print("sentinels ok" if not missing else f"MISSING: {missing}")


if __name__ == "__main__":
    main()
