"""Command-line interface.

    rolefuzzer run --role tutor --responder anthropic/claude-haiku-4-5-20251001 --out runs/tutor_haiku
    rolefuzzer run --role tutor --responder qwen/qwen3-32b \\
        --responder-base-url https://openrouter.ai/api/v1 --responder-disable-thinking \\
        --search no-grid --out runs/tutor_qwen_nogrid
    rolefuzzer seeds build --role tutor --force

Defaults: 8 rounds of 30 conversations, 8 turns, GPT-5-mini for every
environment role.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

from rolefuzzer.paths import REPO_ROOT, ROLES_DIR, role_path
from rolefuzzer.planner.config import SEARCH_MODES
from rolefuzzer.runner import run
from rolefuzzer.spec import load_role


def _roles() -> list[str]:
    return sorted(p.stem for p in ROLES_DIR.glob("*.yaml"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rolefuzzer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a fuzzing campaign against one role.")
    run.add_argument("--role", required=True, help=f"Role id under roles/ ({', '.join(_roles())}).")
    run.add_argument("--responder", required=True,
                     help="Target responder as an Inspect model string, e.g. "
                          "anthropic/claude-haiku-4-5-20251001, or a model name served at "
                          "--responder-base-url.")
    run.add_argument("--out", type=Path, required=True, help="Run directory.")
    run.add_argument("--search", choices=SEARCH_MODES, default="full",
                     help="full: RoleFuzzer. no-grid: scenarios from the specification alone. "
                          "random-cell: uniform cell targeting. seed-only: seed bucket only.")
    run.add_argument("--resume", action="store_true", help="Continue the run in --out.")
    run.add_argument("--dry-run", action="store_true",
                     help="Load the role and seeds, print the plan, make no responder calls.")

    models = run.add_argument_group("models")
    models.add_argument("--env-model", default="openai/gpt-5-mini",
                        help="Planner, user simulator, monitor, judge and seed audit.")
    models.add_argument("--reasoning-effort", default="minimal",
                        choices=["minimal", "low", "medium", "high"],
                        help="Reasoning effort for the environment model.")
    models.add_argument("--responder-base-url", default=None,
                        help="OpenAI-compatible endpoint for the responder (vLLM, OpenRouter). "
                             "The key is read from RESPONDER_API_KEY.")
    models.add_argument("--responder-disable-thinking", action="store_true",
                        help="Turn off Qwen3-style thinking for the responder.")
    models.add_argument("--responder-max-tokens", type=int, default=2048)

    budget = run.add_argument_group("campaign")
    budget.add_argument("--rounds", type=int, default=8)
    budget.add_argument("--per-round", type=int, default=30, help="Conversations per round.")
    budget.add_argument("--turns", type=int, default=8, help="User turns per conversation.")
    budget.add_argument("--num-seeds", type=int, default=None,
                        help="Seed pool size (default: 128 for WildChat pools, all scenes otherwise).")
    budget.add_argument("--seed", type=int, default=42, help="Random seed for seed selection and every random choice in the "
                             "campaign.")
    budget.add_argument("--n-min-breadth", type=int, default=3,
                        help="While any reachable grid cell has fewer conversations than this, "
                             "cell choice ignores violations and favours the least-covered "
                             "cells. 0 disables it.")
    budget.add_argument("--max-mutation-depth", type=int, default=2)
    budget.add_argument("--coverage-threshold", type=float, default=1.01,
                        help="Stop early once grid coverage reaches this ratio (>1 never stops).")
    budget.add_argument("--max-concurrent", type=int, default=24,
                        help="Conversations run in parallel within a round.")
    budget.add_argument("--stop-after-round", type=int, default=None,
                        help="Stop after this round; continue later with --resume.")

    web = run.add_argument_group("web-agent options (roles with environment: web_agent)")
    web.add_argument("--suitecrm-url", default=None, help="Default: http://localhost:8080.")
    web.add_argument("--db-container", default=None)
    web.add_argument("--worker-stack", action="append", default=None,
                     metavar="SUITECRM_URL|DB_CONTAINER",
                     help="Isolated SuiteCRM/MariaDB worker; repeat for parallel episodes.")
    web.add_argument("--task-ids", default=None, help="Comma-separated subset of benchmark tasks.")
    web.add_argument("--steps-per-turn", type=int, default=None, help="Default: 3.")
    web.add_argument("--max-steps", type=int, default=None, help="Default: 20 (the benchmark's cap).")
    web.add_argument("--max-invalid-retries", type=int, default=None, help="Default: 2.")
    web.add_argument("--unparsed-text-as-message", action="store_true", default=None,
                     help="Deliver a reply with text but no browser action to the user as a "
                          "message (useful for models that often reply in plain text, e.g. Claude Haiku 4.5).")

    seeds = sub.add_parser("seeds", help="Seed pools.")
    seeds_sub = seeds.add_subparsers(dest="seeds_command", required=True)
    build = seeds_sub.add_parser("build", help="Build a role's WildChat seed pool (needs the seeds extra).")
    build.add_argument("--role", required=True)
    build.add_argument("--threshold", type=float, default=None,
                       help="Similarity cutoff (default: from seeds/queries/<role>.yaml, else 0.55).")
    build.add_argument("--max-conversations", type=int, default=None,
                       help="Stop after this many English conversations (for testing).")
    build.add_argument("--batch-size", type=int, default=64)
    build.add_argument("--force", action="store_true", help="Replace an existing pool.")

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    args = build_parser().parse_args(argv)
    load_dotenv(REPO_ROOT / ".env", override=False)

    if args.command == "seeds":
        from rolefuzzer.seeds.build import build_seed_pool

        build_seed_pool(args.role, threshold=args.threshold, batch_size=args.batch_size,
                        limit=args.max_conversations, force=args.force)
        return

    path = role_path(args.role)
    if not path.exists():
        raise SystemExit(f"Unknown role {args.role!r}; available: {', '.join(_roles())}")
    role_spec = load_role(path)
    asyncio.run(run(args, role_spec))


if __name__ == "__main__":
    main()
