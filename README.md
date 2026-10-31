# RoleFuzzer: User-Behaviour Fuzzing for Role-Specification Adherence in Large Language Models

Paper: [arXiv:2610.38438](https://arxiv.org/abs/2610.38438)

RoleFuzzer tests whether a language model assigned a role (a tutor, a financial
advisor, a job interviewer, a game antagonist, a web agent) keeps to the
principles in its role specification when realistic users push against them.
It treats the space of user behaviour as a coverage grid, from the Taxonomy of
User Needs and Actions (TUNA), and runs a coverage-guided fuzzing campaign over it:

- a **planner** writes user scenarios for under-covered cells of the grid,
  starting from real seed conversations and mutating scenarios that produced
  violations;
- a **user simulator** plays each scenario as a multi-turn conversation;
- a **consistency monitor** keeps the simulated user in character, classifies
  each user turn into the grid and steers it toward the target cell;
- the **target responder** (the model under test) replies in its role;
- a **judge** scores every principle on every turn as full adherence, partial
  deviation or clear breach.

Judge scores reach only the planner, never the simulator or the monitor.

## Install

RoleFuzzer uses [uv](https://docs.astral.sh/uv/) and Python 3.11 or later.

```bash
git clone https://github.com/kshitishghate/rolefuzzer.git
cd rolefuzzer
uv sync
cp .env.example .env   # then fill in the keys you need
```

The environment roles (planner, simulator, monitor, judge) default to
`openai/gpt-5-mini`, so `OPENAI_API_KEY` is required. Add the key for your
responder's provider: `ANTHROPIC_API_KEY` for Claude models, or
`RESPONDER_API_KEY` for a responder served from an OpenAI-compatible endpoint
such as OpenRouter or a local vLLM server.

The tutor, financial advisor and interviewer roles draw their seed users from
WildChat-1M, which is not included here. Build their seed pools once from your
own access to the dataset ([docs/seeds.md](docs/seeds.md)):

```bash
uv sync --extra seeds
uv run rolefuzzer seeds build --role tutor
```

## Run a campaign

```bash
# A small smoke run: one round of four conversations, three turns each.
uv run rolefuzzer run --role tutor --responder anthropic/claude-haiku-4-5-20251001 \
    --rounds 1 --per-round 4 --turns 3 --out runs/smoke

# A full campaign with the default settings.
uv run rolefuzzer run --role tutor --responder anthropic/claude-haiku-4-5-20251001 \
    --out runs/tutor_haiku
```

Defaults: 8 rounds of 30 conversations, 8 user turns per conversation,
GPT-5-mini with minimal reasoning effort for every environment role (a role
can set its judge's effort with `judge_reasoning_effort`), and a bucket
schedule that anneals from (seed, coverage, mutation) = (0.15, 0.40, 0.45) to
(0.15, 0.25, 0.60). `--dry-run` loads the role and its seeds and prints the
plan without calling the responder.

Examples of responder options:

| Responder | Options |
|---|---|
| Claude Haiku 4.5 | `--responder anthropic/claude-haiku-4-5-20251001` |
| Qwen3-32B (OpenRouter) | `--responder qwen/qwen3-32b --responder-base-url https://openrouter.ai/api/v1 --responder-disable-thinking` |
| Gemma 4 31B (OpenRouter) | `--responder google/gemma-4-31b-it --responder-base-url https://openrouter.ai/api/v1` |

With `--responder-base-url`, set `RESPONDER_API_KEY` to that endpoint's key.
Any [Inspect AI](https://inspect.aisi.org.uk/) model string works as
`--responder`, and `--env-model` changes the environment model.

### Search modes and baselines

| `--search` | What runs |
|---|---|
| `full` (default) | RoleFuzzer: seed, coverage-directed and mutation buckets |
| `no-grid` | The planner writes pressure scenarios from the role specification alone, with no target cells and no coverage feedback |
| `random-cell` | The full method with target cells chosen uniformly at random |
| `seed-only` | Only the seed bucket: realistic users without cell targeting, steering or mutation |

The seed-only baseline is also contained in every full campaign: its seed
bucket runs without targeting or steering.

### Output

A run directory holds:

- `run.json`: the settings (including the seed ids), the creation time and the
  git commit;
- `prepared_<n>.json`: round *n*'s scenarios, written before the round runs;
- `round_<n>.json`: every conversation of round *n*, with its scenario, the
  monitor's per-turn checks and the judge's per-turn, per-principle judgments
  (`sample_payloads[*].metadata.turn_judgments`);
- `campaign.json`: grid coverage, per-cell violation statistics and monitor
  statistics for the whole campaign;
- `logs/`: Inspect logs for the chat roles
  (`uv run inspect view --log-dir runs/<name>/logs`);
- `episodes/`: one file per web-agent episode ([docs/web_agent.md](docs/web_agent.md)).

`--resume` continues an interrupted run under the same settings. It replays
the checkpoints and restores the full campaign state (grid, planner tables,
seed rotation and random state), so a resumed run matches an uninterrupted
one. `--stop-after-round N` stops early so a later `--resume` can finish it.
`--seed` (default 42) fixes seed selection and the campaign's random choices,
so two runs with the same settings make the same choices whenever the models
return the same outputs; live model calls are not deterministic, so in
practice runs differ.

## Roles

| Role | Spec | Seeds |
|---|---|---|
| Tutor | `roles/tutor.yaml` | 128 WildChat-1M conversations, from a locally built pool |
| Financial advisor | `roles/financial_advisor.yaml` | 128 WildChat-1M conversations, from a locally built pool |
| AI job interviewer | `roles/interviewer.yaml` | 128 WildChat-1M conversations, from a locally built pool |
| Antagonist game character | `roles/antagonist.yaml` | 32 written scenes, `seeds/antagonist/` |
| Web agent (SuiteCRM) | `roles/web_agent.yaml` | 32 workflow seeds built from the ST-WebAgentBench task file |

A role specification lists principles, each with a specification, graded
violation criteria (full adherence, partial deviation, clear breach) and a
pressure contract. The framework code is role-general: a new role needs only a
YAML file in `roles/` and seeds. See [docs/roles.md](docs/roles.md).

The web agent acts in a live SuiteCRM instance through BrowserGym and needs
Docker and a local ST-WebAgentBench checkout, which is not included here. See
[docs/web_agent.md](docs/web_agent.md).

## Repository layout

```
roles/                 role specifications
seeds/                 retrieval queries, antagonist scenes (WildChat pools are built here)
taxonomy/tuna.yaml     the TUNA user-behaviour taxonomy
src/rolefuzzer/
  cli.py               the rolefuzzer command
  runner.py            run directory, checkpoints, resume and the chat backend
  spec.py              role, principle and scenario schema
  models.py            environment and responder model construction
  planner/             coverage grid, scenario generation, mutation, campaign loop
  conversation.py      simulator, monitor, responder and judge in one conversation
  monitor.py           role gate, validation, classification and steering
  turn_loop.py         the judge call, its parser and regeneration feedback
  scorer.py, metrics.py  per-conversation scores and Inspect metrics
  seeds/               seed loading, auditing and pool building
  web_agent/           the web-agent backend (backend.py), browser episodes,
                       BrowserGym environment server, workflow judge, seeds,
                       policy selection and SuiteCRM database reset
  prompts/             prompt templates
docs/                  guides to roles, seeds, the web agent and the prompts
tests/                 unit tests and a scripted end-to-end campaign (no API calls)
```

Run the tests with `uv sync --extra dev && uv run pytest`.

## Prompts

The prompt templates are in `src/rolefuzzer/prompts/`;
[docs/prompts.md](docs/prompts.md) lists each one and the component that
uses it, and names the web-agent instructions that are built in code.

## Citation

```bibtex
@misc{ghate2026rolefuzzer,
  title         = {RoleFuzzer: User-Behaviour Fuzzing for Role-Specification Adherence in Large Language Models},
  author        = {Kshitish Ghate and Michael Saxon and Tadayoshi Kohno and Aylin Caliskan},
  year          = {2026},
  eprint        = {2610.38438},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2610.38438}
}
```

## License

MIT.
