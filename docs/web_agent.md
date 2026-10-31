# Web-agent role

The web-agent role (`roles/web_agent.yaml`) is a CRM agent working in a live
SuiteCRM instance from [ST-WebAgentBench](https://github.com/segev-shlomov/ST-WebAgentBench)
through BrowserGym. Its four principles are user consent, strict execution,
hierarchy adherence, and robustness and security. Each episode binds a scenario
to one of ten benchmark workflows, and the agent sees that workflow's
organisational policies in every episode. The planner, simulator, monitor and
judge are the same as for the conversational roles; only the rollout differs:
a turn is one user message plus every browser action up to the next one.

## Setup

You need Docker and uv. The browser runs in its own Python environment, so
BrowserGym's dependencies stay out of RoleFuzzer's.

1. **Benchmark checkout and browser environment.**

   ```bash
   git clone https://github.com/segev-shlomov/ST-WebAgentBench.git external/ST-WebAgentBench
   git -C external/ST-WebAgentBench checkout 67f56dd
   uv venv external/.venv-stweb --python 3.12
   uv pip install --python external/.venv-stweb -e external/ST-WebAgentBench/browsergym/stwebagentbench
   uv pip install --python external/.venv-stweb playwright==1.52.0
   external/.venv-stweb/bin/python -m playwright install chromium
   ```

   To keep these elsewhere, set `ROLEFUZZER_EXTERNAL_DIR` to a directory that
   holds `ST-WebAgentBench/` and `.venv-stweb/`.

2. **SuiteCRM.** The benchmark ships a Docker Compose file and demo data in
   `external/ST-WebAgentBench/suitecrm_setup/`. Its compose file names Bitnami
   images on `public.ecr.aws/bitnami/`, which that registry does not serve; the
   same images are archived as `bitnamilegacy/`. Make a local copy that uses
   them, drops the `linux/amd64` pin (the archived images also publish arm64)
   and takes its port from `SUITECRM_PORT`:

   ```bash
   sed -e 's#public.ecr.aws/bitnami/#bitnamilegacy/#' \
       -e '/platform: linux\/amd64/d' \
       -e 's#"8080:8080"#"${SUITECRM_PORT:-8080}:8080"#' \
       external/ST-WebAgentBench/suitecrm_setup/docker-compose.yaml \
       > external/suitecrm-compose.yaml
   ```

   Start the stack and wait for SuiteCRM to finish its first start-up. The
   web UI answers early with a redirect to its installer, so wait for the log
   line instead:

   ```bash
   docker compose -p rolefuzzer-crm -f external/suitecrm-compose.yaml up -d
   until docker logs rolefuzzer-crm-suitecrm-1 2>&1 | grep -q "SuiteCRM setup finished"; do sleep 5; done
   ```

   Then load the benchmark's demo data and save the result as the snapshot
   every episode restores:

   ```bash
   docker exec -i rolefuzzer-crm-mariadb-1 mysql -u bn_suitecrm -pbitnami123 bitnami_suitecrm \
       < external/ST-WebAgentBench/suitecrm_setup/init-db/demo_data.sql
   uv run python -m rolefuzzer.web_agent.deploy.db_reset --snapshot
   uv run python -m rolefuzzer.web_agent.deploy.db_reset --verify
   ```

   The snapshot is written to `external/suitecrm_snapshot.sql`. Log in with
   the benchmark's credentials (`user` / `bitnami`). The snapshot is restored
   and checked before every episode, because many workflows delete or export
   records. The demo-data load prints only a deprecation warning; `--verify`
   is what confirms it worked, by checking that the records the workflows act
   on are present.

## Run

```bash
uv run rolefuzzer run --role web_agent --responder anthropic/claude-haiku-4-5-20251001 \
    --unparsed-text-as-message --out runs/web_agent_haiku
```

The search modes, budget and model options are the same as for the other
roles, except `--max-concurrent`: episodes run one per SuiteCRM worker. The
judge uses the reasoning effort set in `roles/web_agent.yaml`
(`judge_reasoning_effort: medium`). `--num-seeds` must be a multiple of four,
so every principle gets the same number of seeds; by default all seeds are
used.

Web-agent options:

| Option | Default | Meaning |
|---|---|---|
| `--steps-per-turn` | 3 | Browser actions before the user speaks again, unless the agent messages the user first |
| `--max-steps` | 20 | Browser actions per episode (the benchmark's cap) |
| `--worker-stack URL\|CONTAINER` | none | Extra isolated SuiteCRM workers for parallel episodes |
| `--unparsed-text-as-message` | off | Deliver a reply that has text but no browser action to the user as a message. Useful for models that often answer in plain text, such as Claude Haiku 4.5. |
| `--suitecrm-url`, `--db-container` | `http://localhost:8080`, `rolefuzzer-crm-mariadb-1` | The single SuiteCRM worker, when `--worker-stack` is not given |
| `--task-ids` | all | Comma-separated subset of benchmark tasks |
| `--max-invalid-retries` | 2 | Retries for an episode that ends invalid (for example a judge failure) |

Parallel episodes need one SuiteCRM and database per worker. Start another
stack on its own port and project, then pass each worker:

```bash
SUITECRM_PORT=8081 docker compose -p rolefuzzer-crm-w2 -f external/suitecrm-compose.yaml up -d
uv run rolefuzzer run --role web_agent ... \
    --worker-stack "http://localhost:8080|rolefuzzer-crm-mariadb-1" \
    --worker-stack "http://localhost:8081|rolefuzzer-crm-w2-mariadb-1"
```

Never point two workers at one database.

## Output

The run directory holds `run.json`, `prepared_<n>.json`, `round_<n>.json` and
`campaign.json` as for the other roles, plus `episodes/`: one file per episode,
`r<round>_<index>.json`, with the browser actions, the judge's per-turn
judgments and the benchmark evaluators' report. An invalid attempt is kept in
`episodes/invalid_attempts/` and the episode is retried. Finished episode
files are reused on `--resume`, so an interrupted round does not repeat them.
The evaluators' report is recorded for analysis only: the simulator, monitor,
judge and planner never see it.
