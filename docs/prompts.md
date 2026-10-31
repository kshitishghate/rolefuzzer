# Prompts

The chat roles read every prompt from the templates in
`src/rolefuzzer/prompts/`. The web agent uses the same planner, simulator,
monitor and judge templates, and adds instructions built in code: the
workflow-policy context and action-evidence rules for the judge
(`web_agent/oracle.py`, `web_agent/judge.py`), the browser-setting context for
the monitor (`web_agent/rollout.py`), and the agent's observation prompt
(`web_agent/env_client.py`).

| File | Component | Purpose |
|---|---|---|
| `planner_cell_generation.txt` | planner, coverage bucket | Writes a user scenario for a target grid cell (mode × strategy × principle), anchored on a seed conversation |
| `planner_mutation.txt` | planner, mutation bucket | Mutates a scenario that produced a violation, within its cell (more intensity, or a swapped situational frame) |
| `planner_no_grid.txt` | planner, `--search no-grid` | Writes a pressure scenario from the role specification alone, with no target cell |
| `user_system.txt` | user simulator | Plays the scenario's character, turn by turn |
| `monitor_role_gate_system.txt`, `monitor_role_gate_turn.txt` | monitor, role gate | Rejects turns in which the simulated user speaks as the assistant or as an operator |
| `consistency_monitor.txt`, `consistency_monitor_turn.txt` | monitor, validate | Checks the turn against the character, the scenario and the conversation arc; proposes a revision or ends the conversation |
| `monitor_classify_system.txt`, `monitor_classify_turn.txt` | monitor, classify and steer | Labels an approved turn with its TUNA mode and strategy and writes a steering note toward the target cell |
| `judge_system.txt`, `judge_turn.txt` | judge | Scores every principle on every responder turn as full adherence, partial deviation or clear breach |
| `seed_audit.txt` | seed auditor | Describes a WildChat source user and checks that they fit the role's assigned user ([seeds.md](seeds.md)) |

Each candidate user turn passes the monitor in order: a rule check that
rejects empty turns and collects advisory pattern hints, the role gate, the
validate step, and, for approved turns, the classify-and-steer step. A
rejected turn is regenerated with feedback built from the validator's flags,
up to the revision limit; if every attempt is rejected, the conversation ends.

Templates are role-general: role-specific content comes from the role's YAML
and is rendered into them. Placeholders such as `{role_name}` or
`{principle_pressure_contract}` are filled by the code; a literal brace in a
template is written `{{` or `}}`.
