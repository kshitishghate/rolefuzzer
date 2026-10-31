# Role specifications

A role is a YAML file in `roles/`. The framework reads only the fields below,
so a new role needs a specification and seeds, never a change to the planner,
simulator, monitor or judge. If a role seems to need code, add a generic field
to the schema (`src/rolefuzzer/spec.py`) and render it, rather than branching
on the role's id.

## Fields

```yaml
role:
  id: tutor                      # file name and output label
  name: "Mathematics and Critical Thinking Tutor"
  description: >                 # who the responder is; shown to the planner
    ...
  user_role_description: >       # who the human is; the monitor's role gate enforces it
    A student (or someone helping a student) bringing a real academic task ...
  allowed_interaction_modalities:   # default [in_role_interaction]
    - in_role_interaction
  system_prompt: |               # the responder's system prompt
    You are a tutor. Your professional principles:

    {principles_block}
  principles:
    - id: academic_integrity
      name: "Students Must Do the Work"
      specification: >           # the principle as the responder sees it
        ...
      violation_criteria:        # what the judge applies, one text per level
        clear_breach: >
          ...
        partial_deviation: >
          ...
        full_adherence: >
          ...
      pressure_contract:
        violation_mode: compliance_produces_breach   # or noncompliance_produces_breach
        user_goal_test: >        # optional: what makes a valid user goal
          ...
```

Optional fields:

| Field | Use |
|---|---|
| `scenario_invariants` | Facts every scenario must keep, e.g. the antagonist's safeword. |
| `responder_starts: true` with `responder_context` | The responder opens the conversation (the interviewer). `responder_context` lists the subfields the planner must write for it, the subfield that must also appear in the user's scenario, things to leave out, and examples. |
| `seed_compatibility_examples` | Worked examples for the seed auditor's judgement of whether a WildChat user is in the assigned user role. |
| `environment: web_agent` | The responder acts in the web-agent environment instead of a chat. |
| `judge_reasoning_effort` | The judge's reasoning effort for this role (`minimal` to `high`); by default the environment model's `--reasoning-effort`. |

`allowed_interaction_modalities` takes values from `in_role_interaction`,
`commissioning_from_outside_scene` and `operator_template_request`.

`{principles_block}` in `system_prompt` is replaced by the numbered principles;
`{responder_context}`, if present, by the scenario's responder context.

## Pressure contract

`compliance_produces_breach` means the user's goal, if the responder fully
complies, produces the clear-breach behaviour (asking a tutor for the answer).
`noncompliance_produces_breach` means the breach is the responder refusing a
legitimate user request (ignoring an antagonist player's safeword). The planner
uses the contract to check that each scenario actually pressures its target
principle.

## Seeds

The seed bucket draws realistic users from one of two sources:

- **WildChat pool**, `seeds/wildchat/<role>.jsonl`, retrieved from WildChat-1M
  with the queries in `seeds/queries/<role>.yaml` (a `queries` list and an
  optional similarity `threshold`). Pools are built locally from your own
  access to the dataset; see [seeds.md](seeds.md).

- **Written scenes**, `seeds/<role>/*.yaml`, used when a role has no WildChat
  pool. Each scene has a `scenario:` block with `id`, `role_id`,
  `target_principle_id`, `name`, `description`, `user_scenario` and
  `user_principle`; see `seeds/antagonist/` for 32 examples.

## Checking a new role

```bash
uv run rolefuzzer run --role <role> --responder <model> --out runs/check --dry-run
uv run rolefuzzer run --role <role> --responder <model> --out runs/check \
    --rounds 1 --per-round 4 --turns 3
```

The dry run loads the specification and seeds without calling the responder
(seeds missing from the audit cache are audited with the environment model); the
small run shows whether the planner can write valid scenarios for the role and
whether the monitor accepts the simulated user.
