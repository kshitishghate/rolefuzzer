# Seeds

Every campaign starts from a pool of seed users. The tutor, financial advisor
and interviewer roles draw theirs from real conversations in
[WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M); the
antagonist role uses the written scenes in `seeds/antagonist/`; the web-agent
role builds its seeds from the ST-WebAgentBench task file
([web_agent.md](web_agent.md)).

This repository contains no WildChat data. Each user builds the pools locally
from their own copy of the dataset, and `seeds/wildchat/` is git-ignored so
the pools never enter version control.

## Getting access to WildChat-1M

WildChat-1M is a gated dataset on Hugging Face, released by the Allen Institute
for AI under the ODC-BY license. To use it:

1. Sign in to Hugging Face, open the
   [dataset page](https://huggingface.co/datasets/allenai/WildChat-1M), and
   accept its terms of use.
2. Create an access token with read permission
   ([settings/tokens](https://huggingface.co/settings/tokens)) and add it to
   `.env` as `HF_TOKEN=...`.
3. Install the retrieval dependencies:

   ```bash
   uv sync --extra seeds
   ```

Your use of the conversations is governed by the dataset's license and terms.
If you publish results, cite WildChat (Zhao et al., 2024, "WildChat: 1M
ChatGPT Interaction Logs in the Wild", ICLR).

## Building a role's pool

```bash
uv run rolefuzzer seeds build --role tutor
uv run rolefuzzer seeds build --role financial_advisor
uv run rolefuzzer seeds build --role interviewer
```

For each role, the builder streams the English conversations of WildChat-1M,
embeds each conversation's first four user messages with
`jinaai/jina-embeddings-v3`, scores them by their maximum cosine similarity to
the role's retrieval queries (`seeds/queries/<role>.yaml`), and keeps those at
or above the role's threshold (0.55 by default; 0.5 for the interviewer). The
pool is written to `seeds/wildchat/<role>.jsonl`. Each record keeps only the
conversation text and the fields the seed loader reads (hash, model, language,
turn count, toxicity flag, similarity score); WildChat's hashed IP addresses,
locations and request headers are dropped.

Embedding the full dataset is the slow step; a GPU helps. Options:

| Option | Meaning |
|---|---|
| `--threshold` | Similarity cut-off, overriding the role's query file |
| `--max-conversations N` | Stop after N English conversations (for a quick test pool) |
| `--batch-size` | Embedding batch size (default 64) |
| `--force` | Replace an existing pool |

## Auditing and selection

A pool holds many more conversations than a campaign uses. The first campaign
for a role audits candidates with the environment model (`seed_audit.txt`):
the auditor describes each source user, checks that they plausibly fit the
role's assigned user (`user_role_description`), and rates how well the
conversation can be role-played. Accepted seeds are taken in order of
similarity until the campaign has `--num-seeds` (128 by default; `--seed`
fixes the draw). Verdicts are cached in `seeds/wildchat/constraints.jsonl`,
keyed by conversation and role, so later campaigns, including dry runs, reuse
them without further calls. The first run on a new pool makes those audit
calls even with `--dry-run`, so it needs `OPENAI_API_KEY`.

A campaign for a WildChat role stops with a message naming the build command
if the role's pool has not been built.

## A new WildChat role

Write retrieval queries in `seeds/queries/<role>.yaml`, a `queries` list and
an optional `threshold`, then build the pool as above. Queries work best when
they span the natural range of what users bring to the role rather than
targeting pressure: the campaign supplies the pressure. If the auditor needs
role-specific guidance on who counts as the assigned user, put worked examples
in the role specification's `seed_compatibility_examples` field
([roles.md](roles.md)).
