# autoppia_operator (Miner Example)

This repo is a minimal FastAPI web-agent service intended to run as a **miner** in the Autoppia web-agents subnet.

## What the validator runs

The validator starts your container with:

```bash
uvicorn main:app --host 0.0.0.0 --port $SANDBOX_AGENT_PORT
```

So the only hard requirements are:
- `main.py` exports `app`
- `GET /health` returns 200
- `POST /act` returns `{ "actions": [...] }`

### Sandbox dependencies / requirements.txt

The subnet validator runs your repo inside a sandbox image with a fixed Python environment.

This repo's `requirements.txt` is intended to be **identical** to the sandbox requirements shipped in the subnet (`autoppia_web_agents_subnet/opensource/sandbox/requirements.txt`).

If you change dependencies here, also update the subnet sandbox image (otherwise your miner may work locally but fail in production).


## Gateway / OpenAI routing

In subnet production, miners **must not** call OpenAI directly. The validator runs a local HTTP proxy (the "sandbox gateway") and injects:

- `OPENAI_BASE_URL=http://sandbox-gateway:9000/openai/v1`

Your agent should send requests to `${OPENAI_BASE_URL}/chat/completions` (or the equivalent endpoint) so the gateway can enforce policy, collect metrics, and use the validator-controlled provider keys.

### Task id propagation (required)

Every LLM request must include the header:

- `IWA-Task-ID: <task_id>`

Where `<task_id>` is the `task_id` value received in `POST /act`.

This is how the gateway correlates all model calls to a single evaluation episode.

### Where it is implemented here

- `llm_gateway.py`: minimal OpenAI-compatible client that:
  - reads `OPENAI_BASE_URL`
  - injects `IWA-Task-ID`
  - only uses `OPENAI_API_KEY` when you are **not** using the sandbox gateway
- `agent.py`: calls `openai_chat_completions(task_id=..., ...)` so the header is always present.


## Agent flow

`POST /act` (`agent.py`) receives:
- `task_id`: used for `IWA-Task-ID` header
- `prompt`: natural-language task
- `url`: current page URL
- `snapshot_html`: current page HTML
- `step_index`: current step number
- `history`: last actions (best-effort)

The API layer is intentionally thin: it forwards to `AutoppiaOperator` (class implementing `IWebAgent.act()` from IWA).

The operator then:
1. Extracts interactive candidates from HTML (buttons/links/inputs, etc.).
2. Ranks candidates against the task.
3. Builds a compact deterministic **Page IR** (forms, headings, links, cards, CTAs) plus deltas from previous step.
4. Calls the LLM to choose the next single action (`click`/`type`/`select`/`scroll_*`/`done`) using Page IR + step context.
5. Returns a single IWA action (e.g. `ClickAction`, `TypeAction`, ...).

Credential placeholders like `<username>` / `<password>` are handled by IWA (the evaluator replaces placeholders in actions before execution).

## Built-in inspection tools

The planner can request tools before choosing an action:
- `search_text`
- `visible_text`
- `css_select`
- `xpath_select`
- `extract_forms`
- `list_links`
- `list_candidates`
- `list_cards`
- `find_card` (query-focused card retrieval)

## Local eval

This repo includes a local evaluator:

```bash
python eval.py --model gpt-5.2 --num-tasks 5 --distinct-use-cases
```

Task generation helper (writes the cache consumed by `eval.py`):

```bash
python scripts/generate_tasks.py --project-id autocinema --prompts-per-use-case 1
```

Outputs are written to `data/` (gitignored).


## Model comparison

To compare multiple models/providers on the same task set, use:

```bash
python scripts/compare_eval.py --runs openai:gpt-5.2 openai:gpt-4o-mini --num-tasks 5 --distinct-use-cases
```

Anthropic example (requires `ANTHROPIC_API_KEY` in your env):

```bash
python scripts/compare_eval.py --runs anthropic:claude-sonnet-4 --num-tasks 5 --distinct-use-cases
```

Outputs:
- `data/compare/<provider>__<model>.json`
- `data/compare/compare_summary.json`

## Repo self-check

```bash
python check.py
```

This validates entrypoints, endpoint shapes, and scans for obvious secrets.

⚠️ deploy_check may fail locally if handshake env vars are missing:
- set `MINER_AGENT_NAME` (or `AGENT_NAME`)
- set `MINER_GITHUB_URL` (or `AGENT_GITHUB_URL` / `GITHUB_URL`)
- optional: `AGENT_IMAGE`, `AGENT_VERSION`

If those are absent, subnet metadata can be considered missing during handshake.
