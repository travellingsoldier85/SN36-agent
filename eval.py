#!/usr/bin/env python3
"""
Simple evaluation script for the autoppia_operator /act endpoint.

Uses AsyncStatefulEvaluator from autoppia_iwa and calls the local agent
HTTP API directly (no autoppia_rl dependencies).
"""

import asyncio
import base64
import hashlib
import json
import os
import random
import subprocess
import socket
import sys
import time
from typing import Any
from copy import deepcopy
from pathlib import Path

# ── Ensure the operator repo is on sys.path ─────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OPERATOR_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(OPERATOR_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# ── Load .env from autoppia_operator ────────────────────────────
from dotenv import load_dotenv

operator_env = SCRIPT_DIR / ".env"
if operator_env.exists():
    load_dotenv(operator_env, override=True)

# ── Imports ──────────────────────────────────────────────────────
from loguru import logger

from autoppia_iwa.src.data_generation.tasks.classes import Task
from autoppia_iwa.src.evaluation.stateful_evaluator import AsyncStatefulEvaluator
from autoppia_iwa.src.execution.actions.base import BaseAction
from pricing import estimate_cost_usd
import autoppia_iwa.src.execution.actions.actions  # noqa: F401

# Default task cache path
TASK_CACHE = OPERATOR_ROOT / "autoppia_rl" / "data" / "task_cache" / "autoppia_cinema_tasks.json"

random.seed(time.time())


# ── Task loading ─────────────────────────────────────────────────

def load_tasks(
    cache_path: Path = TASK_CACHE,
    use_case: str | None = None,
    web_project_id: str | None = None,
    task_id: str | None = None,
    limit: int = 20,
) -> list[Task]:
    """Load tasks from the JSON cache, optionally filtered."""
    with open(cache_path) as f:
        data = json.load(f)

    raw_tasks = data["tasks"] if isinstance(data, dict) and "tasks" in data else data

    tasks: list[Task] = []
    for td in raw_tasks:
        # Optional task id filter (exact match)
        if task_id:
            if str(td.get("id", "")) != str(task_id):
                continue

        # Optional use-case filter
        if use_case:
            uc = td.get("use_case", {})
            uc_name = uc.get("name", "") if isinstance(uc, dict) else ""
            if use_case.upper() not in str(uc_name).upper():
                continue

        # Optional web project filter
        if web_project_id is not None:
            if str(td.get("web_project_id", "")) != str(web_project_id):
                continue

        try:
            task = Task(**td)
            tasks.append(task)
        except Exception as e:
            logger.debug(f"Skipping task {td.get('id', '?')}: {e}")

        if len(tasks) >= limit:
            break

    return tasks


def _serialize_screenshot(raw: Any | None) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw or None
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(raw)).decode("ascii")
    return None


def inject_seed(task: Task, seed: int | None = None) -> tuple[Task, int]:
    """Inject a seed into the task URL for variation (or use a provided seed)."""
    t = deepcopy(task)
    seed_i = int(seed) if seed is not None else random.randint(1, 100_000)
    base_url = t.url.split("?")[0] if "?" in t.url else t.url
    t.url = f"{base_url}?seed={seed_i}"
    return t, seed_i


# ── Main evaluation loop ────────────────────────────────────────

async def run_evaluation(
    provider: str = "openai",
    model: str = "gpt-5-mini",
    num_tasks: int = 20,
    max_steps: int = 15,
    use_case: str | None = None,
    web_project_id: str | None = None,
    task_id: str | None = None,
    seed: int | None = None,
    repeat: int = 1,
    temperature: float = 0.2,
    distinct_use_cases: bool = False,
    out_path: str | None = None,
    task_cache: str | None = None,
    strict_model: bool = True,
):
    # Re-load .env here as a guard: some imported modules may mutate env vars.
    try:
        from dotenv import load_dotenv
        operator_env = Path(__file__).resolve().parent / ".env"
        if operator_env.exists():
            load_dotenv(operator_env, override=True)
    except Exception:
        pass

    provider_s = str(provider or os.getenv('LLM_PROVIDER') or 'openai').strip().lower()

    cache_path = Path(task_cache).resolve() if task_cache else TASK_CACHE
    if provider_s == 'anthropic':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        api_key_fpr = hashlib.sha256((api_key or "").encode("utf-8")).hexdigest()[:12] if api_key else "missing"
        logger.info(f"Eval env: ANTHROPIC_API_KEY={'set' if api_key else 'missing'} fpr={api_key_fpr}")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not set. Check .env file.")
            sys.exit(1)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        api_key_fpr = hashlib.sha256((api_key or "").encode("utf-8")).hexdigest()[:12] if api_key else "missing"
        logger.info(f"Eval env: OPENAI_API_KEY={'set' if api_key else 'missing'} fpr={api_key_fpr}")
        # For sandbox-gateway routing, OPENAI_API_KEY may be intentionally absent.
        base_url = (os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1').rstrip('/')
        if not api_key and not base_url.startswith('http://sandbox-gateway') and not base_url.startswith('http://localhost') and not base_url.startswith('http://127.0.0.1'):
            logger.error("OPENAI_API_KEY not set. Check .env file.")
            sys.exit(1)
    logger.info("=" * 60)
    logger.info("  Autoppia Operator – LLM Agent Evaluation")
    logger.info(f"  Provider:   {provider_s}")
    logger.info(f"  Model:      {model}")
    logger.info(f"  Tasks:      {num_tasks}")
    logger.info(f"  Task cache: {cache_path}")
    logger.info(f"  Max steps:  {max_steps}")
    logger.info(f"  Use case:   {use_case or 'all'}")
    logger.info(f"  Web proj:   {web_project_id or 'all'}")
    logger.info(f"  Task id:    {task_id or 'auto'}")
    logger.info(f"  Seed:       {seed if seed is not None else 'random'}")
    logger.info(f"  Repeat:     {int(repeat)}")
    logger.info(f"  Strict mdl: {bool(strict_model)}")
    logger.info("=" * 60)

    # Load tasks. If we want distinct use cases, load more upfront then filter down.
    load_limit = num_tasks
    if distinct_use_cases:
        load_limit = max(500, num_tasks * 20)
    tasks = load_tasks(cache_path=cache_path, use_case=use_case, web_project_id=web_project_id, task_id=task_id, limit=load_limit)
    logger.info(f"Loaded {len(tasks)} tasks")

    if not tasks:
        logger.error("No tasks found. Check task cache path and use_case filter.")
        return

    if distinct_use_cases:
        picked: list[Task] = []
        seen: set[str] = set()
        rest: list[Task] = []
        for t in tasks:
            uc_name = ""
            uc = getattr(t, "use_case", None)
            if isinstance(uc, dict):
                uc_name = str(uc.get("name") or "")
            elif uc is not None and hasattr(uc, "name"):
                uc_name = str(getattr(uc, "name") or "")
            if not uc_name:
                rest.append(t)
                continue
            if uc_name in seen:
                rest.append(t)
                continue
            seen.add(uc_name)
            picked.append(t)
            if len(picked) >= num_tasks:
                break
        if len(picked) < num_tasks:
            for t in rest:
                picked.append(t)
                if len(picked) >= num_tasks:
                    break
        tasks = picked[:num_tasks]
        logger.info(f"Selected {len(tasks)} tasks with distinct use cases")
    else:
        tasks = tasks[:num_tasks]

    # Agent endpoint config
    agent_base_url = os.getenv("AGENT_BASE_URL", "").strip().rstrip("/")
    start_server = os.getenv("START_AGENT_SERVER", "1") in {"1", "true", "yes"}
    web_agent_id = os.getenv("WEB_AGENT_ID", "1").strip() or "1"
    def _port_available(port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False

    def _pick_port(preferred: int) -> int:
        if _port_available(preferred):
            return preferred
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    # If we're starting the server locally, choose a free port and point the client to it.
    if start_server:
        preferred_port = int(os.getenv("AGENT_PORT", "5000"))
        port = _pick_port(preferred_port)
        if not agent_base_url:
            agent_base_url = f"http://127.0.0.1:{port}"

        # Append logs per run so bind errors / tracebacks are preserved.
        out_dir = SCRIPT_DIR / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "agent_server.log"
        log_f = open(log_path, "a", encoding="utf-8")
        log_f.write(f"\n=== uvicorn main:app port={port} ===\n")
        log_f.flush()

        server_env = os.environ.copy()
        server_env["OPENAI_MODEL"] = str(model)
        server_env["LLM_PROVIDER"] = str(provider_s)
        server_env["OPENAI_TEMPERATURE"] = str(temperature)
        server_env["AGENT_RETURN_METRICS"] = "1"
        k = server_env.get("OPENAI_API_KEY") or ""
        k_fpr = hashlib.sha256(k.encode("utf-8")).hexdigest()[:12] if k else "missing"
        logger.info(f"Agent server env: OPENAI_MODEL={server_env.get('OPENAI_MODEL')} LLM_PROVIDER={server_env.get('LLM_PROVIDER')} START_AGENT_SERVER={os.getenv('START_AGENT_SERVER')} AGENT_BASE_URL={agent_base_url or 'local'} OPENAI_API_KEY={'set' if k else "missing"} fpr={k_fpr}")
        server_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "main:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
            ],
            cwd=str(SCRIPT_DIR),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=server_env,
        )
        # Give the server a moment to start
        time.sleep(1.5)
    else:
        if not agent_base_url:
            agent_base_url = "http://127.0.0.1:5000"
        server_proc = None




    async def call_agent_act(
        prepared_task: Task,
        episode_task_id: str,
        snapshot_html: str,
        url: str,
        step_index: int,
        history: list[dict],
        requested_model: str,
        screenshot: str | None = None,
    ) -> tuple[list[BaseAction], dict]:
        import aiohttp

        payload = {
            "task_id": str(episode_task_id),
            "prompt": prepared_task.prompt,
            "url": url,
            "snapshot_html": snapshot_html,
            "screenshot": screenshot,
            "step_index": int(step_index),
            "web_project_id": prepared_task.web_project_id,
            "history": history,
            "model": str(requested_model),
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{agent_base_url}/act", json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()

        metrics = data.get("metrics") if isinstance(data, dict) else {}
        if not isinstance(metrics, dict):
            metrics = {}

        actions_payload = data.get("actions") if isinstance(data, dict) else None
        if not isinstance(actions_payload, list):
            return [], metrics
        actions: list[BaseAction] = []
        for raw in actions_payload:
            if not isinstance(raw, dict):
                continue
            try:
                act = BaseAction.create_action(raw)
                if act is not None:
                    actions.append(act)
            except Exception:
                continue
        return actions, metrics

    # Results tracking
    requested_model = str(model)
    results = {
        "provider": provider_s,
        "model": model,
        "requested_model": requested_model,
        "num_tasks": 0,
        "successes": 0,
        "failures": 0,
        "errors": 0,
        "model_mismatch_errors": 0,
        "timing": {
            "total_seconds": 0.0,
            "avg_task_seconds": 0.0,
            "avg_step_seconds": 0.0,
        },
        "episodes": [],
    }

    t_start = time.time()

    for i, base_task in enumerate(tasks):
        reps = max(1, int(repeat))
        for r in range(reps):
            # If a fixed seed is provided, vary it across repeats to explore different pages.
            seed_i = (int(seed) + int(r)) if seed is not None else None
            task, seed_used = inject_seed(base_task, seed=seed_i)

            uc_name = ""
            if hasattr(task, "use_case") and task.use_case:
                uc = task.use_case
                if isinstance(uc, dict):
                    uc_name = uc.get("name", "unknown")
                elif hasattr(uc, "name"):
                    uc_name = uc.name
                else:
                    uc_name = str(uc)

            rep_label = f" r={r + 1}/{reps}" if reps > 1 else ""
            logger.info(
                f"[{i + 1}/{len(tasks)}]{rep_label} seed={seed_used} | {uc_name} | {task.prompt[:50]}..."
            )

            task_start = time.time()
            episode_llm_calls = 0
            episode_prompt_tokens = 0
            episode_completion_tokens = 0
            episode_total_tokens = 0
            episode_cost_usd = 0.0
            episode_model = str(model)

            try:
                prepared_task = task
                evaluator = None
                # Unique id for the agent/gateway header to avoid leaking state across repeats.
                episode_task_id = f"{prepared_task.id}-{seed_used}-{r}"

                evaluator = AsyncStatefulEvaluator(task=prepared_task, web_agent_id=web_agent_id)
                step_result = await evaluator.reset()

                history: list[dict] = []

                final_score = 0.0
                final_success = False
                total_steps = 0

                for step_idx in range(max_steps):
                    actions, metrics = await call_agent_act(
                        prepared_task,
                        episode_task_id=episode_task_id,
                        snapshot_html=step_result.snapshot.html,
                        url=step_result.snapshot.url,
                        step_index=step_idx,
                        screenshot=_serialize_screenshot(getattr(step_result.snapshot, "screenshot", None)),
                        history=history,
                        requested_model=str(requested_model),
                    )

                    # Metrics are returned by the agent when AGENT_RETURN_METRICS=1.
                    llm_meta = metrics.get("llm") if isinstance(metrics, dict) else None
                    if isinstance(llm_meta, dict):
                        usages = llm_meta.get("llm_usages")
                        model_name = llm_meta.get("model") or model
                        episode_model = str(model_name)
                        if bool(strict_model) and str(episode_model).strip() != str(requested_model).strip():
                            raise RuntimeError(
                                f"model_mismatch requested={requested_model} effective={episode_model} task={episode_task_id} step={step_idx}"
                            )
                        if isinstance(usages, list):
                            for u in usages:
                                if not isinstance(u, dict):
                                    continue
                                pt = int(u.get("prompt_tokens") or 0)
                                ct = int(u.get("completion_tokens") or 0)
                                tt = int(u.get("total_tokens") or (pt + ct))
                                episode_prompt_tokens += pt
                                episode_completion_tokens += ct
                                episode_total_tokens += tt
                                c, _ = estimate_cost_usd(str(model_name), u)
                                episode_cost_usd += float(c)
                            episode_llm_calls += int(llm_meta.get("llm_calls") or len(usages))
                    if actions:
                        action = actions[0]
                        step_result = await evaluator.step(action)
                    else:
                        action = None
                        step_result = await evaluator.step(None)

                    # Optional: persist per-step trace for debugging.
                    if os.getenv('EVAL_SAVE_TRACES', '0').lower() in {'1','true','yes'}:
                        try:
                            trace_dir = (SCRIPT_DIR / 'data' / 'traces' / str(episode_task_id))
                            trace_dir.mkdir(parents=True, exist_ok=True)
                            (trace_dir / f'{step_idx:02d}.url.txt').write_text(str(step_result.snapshot.url), encoding='utf-8')
                            (trace_dir / f'{step_idx:02d}.html').write_text(str(step_result.snapshot.html), encoding='utf-8', errors='replace')
                        except Exception:
                            pass

                    final_score = step_result.score.raw_score
                    final_success = step_result.score.success
                    total_steps = step_idx + 1

                    cid = None
                    if isinstance(metrics, dict):
                        cid = metrics.get("candidate_id")
                    if isinstance(cid, str) and cid.isdigit():
                        cid = int(cid)

                    ar = step_result.action_result
                    exec_ok = True
                    exec_err = None
                    try:
                        if ar is not None:
                            exec_ok = bool(getattr(ar, 'successfully_executed', True))
                            exec_err = getattr(ar, 'error', None)
                    except Exception:
                        exec_ok = True
                        exec_err = None

                    history.append({
                        "step": step_idx,
                        "url": str(step_result.snapshot.url),
                        "action": action.type if action else "done",
                        "candidate_id": cid,
                        "text": getattr(action, "text", None) if action else None,
                        "exec_ok": exec_ok,
                        "error": exec_err,
                        "agent_decision": (metrics.get("decision") if isinstance(metrics, dict) else None),
                        "llm_calls": int((llm_meta.get("llm_calls") if isinstance(llm_meta, dict) else 0) or 0),
                        "prompt_tokens": int(sum(int(u.get("prompt_tokens") or 0) for u in (llm_meta.get("llm_usages") if isinstance(llm_meta, dict) and isinstance(llm_meta.get("llm_usages"), list) else []))),
                        "completion_tokens": int(sum(int(u.get("completion_tokens") or 0) for u in (llm_meta.get("llm_usages") if isinstance(llm_meta, dict) and isinstance(llm_meta.get("llm_usages"), list) else []))),
                    })

                    if final_success:
                        break

                await evaluator.close()

                task_elapsed = time.time() - task_start

                # If failed, persist the last HTML snapshot for quick iteration/debugging.
                if not final_success:
                    try:
                        out_dir = SCRIPT_DIR / "data"
                        fail_dir = out_dir / "failures"
                        fail_dir.mkdir(parents=True, exist_ok=True)
                        (fail_dir / f"{episode_task_id}.url.txt").write_text(str(step_result.snapshot.url), encoding="utf-8")
                        (fail_dir / f"{episode_task_id}.html").write_text(str(step_result.snapshot.html), encoding="utf-8", errors="replace")
                    except Exception:
                        pass

                results["num_tasks"] += 1
                steps_count = total_steps or 0
                avg_step_seconds = (task_elapsed / steps_count) if steps_count > 0 else 0.0
                ep_data = {
                    "task_id": str(task.id),
                    "episode_task_id": str(episode_task_id),
                    "model": str(episode_model),
                    "repeat_index": int(r),
                    "use_case": uc_name,
                    "seed": seed_used,
                    "success": bool(final_success),
                    "score": float(final_score),
                    "steps": steps_count,
                    "task_seconds": round(task_elapsed, 4),
                    "llm_calls": int(episode_llm_calls),
                    "prompt_tokens": int(episode_prompt_tokens),
                    "completion_tokens": int(episode_completion_tokens),
                    "total_tokens": int(episode_total_tokens),
                    "estimated_cost_usd": round(float(episode_cost_usd), 6),
                    "avg_step_seconds": round(avg_step_seconds, 4),
                }
                results["episodes"].append(ep_data)

                if final_success:
                    results["successes"] += 1
                    logger.info(f"  -> SUCCESS (score={final_score:.2f}, steps={steps_count})")
                else:
                    results["failures"] += 1
                    logger.info(f"  -> FAILED  (score={final_score:.2f}, steps={steps_count})")

            except Exception as e:
                # Best-effort cleanup (errors can happen before we reach the normal close path).
                try:
                    if 'evaluator' in locals() and locals().get('evaluator') is not None:
                        await locals()['evaluator'].close()
                except Exception:
                    pass
                task_elapsed = time.time() - task_start
                results["num_tasks"] += 1
                results["errors"] += 1
                if "model_mismatch" in str(e):
                    results["model_mismatch_errors"] += 1
                results["episodes"].append({
                    "task_id": str(task.id),
                    "episode_task_id": str(getattr(locals().get('prepared_task', None), 'id', task.id)),
                    "repeat_index": int(r),
                    "use_case": uc_name,
                    "seed": seed_used,
                    "success": False,
                    "score": 0.0,
                    "steps": 0,
                    "task_seconds": round(task_elapsed, 4),
                    "llm_calls": int(episode_llm_calls),
                    "prompt_tokens": int(episode_prompt_tokens),
                    "completion_tokens": int(episode_completion_tokens),
                    "total_tokens": int(episode_total_tokens),
                    "estimated_cost_usd": round(float(episode_cost_usd), 6),
                    "avg_step_seconds": 0.0,
                    "error": str(e),
                })
                logger.error(f"  -> ERROR: {e}")
    elapsed = time.time() - t_start

    # ── Summary ──────────────────────────────────────────────────
    total = results["num_tasks"]
    succ = results["successes"]
    rate = succ / total if total > 0 else 0
    avg_score = (
        sum(ep["score"] for ep in results["episodes"]) / total if total > 0 else 0
    )
    avg_steps = (
        sum(ep["steps"] for ep in results["episodes"]) / total if total > 0 else 0
    )
    avg_task_seconds = (
        sum(ep.get("task_seconds", 0.0) for ep in results["episodes"]) / total if total > 0 else 0
    )
    total_steps = sum(ep["steps"] for ep in results["episodes"])
    avg_step_seconds = (
        sum(ep.get("task_seconds", 0.0) for ep in results["episodes"]) / total_steps
        if total_steps > 0
        else 0
    )

    results["timing"]["total_seconds"] = round(elapsed, 4)
    results["timing"]["avg_task_seconds"] = round(avg_task_seconds, 4)
    results["timing"]["avg_step_seconds"] = round(avg_step_seconds, 4)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Provider:       {provider_s}")
    print(f"  Model:          {model}")
    print(f"  Tasks run:      {total}")
    print(f"  Successes:      {succ}")
    print(f"  Failures:       {results['failures']}")
    print(f"  Errors:         {results['errors']}")
    print(f"  Model mismatch: {results.get('model_mismatch_errors', 0)}")
    print(f"  Success rate:   {rate:.1%}")
    print(f"  Avg score:      {avg_score:.3f}")
    print(f"  Avg steps:      {avg_steps:.1f}")
    print(f"  Avg task time:  {avg_task_seconds:.2f}s")
    total_cost = sum(float(ep.get("estimated_cost_usd", 0.0)) for ep in results["episodes"])
    avg_cost = (total_cost / total) if total > 0 else 0.0
    total_tokens_all = sum(int(ep.get("total_tokens", 0)) for ep in results["episodes"])
    print(f"  Avg step time:  {avg_step_seconds:.2f}s")
    print(f"  Est. cost:      ${total_cost:.4f} (avg ${avg_cost:.4f}/task)")
    print(f"  Total tokens:   {total_tokens_all}")
    print(f"  Total time:     {elapsed:.1f}s")
    print("=" * 60)

    # Per-use-case breakdown
    uc_stats: dict[str, dict] = {}
    for ep in results["episodes"]:
        uc = ep.get("use_case", "unknown")
        if uc not in uc_stats:
            uc_stats[uc] = {"total": 0, "success": 0}
        uc_stats[uc]["total"] += 1
        if ep["success"]:
            uc_stats[uc]["success"] += 1

    if len(uc_stats) > 1:
        print("\n  Per use-case breakdown:")
        for uc, st in sorted(uc_stats.items()):
            uc_rate = st["success"] / st["total"] if st["total"] > 0 else 0
            print(f"    {uc:30s}  {st['success']}/{st['total']}  ({uc_rate:.0%})")
        print()

    # Save results
    out_dir = SCRIPT_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path).resolve() if out_path else (out_dir / 'eval_results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {out_path}\n")

    if server_proc:
        try:
            log_f.flush()
            log_f.close()
        except Exception:
            pass
        server_proc.terminate()
        server_proc.wait(timeout=5)

    return results


# ── CLI ──────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Autoppia Operator - LLM Agent Evaluation")
    parser.add_argument('--provider', default='chutes', help='LLM provider: openai|chutes|anthropic')
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3-0324", help="Model name")
    parser.add_argument("--num-tasks", type=int, default=20, help="Number of tasks to evaluate")
    parser.add_argument("--max-steps", type=int, default=15, help="Max steps per episode")
    parser.add_argument("--use-case", default=None, help="Filter by use case (e.g. LOGIN)")
    parser.add_argument("--web-project-id", default=None, help="Filter by web_project_id (exact match)")
    parser.add_argument("--task-id", default=None, help="Run a specific task id (exact match)")
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed (otherwise random)")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat each selected task N times")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument('--out', default=None, help='Output JSON path (default: data/eval_results.json)')
    parser.add_argument('--task-cache', default=None, help='Task cache JSON path')
    parser.add_argument("--strict-model", action=argparse.BooleanOptionalAction, default=True, help="Fail episodes when effective model != requested model")
    parser.add_argument("--distinct-use-cases", action="store_true", help="Pick tasks with distinct use cases")
    args = parser.parse_args()

    asyncio.run(
        run_evaluation(
            provider=args.provider,
            model=args.model,
            num_tasks=args.num_tasks,
            max_steps=args.max_steps,
            use_case=args.use_case,
            web_project_id=args.web_project_id,
            task_id=args.task_id,
            seed=args.seed,
            repeat=args.repeat,
            temperature=args.temperature,
            distinct_use_cases=bool(args.distinct_use_cases),
            out_path=args.out,
            task_cache=args.task_cache,
            strict_model=bool(args.strict_model),
        )
    )


if __name__ == "__main__":
    main()
