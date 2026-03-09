#!/usr/bin/env python3
"""Generate and cache benchmark tasks using autoppia_iwa's TaskGenerationPipeline.

Why this exists:
- Running ad-hoc python does NOT load this repo's .env by default.
- autoppia_iwa may load its own env or rely on process env vars.
- We want an explicit way to load autoppia_operator/.env (override=True) so
  OPENAI_API_KEY is always the one we expect.

Default output matches what eval.py consumes:
  ../autoppia_rl/data/task_cache/autoppia_cinema_tasks.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def _load_operator_env(operator_dir: Path) -> None:
    # Load dotenv before importing autoppia_iwa so our env wins.
    try:
        from dotenv import load_dotenv
    except Exception as e:  # pragma: no cover
        raise RuntimeError("python-dotenv is required to load .env for task generation") from e

    env_path = operator_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)


async def _generate(project_id: str, prompts_per_use_case: int, dynamic: bool) -> dict:
    from autoppia_iwa.entrypoints.benchmark.utils.task_generation import (
        generate_tasks_for_project,
        get_projects_by_ids,
    )
    from autoppia_iwa.src.demo_webs.config import demo_web_projects

    [project] = get_projects_by_ids(demo_web_projects, [project_id])
    tasks = await generate_tasks_for_project(
        project,
        prompts_per_use_case=prompts_per_use_case,
        use_cases=None,
        dynamic=dynamic,
    )

    return {
        "project_id": project.id,
        "project_name": project.name,
        "timestamp": datetime.now().isoformat(),
        "tasks": [t.serialize() for t in tasks],
    }


def main() -> None:
    operator_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Generate and cache tasks via autoppia_iwa.")
    parser.add_argument(
        "--project-id",
        action="append",
        default=[],
        help="Web project id (repeatable). Example: --project-id autocinema --project-id autobooks",
    )
    parser.add_argument(
        "--project-ids",
        default=None,
        help="Comma-separated project ids (alternative to repeating --project-id)",
    )
    parser.add_argument("--prompts-per-use-case", type=int, default=1, help="Prompt variants per use case")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic task generation")
    parser.add_argument(
        "--out",
        default=str(operator_dir / "data" / "task_cache" / "tasks_cache.json"),
        help="Output cache JSON path",
    )
    args = parser.parse_args()

    project_ids: list[str] = []
    if args.project_ids:
        project_ids.extend([p.strip() for p in str(args.project_ids).split(',') if p.strip()])
    project_ids.extend([p.strip() for p in (args.project_id or []) if p.strip()])
    if not project_ids:
        project_ids = ["autocinema"]

    _load_operator_env(operator_dir)

    k = os.getenv("OPENAI_API_KEY") or ""
    k_fpr = hashlib.sha256(k.encode("utf-8")).hexdigest()[:12] if k else "missing"
    if not k:
        raise SystemExit("OPENAI_API_KEY missing in environment (check autoppia_operator/.env).")
    print(f"OPENAI_API_KEY=set fpr={k_fpr}")

    async def _run() -> dict:
        all_tasks = []
        projects = []
        for pid in project_ids:
            payload = await _generate(pid, int(args.prompts_per_use_case), bool(args.dynamic))
            projects.append({
                'project_id': payload.get('project_id'),
                'project_name': payload.get('project_name'),
                'num_tasks': len(payload.get('tasks') or []),
            })
            all_tasks.extend(payload.get('tasks') or [])

        return {
            'timestamp': datetime.now().isoformat(),
            'projects': projects,
            'tasks': all_tasks,
        }

    payload = asyncio.run(_run())

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {len(payload['tasks'])} tasks from {len(payload['projects'])} projects -> {out_path}")

    if not payload["tasks"]:
        raise SystemExit("Generated 0 tasks (check OPENAI key/model access and generation logs).")


if __name__ == "__main__":
    main()
