#!/usr/bin/env python3
"""compare_eval.py

Run eval.py across multiple provider/model combos and aggregate results.

Example:
  python scripts/compare_eval.py     --runs openai:gpt-5.2 openai:gpt-4o openai:o4-mini     --runs anthropic:claude-sonnet-4-5-20250929     --num-tasks 5 --distinct-use-cases

This script writes:
  data/compare/<provider>__<model>.json
  data/compare/compare_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RunSpec:
    provider: str
    model: str

    @property
    def slug(self) -> str:
        # filesystem safe
        s = f"{self.provider}__{self.model}".lower()
        s = re.sub(r"[^a-z0-9._-]+", "_", s)
        return s


def _parse_run(s: str) -> RunSpec:
    if ':' not in s:
        raise SystemExit(f"Invalid run '{s}'. Use provider:model, e.g. openai:gpt-5.2")
    prov, model = s.split(':', 1)
    prov = prov.strip().lower()
    model = model.strip()
    if not prov or not model:
        raise SystemExit(f"Invalid run '{s}'.")
    return RunSpec(provider=prov, model=model)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> None:
    ap = argparse.ArgumentParser(description='Compare eval across multiple models/providers')
    ap.add_argument('--runs', nargs='+', action='append', required=True, help='List of provider:model entries')
    ap.add_argument('--num-tasks', type=int, default=5)
    ap.add_argument('--max-steps', type=int, default=15)
    ap.add_argument('--distinct-use-cases', action='store_true')
    ap.add_argument('--use-case', default=None)
    ap.add_argument('--web-project-id', default=None)
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--temperature', type=float, default=0.2)
    args = ap.parse_args()

    runs: list[RunSpec] = []
    for group in (args.runs or []):
        for item in group:
            runs.append(_parse_run(item))

    out_dir = REPO_DIR / 'data' / 'compare'
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for r in runs:
        out_path = out_dir / f"{r.slug}.json"
        cmd = [
            os.environ.get('PYTHON', 'python'),
            str(REPO_DIR / 'eval.py'),
            '--provider', r.provider,
            '--model', r.model,
            '--num-tasks', str(int(args.num_tasks)),
            '--max-steps', str(int(args.max_steps)),
            '--repeat', str(int(args.repeat)),
            '--temperature', str(float(args.temperature)),
            '--out', str(out_path),
        ]
        if args.distinct_use_cases:
            cmd.append('--distinct-use-cases')
        if args.use_case:
            cmd += ['--use-case', str(args.use_case)]
        if args.web_project_id:
            cmd += ['--web-project-id', str(args.web_project_id)]

        print(f"\n=== RUN {r.provider}:{r.model} -> {out_path} ===")
        proc = subprocess.run(cmd, cwd=str(REPO_DIR), env=os.environ.copy())
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

        j = _load_json(out_path)
        # compute summary
        total = int(j.get('num_tasks') or 0)
        succ = int(j.get('successes') or 0)
        rate = (succ / total) if total else 0.0
        total_cost = 0.0
        total_tokens = 0
        avg_task_s = float(j.get('timing', {}).get('avg_task_seconds') or 0.0)
        for ep in j.get('episodes', []) if isinstance(j.get('episodes'), list) else []:
            try:
                total_cost += float(ep.get('estimated_cost_usd') or 0.0)
            except Exception:
                pass
            try:
                total_tokens += int(ep.get('total_tokens') or 0)
            except Exception:
                pass

        results.append({
            'provider': j.get('provider') or r.provider,
            'model': j.get('model') or r.model,
            'success_rate': rate,
            'successes': succ,
            'tasks': total,
            'avg_task_seconds': avg_task_s,
            'total_tokens': total_tokens,
            'estimated_cost_usd': round(total_cost, 6),
            'out_path': str(out_path),
        })

    # Pretty print
    print("\n=== COMPARE SUMMARY ===")
    results_sorted = sorted(results, key=lambda x: (-float(x.get('success_rate') or 0.0), float(x.get('avg_task_seconds') or 0.0)))
    for row in results_sorted:
        prov = str(row['provider'])
        model = str(row['model'])
        sr = float(row['success_rate'])
        tasks = int(row['tasks'])
        succ = int(row['successes'])
        t = float(row['avg_task_seconds'])
        cost = float(row['estimated_cost_usd'])
        toks = int(row['total_tokens'])
        print(f"{prov:10s} {model:40.40s}  {succ}/{tasks} ({sr:.0%})  avg_task={t:6.2f}s  tokens={toks:7d}  est_cost=${cost:.4f}")

    summary = {
        'runs': results_sorted,
    }
    (out_dir / 'compare_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"\nWrote: {out_dir / 'compare_summary.json'}")


if __name__ == '__main__':
    main()
