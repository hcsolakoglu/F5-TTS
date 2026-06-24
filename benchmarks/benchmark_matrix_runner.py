#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = Path("/content/f5tts_matrix_spec.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a JSON-defined F5-TTS benchmark matrix.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    return parser.parse_args()


def load_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if not isinstance(spec.get("cases"), list):
        raise ValueError("Benchmark matrix spec must contain a list field named 'cases'.")
    return spec


def run_case(case: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    name = case["name"]
    timeout = float(case.get("timeout_s", 1800))
    args = [str(item) for item in case["args"]]
    output_path = output_dir / f"{name}.json"
    command = [
        sys.executable,
        str(REPO_ROOT / "benchmarks" / "eager_training_baseline.py"),
        *args,
        "--output",
        str(output_path),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    status = "passed" if completed.returncode == 0 else "failed"
    result: dict[str, Any] = {
        "name": name,
        "status": status,
        "returncode": completed.returncode,
        "elapsed_s": elapsed,
        "timeout_s": timeout,
        "command": command,
        "output_path": str(output_path),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if output_path.exists():
        result["artifact_exists"] = True
    return result


def main() -> None:
    args = parse_args()
    spec = load_spec(args.spec)
    output_dir = Path(spec.get("output_dir", "/content/f5tts_matrix_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "spec_path": str(args.spec),
        "repo_root": str(REPO_ROOT),
        "cwd": os.getcwd(),
        "cases": [],
    }
    for case in spec["cases"]:
        try:
            result = run_case(case, output_dir)
        except subprocess.TimeoutExpired as exc:
            result = {
                "name": case["name"],
                "status": "timeout",
                "elapsed_s": float(case.get("timeout_s", 1800)),
                "timeout_s": float(case.get("timeout_s", 1800)),
                "command": exc.cmd,
                "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            }
        summary["cases"].append(result)
        summary_path = output_dir / "matrix_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    failed = [case for case in summary["cases"] if case["status"] not in {"passed"}]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
