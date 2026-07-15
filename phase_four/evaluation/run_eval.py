r"""
evaluation/run_eval.py — Phase 4 comparison harness

Runs evaluation/test_cases.json through all three configurations:
    1. baseline_simple_rag      (Simple RAG, no agent, no triage)
    2. ablation_agent_no_triage (RAG + agent, no triage classifier)
    3. full_agent_with_triage   (RAG + agent + triage classifier — proposed)

and produces:
    evaluation/results/raw_results.json   — every (config, test case) run
    evaluation/results/summary.csv        — aggregated metrics per config
    a printed summary table

HOW TO RUN (use forward slashes in the model path, even on Windows):
    python phase_four/evaluation/run_eval.py --model C:/Users/gunn/models/Phi-3-mini-4k-instruct-q4.gguf

    Optional:
        --db PATH        (default: data/lancedb)
        --limit N         only run the first N test cases (fast smoke test)
        --configs a,b,c   only run specific configs, e.g. --configs baseline_simple_rag
"""

import sys
import json
import time
import argparse
import csv
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.configs import build_configs
from evaluation.metrics import keyword_score, refusal_is_correct, intent_match, source_grounded

DEFAULT_DB_PATH = "data/lancedb"
TEST_CASES_PATH = Path(__file__).parent / "test_cases.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_test_cases(limit: int | None = None) -> list[dict]:
    cases = json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))
    return cases[:limit] if limit else cases


def run_one(config, case: dict) -> dict:
    t0 = time.time()
    try:
        result = config.run(case["query"])
        error = None
    except Exception as e:
        result = {"answer": "", "sources": [], "intent": None, "confidence": None, "iterations": None}
        error = str(e)
    latency = round(time.time() - t0, 3)

    answer = result["answer"]
    row = {
        "config": config.name,
        "case_id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "answer": answer,
        "reported_intent": result["intent"],
        "expected_intent": case.get("expected_intent"),
        "intent_match": intent_match(result["intent"], case.get("expected_intent")),
        "keyword_score": keyword_score(answer, case.get("expected_keywords", [])),
        "should_refuse": case.get("should_refuse", False),
        "refusal_correct": refusal_is_correct(answer, case.get("should_refuse", False)),
        "num_sources": len(result["sources"]),
        "source_grounded": source_grounded(result["sources"]),
        "iterations": result["iterations"],
        "latency_sec": latency,
        "error": error,
    }
    return row


def summarize(rows: list[dict]) -> list[dict]:
    by_config = defaultdict(list)
    for r in rows:
        by_config[r["config"]].append(r)

    summary = []
    for config_name, config_rows in by_config.items():
        n = len(config_rows)
        errors = sum(1 for r in config_rows if r["error"])

        intent_checked = [r for r in config_rows if r["intent_match"] != "n/a"]
        intent_acc = (
            round(sum(1 for r in intent_checked if r["intent_match"] == "match") / len(intent_checked), 3)
            if intent_checked else None
        )

        summary.append({
            "config": config_name,
            "n_cases": n,
            "errors": errors,
            "avg_keyword_score": round(sum(r["keyword_score"] for r in config_rows) / n, 3),
            "intent_accuracy": intent_acc,
            "refusal_correct_rate": round(sum(r["refusal_correct"] for r in config_rows) / n, 3),
            "source_grounded_rate": round(sum(r["source_grounded"] for r in config_rows) / n, 3),
            "avg_latency_sec": round(sum(r["latency_sec"] for r in config_rows) / n, 3),
        })

    # Keep a stable, meaningful order for the paper table
    order = {"baseline_simple_rag": 0, "ablation_agent_no_triage": 1, "full_agent_with_triage": 2}
    summary.sort(key=lambda s: order.get(s["config"], 99))
    return summary


def print_summary_table(summary: list[dict]):
    headers = ["config", "n_cases", "errors", "avg_keyword_score",
               "intent_accuracy", "refusal_correct_rate", "source_grounded_rate", "avg_latency_sec"]
    widths = {h: max(len(h), *(len(str(row[h])) for row in summary)) for h in headers}

    def fmt_row(values):
        return " | ".join(str(v).ljust(widths[h]) for h, v in zip(headers, values))

    print("\n" + "=" * 100)
    print("PHASE 4 — CONFIGURATION COMPARISON")
    print("=" * 100)
    print(fmt_row(headers))
    print("-" * 100)
    for row in summary:
        print(fmt_row([row[h] for h in headers]))
    print("=" * 100)


def write_outputs(rows: list[dict], summary: list[dict]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RESULTS_DIR / "raw_results.json"
    raw_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    summary_path = RESULTS_DIR / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nRaw results: {raw_path}")
    print(f"Summary CSV: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4 — 3-config comparison harness")
    parser.add_argument("--model", required=True, help="Path to the .gguf model file")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to LanceDB folder")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N test cases")
    parser.add_argument(
        "--configs", default=None,
        help="Comma-separated subset of configs to run, e.g. baseline_simple_rag,full_agent_with_triage"
    )
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"[ERROR] Model not found: {args.model}")
        sys.exit(1)
    if not Path(args.db).exists():
        print(f"[ERROR] Database not found: {args.db}\nRun Phase 1 first: python phase_one/ingest.py")
        sys.exit(1)

    cases = load_test_cases(limit=args.limit)
    print(f"Loaded {len(cases)} test cases from {TEST_CASES_PATH}")

    all_configs = build_configs(db_path=args.db, model_path=args.model)
    if args.configs:
        wanted = set(args.configs.split(","))
        all_configs = {k: v for k, v in all_configs.items() if k in wanted}
        if not all_configs:
            print(f"[ERROR] None of the requested configs matched: {args.configs}")
            sys.exit(1)

    rows = []
    total_runs = len(all_configs) * len(cases)
    done = 0
    for config_name, config in all_configs.items():
        print(f"\n{'#'*60}\nRUNNING CONFIG: {config_name}\n{'#'*60}")
        for case in cases:
            done += 1
            print(f"[{done}/{total_runs}] {config_name} :: {case['id']} — {case['query'][:60]}")
            rows.append(run_one(config, case))

    summary = summarize(rows)
    print_summary_table(summary)
    write_outputs(rows, summary)


if __name__ == "__main__":
    main()