#!/usr/bin/env python3
"""
NetSage AI — command line pipeline runner.

Runs the full chain over the case dataset and writes the artefacts the dashboard reads:

    data/cases.csv  ->  rule_checker  ->  ai_engine  ->  data/ai_outputs.csv
                                                     ->  data/reviews.csv  (with --auto-review)

Examples
--------
    python run_pipeline.py                       # all 30 cases, offline mock engine
    python run_pipeline.py --auto-review         # also seed the review log and print metrics
    python run_pipeline.py --provider groq       # use the Groq free tier (needs GROQ_API_KEY)
    python run_pipeline.py --case NS-07 --verbose  # one case, full diagnosis printed
    python run_pipeline.py --rules-only          # deterministic checks only, no AI call
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.ai_engine import DiagnosisEngine, PROVIDERS, ProviderError  # noqa: E402
from src.review_manager import ReviewManager, ReviewStore  # noqa: E402
from src.rule_checker import RuleChecker  # noqa: E402

CASES_PATH = ROOT / "data" / "cases.csv"
OUTPUT_PATH = ROOT / "data" / "ai_outputs.csv"
REVIEWS_PATH = ROOT / "data" / "reviews.csv"

BAR = "=" * 86


def load_cases(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run this script from the netsage_ai/ directory.")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"ERROR: {path} is empty.")
    return rows


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, quoting=csv.QUOTE_ALL, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def cmd_rules_only(cases: List[Dict[str, Any]], verbose: bool) -> int:
    rc = RuleChecker()
    covered = total = 0
    print(BAR)
    print("Deterministic rule checker — no AI, no network, no cost")
    print(BAR)
    for case in cases:
        findings = [f for f in rc.run(case) if f.check_id != "CHECK_ERROR"]
        total += len(findings)
        covered += 1 if findings else 0
        status = f"{len(findings)} finding(s)" if findings else "no finding"
        print(f"\n{case['case_id']}  {case['title']}  [{status}]")
        print(f"   expected : {case['expected_fault']}")
        for f in findings:
            print(f"   -> [{f.severity}/{f.osi_layer}] {f.check_id}: {f.detail}")
            if verbose:
                for ev in f.evidence:
                    print(f"        evidence: {ev}")
    print("\n" + BAR)
    print(f"{covered}/{len(cases)} cases produced at least one finding — {total} findings total.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run the NetSage AI troubleshooting pipeline over the case dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--provider", choices=sorted(PROVIDERS), default=None,
                   help="AI provider. Defaults to NETSAGE_PROVIDER in .env, or 'mock'.")
    p.add_argument("--cases", type=Path, default=CASES_PATH, help="Path to cases.csv")
    p.add_argument("--out", type=Path, default=OUTPUT_PATH, help="Where to write AI outputs")
    p.add_argument("--case", action="append", default=None,
                   help="Run only this case id (repeatable), e.g. --case NS-07")
    p.add_argument("--limit", type=int, default=None, help="Run only the first N cases")
    p.add_argument("--auto-review", action="store_true",
                   help="Seed data/reviews.csv by grading each answer against the known result. "
                        "Labelled 'auto-grader' in the log; a human decision made later wins.")
    p.add_argument("--reset-reviews", action="store_true",
                   help="Empty data/reviews.csv before running (use with --auto-review).")
    p.add_argument("--rules-only", action="store_true",
                   help="Run only the deterministic checks and print them. No AI call.")
    p.add_argument("--no-fallback", action="store_true",
                   help="Fail loudly instead of falling back to the mock engine when a "
                        "provider call errors.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print the full diagnosis for every case.")
    args = p.parse_args()

    cases = load_cases(args.cases)
    if args.case:
        wanted = {c.strip().upper() for c in args.case}
        cases = [c for c in cases if str(c["case_id"]).upper() in wanted]
        if not cases:
            sys.exit(f"ERROR: no case matched {sorted(wanted)}")
    if args.limit:
        cases = cases[:args.limit]

    if args.rules_only:
        return cmd_rules_only(cases, args.verbose)

    try:
        engine = DiagnosisEngine.from_env(args.provider, fallback_to_mock=not args.no_fallback)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    print(BAR)
    print("NetSage AI — pipeline run")
    print(BAR)
    print(f"cases    : {len(cases)}  (from {args.cases})")
    print(f"provider : {engine.provider_status()}")
    if engine.provider_name != "mock" and not engine.provider.is_configured:
        print("           >>> add the API key to .env, or run without --provider to use the "
              "offline mock engine.")
    print(BAR)

    results = []
    for i, case in enumerate(cases, 1):
        try:
            res = engine.diagnose(case)
        except ProviderError as exc:
            sys.exit(f"\nERROR on {case['case_id']}: {exc}\n"
                     f"Re-run without --no-fallback to continue using the mock engine.")
        results.append(res)
        g = res.grade(case)
        mark = "OK  " if g["ai_correct"] else "MISS"
        note = ""
        if res.fell_back_to_mock:
            note = "  [fell back to mock: " + res.error[:60] + "]"
        print(f"[{i:>2}/{len(cases)}] {mark} {res.case_id}  "
              f"{res.diagnosis.concept_tag:<18} conf={res.diagnosis.confidence:<6} "
              f"{len([f for f in res.rule_findings if f.check_id != 'CHECK_ERROR'])} rule-finding(s)"
              f"{note}")
        if args.verbose:
            d = res.diagnosis
            print(f"        root cause : {d.root_cause}")
            print(f"        next cmd   : {d.next_command}")
            for ev in d.evidence:
                print(f"        evidence   : {ev}")
            for step in d.fix_steps:
                print(f"        fix        : {step}")
            print(f"        risk       : {d.risk_note}")
            print(f"        expected   : {case['expected_fault']}")

    rows = [r.to_row(c) for r, c in zip(results, cases)]
    write_rows(args.out, rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    store = ReviewStore(REVIEWS_PATH)
    if args.reset_reviews:
        store.clear()
        print(f"Cleared {REVIEWS_PATH}")
    mgr = ReviewManager(store)
    if args.auto_review:
        n = mgr.auto_review(cases, results)
        print(f"Seeded {n} auto-graded review decisions in {REVIEWS_PATH}")

    m = mgr.metrics(cases, results)
    print("\n" + BAR)
    print("SUMMARY")
    print(BAR)
    for line in m.summary_lines():
        print("  " + line)
    if engine.provider_name == "mock":
        print("\n  NOTE: the mock engine reads the same deterministic findings the checker\n"
              "  produces, so its concept and layer accuracy measure whether the pipeline is\n"
              "  wired correctly — not how good an LLM is. For a real accuracy number, run\n"
              "  with --provider groq or --provider gemini and compare the two.")
    print(BAR)
    print("\nNext: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
