"""
NetSage AI — human-in-the-loop review and metrics.

Nothing this project produces is a "fix" until a human has looked at it. This module owns
that gate:

    Accepted  the AI diagnosis is correct as written and may be applied
    Edited    the AI was on the right track but a human corrected it before use
    Rejected  the AI was wrong; the human diagnosis replaces it entirely

Every decision is appended to `data/reviews.csv` with a timestamp and a reviewer name, so
the log is an audit trail rather than a snapshot. The latest decision per case wins.

The metrics here deliberately separate three different questions that are easy to
conflate:

    1. Was the AI right?         -> auto-graded against the lab's known answer
    2. Did a human agree?        -> the accept / edit / reject split
    3. Did it cite real evidence? -> the anti-hallucination gate

A high score on (1) with a low score on (3) means the model guessed well, not that it
reasoned well, and the dashboard shows both.
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
REVIEWS_PATH = ROOT / "data" / "reviews.csv"

DECISIONS = ("Accepted", "Edited", "Rejected")

REVIEW_FIELDS = [
    "review_id", "timestamp", "case_id", "reviewer", "decision",
    "ai_root_cause", "ai_concept_tag", "ai_osi_layer", "ai_confidence",
    "corrected_root_cause", "corrected_concept_tag", "corrected_osi_layer",
    "corrected_next_command", "notes", "provider", "model",
]


# ======================================================================================
# Record
# ======================================================================================

@dataclass
class ReviewRecord:
    case_id: str
    decision: str
    reviewer: str = "reviewer"
    ai_root_cause: str = ""
    ai_concept_tag: str = ""
    ai_osi_layer: str = ""
    ai_confidence: str = ""
    corrected_root_cause: str = ""
    corrected_concept_tag: str = ""
    corrected_osi_layer: str = ""
    corrected_next_command: str = ""
    notes: str = ""
    provider: str = ""
    model: str = ""
    timestamp: str = ""
    review_id: str = ""

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}, got {self.decision!r}")
        if not self.timestamp:
            self.timestamp = _dt.datetime.now().isoformat(timespec="seconds")
        if not self.review_id:
            self.review_id = f"{self.case_id}-{self.timestamp.replace(':', '').replace('-', '')}"
        if self.decision == "Accepted" and not self.corrected_root_cause:
            self.corrected_root_cause = self.ai_root_cause
            self.corrected_concept_tag = self.corrected_concept_tag or self.ai_concept_tag
            self.corrected_osi_layer = self.corrected_osi_layer or self.ai_osi_layer

    @property
    def final_root_cause(self) -> str:
        return self.corrected_root_cause or self.ai_root_cause

    @property
    def final_concept_tag(self) -> str:
        return self.corrected_concept_tag or self.ai_concept_tag

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: d.get(k, "") for k in REVIEW_FIELDS}


# ======================================================================================
# Store
# ======================================================================================

class ReviewStore:
    """Append-only CSV of review decisions. Latest decision per case wins."""

    def __init__(self, path: Path | str = REVIEWS_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_all([])

    # -- io ----------------------------------------------------------------------------

    def _write_all(self, rows: Sequence[Dict[str, Any]]) -> None:
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=REVIEW_FIELDS, quoting=csv.QUOTE_ALL,
                               lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in REVIEW_FIELDS})

    def all_rows(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        with open(self.path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def add(self, record: ReviewRecord) -> ReviewRecord:
        rows = self.all_rows()
        rows.append(record.to_row())
        self._write_all(rows)
        return record

    def add_many(self, records: Iterable[ReviewRecord]) -> int:
        rows = self.all_rows()
        n = 0
        for r in records:
            rows.append(r.to_row())
            n += 1
        self._write_all(rows)
        return n

    def latest(self) -> Dict[str, Dict[str, Any]]:
        """case_id -> most recent review row."""
        out: Dict[str, Dict[str, Any]] = {}
        for row in self.all_rows():
            cid = row.get("case_id", "")
            if not cid:
                continue
            prev = out.get(cid)
            if prev is None or str(row.get("timestamp", "")) >= str(prev.get("timestamp", "")):
                out[cid] = row
        return out

    def for_case(self, case_id: str) -> List[Dict[str, Any]]:
        return [r for r in self.all_rows() if r.get("case_id") == case_id]

    def clear(self) -> None:
        self._write_all([])

    def __len__(self) -> int:
        return len(self.latest())


# ======================================================================================
# Metrics
# ======================================================================================

@dataclass
class Metrics:
    total_cases: int = 0
    diagnosed: int = 0
    reviewed: int = 0

    accepted: int = 0
    edited: int = 0
    rejected: int = 0

    ai_correct: int = 0
    layer_correct: int = 0
    evidence_grounded: int = 0
    fell_back_to_mock: int = 0
    parse_repaired: int = 0

    rule_coverage: int = 0
    rule_findings_total: int = 0

    by_concept: Dict[str, int] = field(default_factory=dict)
    by_layer: Dict[str, int] = field(default_factory=dict)
    by_severity: Dict[str, int] = field(default_factory=dict)
    by_confidence: Dict[str, int] = field(default_factory=dict)
    disagreement_by_concept: Dict[str, int] = field(default_factory=dict)

    mean_keyword_overlap: float = 0.0
    mean_latency_ms: float = 0.0

    # -- derived rates -----------------------------------------------------------------

    @property
    def agreement_rate(self) -> float:
        """Share of REVIEWED cases a human accepted without edits."""
        return self.accepted / self.reviewed if self.reviewed else 0.0

    @property
    def usable_rate(self) -> float:
        """Accepted plus edited: the AI gave the reviewer a usable starting point."""
        return (self.accepted + self.edited) / self.reviewed if self.reviewed else 0.0

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.reviewed if self.reviewed else 0.0

    @property
    def auto_accuracy(self) -> float:
        """Auto-graded accuracy against the lab's known-correct answer."""
        return self.ai_correct / self.diagnosed if self.diagnosed else 0.0

    @property
    def grounding_rate(self) -> float:
        return self.evidence_grounded / self.diagnosed if self.diagnosed else 0.0

    @property
    def rule_coverage_rate(self) -> float:
        return self.rule_coverage / self.diagnosed if self.diagnosed else 0.0

    @property
    def review_completion(self) -> float:
        return self.reviewed / self.total_cases if self.total_cases else 0.0

    def as_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in asdict(self).items()}
        d.update({
            "agreement_rate": round(self.agreement_rate, 4),
            "usable_rate": round(self.usable_rate, 4),
            "rejection_rate": round(self.rejection_rate, 4),
            "auto_accuracy": round(self.auto_accuracy, 4),
            "grounding_rate": round(self.grounding_rate, 4),
            "rule_coverage_rate": round(self.rule_coverage_rate, 4),
            "review_completion": round(self.review_completion, 4),
        })
        return d

    def summary_lines(self) -> List[str]:
        return [
            f"Cases                 : {self.total_cases}",
            f"Diagnosed             : {self.diagnosed}",
            f"Rule-checker coverage : {self.rule_coverage}/{self.diagnosed} "
            f"({self.rule_coverage_rate:.0%})  |  {self.rule_findings_total} findings total",
            f"Auto-graded accuracy  : {self.ai_correct}/{self.diagnosed} ({self.auto_accuracy:.0%})",
            f"Evidence grounded     : {self.evidence_grounded}/{self.diagnosed} "
            f"({self.grounding_rate:.0%})",
            f"Root-cause overlap    : {self.mean_keyword_overlap:.2f} mean "
            f"(text similarity to the known answer)",
            f"Reviewed              : {self.reviewed}/{self.total_cases} "
            f"({self.review_completion:.0%})",
            f"  Accepted            : {self.accepted}",
            f"  Edited              : {self.edited}",
            f"  Rejected            : {self.rejected}",
            f"AI/human agreement    : {self.agreement_rate:.0%} accepted as written, "
            f"{self.usable_rate:.0%} usable after edit",
        ]


class ReviewManager:
    """Ties cases, AI outputs and human decisions together and computes the metrics."""

    def __init__(self, store: Optional[ReviewStore] = None):
        # `store or ReviewStore()` is WRONG here: ReviewStore defines __len__, so an empty
        # store is falsy and would be silently swapped for one pointing at the default path.
        # That let a test write its rows into the real data/reviews.csv.
        self.store = store if store is not None else ReviewStore()

    # -- decisions ---------------------------------------------------------------------

    def accept(self, case_id: str, result, reviewer: str = "reviewer",
               notes: str = "") -> ReviewRecord:
        return self.store.add(self._record(case_id, result, "Accepted", reviewer, notes=notes))

    def edit(self, case_id: str, result, corrected_root_cause: str,
             corrected_concept_tag: str = "", corrected_osi_layer: str = "",
             corrected_next_command: str = "", reviewer: str = "reviewer",
             notes: str = "") -> ReviewRecord:
        return self.store.add(self._record(
            case_id, result, "Edited", reviewer, notes=notes,
            corrected_root_cause=corrected_root_cause,
            corrected_concept_tag=corrected_concept_tag,
            corrected_osi_layer=corrected_osi_layer,
            corrected_next_command=corrected_next_command,
        ))

    def reject(self, case_id: str, result, corrected_root_cause: str,
               corrected_concept_tag: str = "", corrected_osi_layer: str = "",
               corrected_next_command: str = "", reviewer: str = "reviewer",
               notes: str = "") -> ReviewRecord:
        return self.store.add(self._record(
            case_id, result, "Rejected", reviewer, notes=notes,
            corrected_root_cause=corrected_root_cause,
            corrected_concept_tag=corrected_concept_tag,
            corrected_osi_layer=corrected_osi_layer,
            corrected_next_command=corrected_next_command,
        ))

    @staticmethod
    def _record(case_id: str, result, decision: str, reviewer: str,
                notes: str = "", **corrections) -> ReviewRecord:
        d = getattr(result, "diagnosis", None)
        return ReviewRecord(
            case_id=case_id,
            decision=decision,
            reviewer=reviewer or "reviewer",
            ai_root_cause=getattr(d, "root_cause", "") if d else "",
            ai_concept_tag=getattr(d, "concept_tag", "") if d else "",
            ai_osi_layer=getattr(d, "osi_layer", "") if d else "",
            ai_confidence=getattr(d, "confidence", "") if d else "",
            provider=getattr(result, "provider", ""),
            model=getattr(result, "model", ""),
            notes=notes,
            **corrections,
        )

    # -- auto review -------------------------------------------------------------------

    def auto_review(self, cases: Sequence[Dict[str, Any]], results: Sequence[Any],
                    reviewer: str = "auto-grader") -> int:
        """Seed the review log by grading every diagnosis against the known answer.

        This is a convenience for producing a full dashboard in one command; it is NOT a
        substitute for human review and it labels itself as `auto-grader` in the log so the
        two can never be confused. A case the auto-grader marks Accepted can still be
        edited by a person afterwards, and the later decision wins.
        """
        records = []
        for case, res in zip(cases, results):
            g = res.grade(case)
            if g["ai_correct"] and g["keyword_overlap"] >= 0.30:
                records.append(self._record(case["case_id"], res, "Accepted", reviewer,
                                            notes="Auto-graded: concept, layer and evidence "
                                                  "all match the known answer."))
            elif g["concept_correct"]:
                records.append(self._record(
                    case["case_id"], res, "Edited", reviewer,
                    notes="Auto-graded: correct subsystem, but the root-cause wording is "
                          "imprecise or the evidence citation is weak. Replaced with the "
                          "lab's verified answer.",
                    corrected_root_cause=case.get("expected_root_cause", ""),
                    corrected_concept_tag=case.get("concept_tag", ""),
                    corrected_osi_layer=case.get("osi_layer", ""),
                    corrected_next_command=case.get("expected_next_command", ""),
                ))
            else:
                records.append(self._record(
                    case["case_id"], res, "Rejected", reviewer,
                    notes="Auto-graded: the AI identified the wrong subsystem. Replaced with "
                          "the lab's verified answer.",
                    corrected_root_cause=case.get("expected_root_cause", ""),
                    corrected_concept_tag=case.get("concept_tag", ""),
                    corrected_osi_layer=case.get("osi_layer", ""),
                    corrected_next_command=case.get("expected_next_command", ""),
                ))
        return self.store.add_many(records)

    # -- metrics -----------------------------------------------------------------------

    def metrics(self, cases: Sequence[Dict[str, Any]],
                results: Optional[Sequence[Any]] = None,
                ai_rows: Optional[Sequence[Dict[str, Any]]] = None) -> Metrics:
        """Compute metrics from live results or from a previously written outputs CSV."""
        m = Metrics(total_cases=len(cases))
        by_id = {str(c.get("case_id")): c for c in cases}

        rows: List[Dict[str, Any]] = []
        if results is not None:
            rows = [r.to_row(by_id.get(r.case_id, {})) for r in results]
        elif ai_rows is not None:
            rows = [dict(r) for r in ai_rows]

        overlaps: List[float] = []
        latencies: List[float] = []
        for row in rows:
            m.diagnosed += 1
            if _truthy(row.get("ai_correct")):
                m.ai_correct += 1
            if _truthy(row.get("layer_correct")):
                m.layer_correct += 1
            if _truthy(row.get("evidence_grounded")):
                m.evidence_grounded += 1
            if _truthy(row.get("fell_back_to_mock")):
                m.fell_back_to_mock += 1
            if _truthy(row.get("parse_repaired")):
                m.parse_repaired += 1
            n_find = _int(row.get("rule_finding_count"))
            m.rule_findings_total += n_find
            if n_find > 0:
                m.rule_coverage += 1
            overlaps.append(_float(row.get("keyword_overlap")))
            latencies.append(_float(row.get("latency_ms")))
            _bump(m.by_confidence, row.get("ai_confidence"))

        m.mean_keyword_overlap = round(sum(overlaps) / len(overlaps), 3) if overlaps else 0.0
        m.mean_latency_ms = round(sum(latencies) / len(latencies), 1) if latencies else 0.0

        # Case-level distributions come from the dataset itself so the charts are complete
        # even before a single case has been diagnosed.
        for c in cases:
            _bump(m.by_concept, c.get("concept_tag"))
            _bump(m.by_layer, c.get("osi_layer"))
            _bump(m.by_severity, c.get("severity"))

        latest = self.store.latest()
        for cid, row in latest.items():
            if cid not in by_id:
                continue
            m.reviewed += 1
            dec = row.get("decision", "")
            if dec == "Accepted":
                m.accepted += 1
            elif dec == "Edited":
                m.edited += 1
            elif dec == "Rejected":
                m.rejected += 1
            if dec in ("Edited", "Rejected"):
                _bump(m.disagreement_by_concept, by_id[cid].get("concept_tag"))
        return m

    # -- export ------------------------------------------------------------------------

    def merged_table(self, cases: Sequence[Dict[str, Any]],
                     ai_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Join cases + AI outputs + latest review into one flat table for the dashboard."""
        latest = self.store.latest()
        ai_by_id = {str(r.get("case_id")): r for r in ai_rows}
        out = []
        for c in cases:
            cid = str(c.get("case_id"))
            row: Dict[str, Any] = {
                "case_id": cid,
                "title": c.get("title", ""),
                "concept_tag": c.get("concept_tag", ""),
                "osi_layer": c.get("osi_layer", ""),
                "severity": c.get("severity", ""),
                "expected_fault": c.get("expected_fault", ""),
            }
            ai = ai_by_id.get(cid, {})
            row.update({
                "ai_root_cause": ai.get("ai_root_cause", ""),
                "ai_concept_tag": ai.get("ai_concept_tag", ""),
                "ai_osi_layer": ai.get("ai_osi_layer", ""),
                "ai_confidence": ai.get("ai_confidence", ""),
                "ai_correct": ai.get("ai_correct", ""),
                "evidence_grounded": ai.get("evidence_grounded", ""),
                "rule_findings": ai.get("rule_findings", ""),
            })
            rv = latest.get(cid, {})
            row.update({
                "decision": rv.get("decision", "Not reviewed"),
                "reviewer": rv.get("reviewer", ""),
                "review_notes": rv.get("notes", ""),
                "final_root_cause": rv.get("corrected_root_cause") or ai.get("ai_root_cause", ""),
            })
            out.append(row)
        return out


# ======================================================================================
# helpers
# ======================================================================================

def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}


def _int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _bump(d: Dict[str, int], key: Any) -> None:
    k = str(key or "").strip()
    if k:
        d[k] = d.get(k, 0) + 1


# --------------------------------------------------------------------------------------
# Self-test:  python -m src.review_manager
# --------------------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    from .ai_engine import DiagnosisEngine

    with open(ROOT / "data" / "cases.csv", newline="", encoding="utf-8") as fh:
        cases = list(csv.DictReader(fh))

    engine = DiagnosisEngine.from_env()
    results = engine.diagnose_batch(cases)

    mgr = ReviewManager(ReviewStore(ROOT / "data" / "_selftest_reviews.csv"))
    mgr.store.clear()
    mgr.auto_review(cases, results)

    m = mgr.metrics(cases, results)
    print("\n".join(m.summary_lines()))
    print("\nBy concept   :", m.by_concept)
    print("Disagreements:", m.disagreement_by_concept)
    os.remove(ROOT / "data" / "_selftest_reviews.csv")
