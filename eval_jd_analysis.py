# -*- coding: utf-8 -*-
"""Evaluate grounded JDAnalysis on phenomena-dev or private full-JD sets.

The default run uses an isolated temporary cache.  This prevents previously
generated model outputs from inflating latency, stability or extraction
results.  ``--reuse-cache`` is available only for an explicit warm-cache run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import time
from typing import Any

from eval_private_sets import (PrivateEvalIntegrityError, PrivateEvalUnavailable,
                               load_private_eval, unavailable_report)


FIXTURE = Path(__file__).resolve().parent / "tests" / "fixtures" / "jd_analysis_frozen_v1.json"
BLIND_MANIFEST = Path(__file__).resolve().parent / "tests" / "fixtures" / "jd_blind_v1.manifest.json"
SET_ALIASES = {
    "phenomena_dev": "jd_phenomena_dev_v1",
    "blind": "jd_blind_v1",
}


def _load_phenomena_cases() -> list[dict]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cases = []
    for group in data["groups"]:
        for index, line in enumerate(group["lines"]):
            jd_text = f"{group['prefix']}\n{group['heading']}\n{line}"
            start = jd_text.index(line)
            cases.append({
                "id": f"{group['id']}:{index:02d}",
                "category": group["id"],
                "jd_text": jd_text,
                "evidence": line,
                "kind": group["kind"],
                "modality": group["modality"],
                "requirements": [{
                    "requirement_id": f"{group['id']}:{index:02d}:r1",
                    "text": line,
                    "start": start,
                    "end": start + len(line),
                    "kind": group["kind"],
                    "modality": group["modality"],
                    "alternative_group": None,
                    "keywords": [],
                }],
                "expected_fields": {},
            })
    if len(cases) != int(data["expected_count"]):
        raise RuntimeError("JD regression-set count mismatch")
    if len({case["jd_text"] for case in cases}) != len(cases):
        raise RuntimeError("JD regression set contains duplicate documents")
    return cases


def _load_blind_cases(private_root: str | Path | None = None) -> tuple[list[dict], str]:
    bundle = load_private_eval(BLIND_MANIFEST, private_root=private_root)
    cases: list[dict] = []
    for raw in bundle.data["items"]:
        if not isinstance(raw, dict):
            raise PrivateEvalIntegrityError("private JD item must be an object")
        jd_text = str(raw.get("jd_text") or "")
        if not (800 <= len(jd_text) <= 30000):
            raise PrivateEvalIntegrityError(
                "private JD text must be between 800 and 30000 characters"
            )
        requirements = list(raw.get("requirements") or [])
        requirement_ids: list[str] = []
        for gold in requirements:
            try:
                start, end = int(gold["start"]), int(gold["end"])
                text = str(gold["text"])
                requirement_ids.append(str(gold["requirement_id"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise PrivateEvalIntegrityError(
                    "private JD requirement is missing its span contract"
                ) from exc
            if not (0 <= start < end <= len(jd_text) and jd_text[start:end] == text):
                raise PrivateEvalIntegrityError(
                    "private JD gold span does not match the source document"
                )
        if len(set(requirement_ids)) != len(requirement_ids):
            raise PrivateEvalIntegrityError("duplicate requirement_id in private JD")
        cases.append({
            "id": str(raw["id"]),
            "category": str(raw.get("category") or "unknown"),
            "jd_text": jd_text,
            "requirements": requirements,
            "expected_fields": dict(raw.get("expected_fields") or {}),
            "annotation": dict(raw.get("annotation") or {}),
        })
    manifest = json.loads(BLIND_MANIFEST.read_text(encoding="utf-8"))
    expected_quotas = {str(key): int(value) for key, value in manifest["quotas"].items()}
    actual_quotas = Counter(case["category"] for case in cases)
    if dict(actual_quotas) != expected_quotas:
        raise PrivateEvalIntegrityError(
            f"private JD quota mismatch: {dict(actual_quotas)} != {expected_quotas}"
        )
    double_annotations = [
        case["annotation"] for case in cases
        if case["annotation"].get("double_annotated")
    ]
    if len(double_annotations) < int(manifest.get("minimum_double_annotated") or 0):
        raise PrivateEvalIntegrityError("private JD double-annotation quota is not met")
    minimum_kappa = float(manifest.get("minimum_agreement_kappa") or 0)
    if any(
        not annotation.get("adjudicated")
        or float(annotation.get("agreement_kappa") or 0) < minimum_kappa
        for annotation in double_annotations
    ):
        raise PrivateEvalIntegrityError(
            "private JD annotation agreement/adjudication gate is not met"
        )
    return cases, bundle.sha256


def load_cases(set_name: str = "phenomena_dev", *,
               private_root: str | Path | None = None) -> list[dict]:
    """Compatibility loader; defaults to the 120-case phenomena dev set."""
    if set_name == "phenomena_dev":
        return _load_phenomena_cases()
    if set_name == "blind":
        return _load_blind_cases(private_root)[0]
    raise ValueError(f"unknown JD evaluation set: {set_name}")


def _normalise_text(value: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(value or "").lower(), flags=re.UNICODE)


def _span_valid(jd_text: str, span: Any) -> bool:
    try:
        start, end, text = int(span.start), int(span.end), str(span.text)
    except (AttributeError, TypeError, ValueError):
        return False
    return 0 <= start < end <= len(jd_text) and jd_text[start:end] == text


def _requirement_evidence_entails(requirement: Any, jd_text: str) -> bool:
    """Conservative deterministic support check, separate from offset validity.

    This is intentionally stricter than merely locating an arbitrary source
    span.  The normalised claim must contain, or be contained by, at least one
    exact source span.  It is an auditable lexical entailment proxy rather than
    an LLM judge.
    """
    claim = _normalise_text(getattr(requirement, "text", ""))
    spans = list(getattr(requirement, "evidence_spans", []) or [])
    if not claim or not spans or not all(_span_valid(jd_text, span) for span in spans):
        return False
    for span in spans:
        evidence = _normalise_text(span.text)
        if evidence and (claim in evidence or evidence in claim):
            return True
    return False


def _matched_requirement(analysis: Any, expected: str) -> Any | None:
    expected_normalised = _normalise_text(expected)
    for requirement in analysis.requirements:
        claim = _normalise_text(requirement.text)
        if claim == expected_normalised:
            return requirement
        if any(_normalise_text(span.text) == expected_normalised
               for span in requirement.evidence_spans):
            return requirement
    return None


def _gold_prediction_score(gold: dict[str, Any], requirement: Any) -> tuple[int, bool]:
    """Return a conservative lexical match score and exact-span status."""
    gold_text = str(gold.get("text") or "")
    gold_normalised = _normalise_text(gold_text)
    try:
        gold_start, gold_end = int(gold["start"]), int(gold["end"])
    except (KeyError, TypeError, ValueError):
        gold_start = gold_end = -1
    spans = list(getattr(requirement, "evidence_spans", []) or [])
    exact_span = any(
        int(getattr(span, "start", -2)) == gold_start
        and int(getattr(span, "end", -2)) == gold_end
        and str(getattr(span, "text", "")) == gold_text
        for span in spans
    )
    if exact_span:
        return 3, True
    if any(_normalise_text(getattr(span, "text", "")) == gold_normalised
           for span in spans):
        return 2, False
    if _normalise_text(getattr(requirement, "text", "")) == gold_normalised:
        return 1, False
    return 0, False


def _match_requirements(gold: list[dict[str, Any]], predicted: list[Any]) -> list[dict[str, Any]]:
    """One-to-one best lexical matching for full-document micro/macro metrics."""
    candidates: list[tuple[int, int, int, bool]] = []
    for gold_index, annotation in enumerate(gold):
        for predicted_index, requirement in enumerate(predicted):
            score, exact_span = _gold_prediction_score(annotation, requirement)
            if score:
                candidates.append((score, gold_index, predicted_index, exact_span))
    candidates.sort(reverse=True)
    used_gold: set[int] = set()
    used_predicted: set[int] = set()
    matches: list[dict[str, Any]] = []
    for score, gold_index, predicted_index, exact_span in candidates:
        if gold_index in used_gold or predicted_index in used_predicted:
            continue
        used_gold.add(gold_index)
        used_predicted.add(predicted_index)
        matches.append({
            "gold_index": gold_index,
            "predicted_index": predicted_index,
            "score": score,
            "exact_span": exact_span,
        })
    return matches


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * .95) - 1)]


def _ratio(hits: int, cases: int) -> float:
    return hits / cases if cases else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=sorted(SET_ALIASES), default="phenomena_dev")
    parser.add_argument("--private-root", default="",
                        help="Override OFFERCLAW_PRIVATE_EVAL_ROOT for blind evaluation")
    parser.add_argument("--mode", choices=["deterministic", "shadow", "intelligent"],
                        default="intelligent")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--reuse-cache", action="store_true",
                        help="Use the normal JD cache (default: isolated temporary cache)")
    parser.add_argument("--no-gate", action="store_true",
                        help="Report metrics without returning non-zero on threshold failure")
    parser.add_argument("--min-requirement-recall", type=float, default=.95)
    parser.add_argument("--min-requirement-precision", type=float, default=.95)
    parser.add_argument("--min-kind-accuracy", type=float, default=.90)
    parser.add_argument("--min-modality-accuracy", type=float, default=.90)
    parser.add_argument("--min-offset-validity", type=float, default=1.0)
    parser.add_argument("--min-evidence-entailment", type=float, default=1.0)
    parser.add_argument("--min-gold-span-recall", type=float, default=.95)
    parser.add_argument("--min-schema-valid-rate", type=float, default=.995)
    parser.add_argument("--max-p95-ms", type=float, default=0.0,
                        help="Optional extraction p95 ceiling; 0 disables this gate")
    args = parser.parse_args()
    os.environ["JD_ANALYZER_MODE"] = args.mode
    from jd_parser import JDAnalysis, JD_ANALYSIS_SCHEMA_VERSION, analyze_jd

    private_hash = ""
    try:
        if args.set == "blind":
            cases, private_hash = _load_blind_cases(args.private_root or None)
        else:
            cases = load_cases(args.set)
    except (PrivateEvalUnavailable, PrivateEvalIntegrityError) as exc:
        report = unavailable_report(dataset_id=SET_ALIASES[args.set], error=exc)
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        print(rendered)
        return 2
    if args.set == "blind" and args.limit:
        parser.error("--limit is not allowed for aggregate-only blind evaluation")
    cases = cases[:args.limit or None]
    if not cases:
        parser.error("evaluation set is empty")

    cache_context = nullcontext(None) if args.reuse_cache else TemporaryDirectory(
        prefix="offerclaw-jd-eval-"
    )
    rows: list[dict[str, Any]] = []
    total_spans = valid_spans = 0
    total_requirements = entailed_requirements = 0
    gold_requirements = matched_requirements = exact_gold_spans = 0
    kind_hits = modality_hits = 0
    macro_precisions: list[float] = []
    macro_recalls: list[float] = []
    expected_field_total = expected_field_hits = 0
    total_keywords = grounded_keywords = 0
    latencies: list[float] = []
    timeout_count = fallback_count = 0

    with cache_context as isolated_cache:
        cache_dir = None if args.reuse_cache else isolated_cache
        for case in cases:
            started = time.perf_counter()
            analysis = analyze_jd(case["jd_text"], mode=args.mode, cache_dir=cache_dir)
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            latencies.append(latency_ms)
            gold = list(case.get("requirements") or [])
            predicted = list(analysis.requirements)
            matches = _match_requirements(gold, predicted)
            gold_requirements += len(gold)
            matched_requirements += len(matches)
            exact_gold_spans += sum(match["exact_span"] for match in matches)
            document_precision = len(matches) / len(predicted) if predicted else (
                1.0 if not gold else 0.0
            )
            document_recall = len(matches) / len(gold) if gold else 1.0
            macro_precisions.append(document_precision)
            macro_recalls.append(document_recall)
            document_kind_ok = True
            document_modality_ok = True
            for match in matches:
                expected_requirement = gold[match["gold_index"]]
                predicted_requirement = predicted[match["predicted_index"]]
                kind_match = str(predicted_requirement.kind) == str(expected_requirement["kind"])
                modality_match = (
                    str(predicted_requirement.modality) == str(expected_requirement["modality"])
                )
                kind_hits += int(kind_match)
                modality_hits += int(modality_match)
                document_kind_ok = document_kind_ok and kind_match
                document_modality_ok = document_modality_ok and modality_match

            field_results: dict[str, bool] = {}
            for field_name, expected_value in dict(case.get("expected_fields") or {}).items():
                expected_field_total += 1
                actual_value = getattr(analysis, field_name, None)
                field_ok = _normalise_text(actual_value) == _normalise_text(expected_value)
                field_results[field_name] = field_ok
                expected_field_hits += int(field_ok)

            requirement_spans = [span for requirement in analysis.requirements
                                 for span in requirement.evidence_spans]
            keyword_spans = [span for keyword in analysis.keywords
                             for span in keyword.evidence_spans]
            spans = requirement_spans + keyword_spans
            span_results = [_span_valid(case["jd_text"], span) for span in spans]
            total_spans += len(span_results)
            valid_spans += sum(span_results)
            # Empty output must not receive perfect document-level grounding.
            offset_valid = bool(span_results) and all(span_results)

            requirement_support = [
                _requirement_evidence_entails(requirement, case["jd_text"])
                for requirement in analysis.requirements
            ]
            total_requirements += len(requirement_support)
            entailed_requirements += sum(requirement_support)
            evidence_entails = bool(requirement_support) and all(requirement_support)

            keyword_support: list[bool] = []
            for keyword in analysis.keywords:
                spans_ok = bool(keyword.evidence_spans) and all(
                    _span_valid(case["jd_text"], span) for span in keyword.evidence_spans
                )
                forms_ok = bool(keyword.surface_forms) and all(
                    re.search(
                        re.escape(str(form)), case["jd_text"], re.IGNORECASE
                    ) is not None for form in keyword.surface_forms
                )
                keyword_support.append(spans_ok and forms_ok)
            total_keywords += len(keyword_support)
            grounded_keywords += sum(keyword_support)

            schema_valid = bool(
                isinstance(analysis, JDAnalysis)
                and analysis.schema_version == JD_ANALYSIS_SCHEMA_VERSION
                and analysis.input_hash
                and analysis.source in {"llm", "deterministic", "cache", "legacy"}
            )
            intelligent_result = analysis.source in {"llm", "cache"}
            recalled = len(matches) == len(gold)
            kind_ok = recalled and document_kind_ok
            modality_ok = recalled and document_modality_ok
            warnings_text = " ".join(str(item) for item in analysis.warnings).lower()
            timed_out = "timeout" in warnings_text or "超时" in warnings_text
            used_fallback = args.mode == "intelligent" and not intelligent_result
            timeout_count += int(timed_out)
            fallback_count += int(used_fallback)
            failed = bool(
                not recalled or not kind_ok or not modality_ok or not offset_valid
                or not evidence_entails or not schema_valid
                or any(not value for value in field_results.values())
                or (args.mode == "intelligent" and not intelligent_result)
            )
            rows.append({
                "id": case["id"],
                "category": case.get("category", ""),
                "source": analysis.source,
                "model": analysis.model,
                "schema_version": analysis.schema_version,
                "schema_valid": schema_valid,
                "intelligent_result": intelligent_result,
                "requirement_count": len(analysis.requirements),
                "gold_requirement_count": len(gold),
                "matched_requirement_count": len(matches),
                "requirement_precision": document_precision,
                "requirement_recall": document_recall,
                "exact_gold_span_count": sum(match["exact_span"] for match in matches),
                "keyword_count": len(analysis.keywords),
                "span_count": len(spans),
                "offset_valid": offset_valid,
                "requirement_evidence_entails": evidence_entails,
                "recalled": recalled,
                "kind_ok": kind_ok,
                "modality_ok": modality_ok,
                "expected_fields": field_results,
                "timed_out": timed_out,
                "used_fallback": used_fallback,
                "warnings": analysis.warnings,
                "latency_ms": latency_ms,
                "failed": failed,
            })

    recall = _ratio(matched_requirements, gold_requirements)
    requirement_precision = _ratio(matched_requirements, total_requirements)
    macro_recall = _ratio(sum(macro_recalls), len(macro_recalls))
    macro_precision = _ratio(sum(macro_precisions), len(macro_precisions))
    kind_accuracy = _ratio(kind_hits, matched_requirements)
    modality_accuracy = _ratio(modality_hits, matched_requirements)
    exact_span_recall = _ratio(exact_gold_spans, gold_requirements)
    exact_span_on_matched = _ratio(exact_gold_spans, matched_requirements)
    expected_field_accuracy = _ratio(expected_field_hits, expected_field_total)
    span_offset_validity = _ratio(valid_spans, total_spans)
    offset_document_rate = _ratio(sum(row["offset_valid"] for row in rows), len(rows))
    evidence_entailment = _ratio(entailed_requirements, total_requirements)
    evidence_document_rate = _ratio(
        sum(row["requirement_evidence_entails"] for row in rows), len(rows)
    )
    keyword_grounding = _ratio(grounded_keywords, total_keywords)
    schema_valid_rate = _ratio(sum(row["schema_valid"] for row in rows), len(rows))
    intelligent_result_rate = _ratio(
        sum(row["intelligent_result"] for row in rows), len(rows)
    )
    p95_ms = _p95(latencies)
    p50_ms = sorted(latencies)[max(0, math.ceil(len(latencies) * .50) - 1)]
    p99_ms = sorted(latencies)[max(0, math.ceil(len(latencies) * .99) - 1)]

    gates = {
        "requirement_recall": recall >= args.min_requirement_recall,
        "requirement_precision": requirement_precision >= args.min_requirement_precision,
        "requirement_macro_recall": macro_recall >= args.min_requirement_recall,
        "requirement_macro_precision": macro_precision >= args.min_requirement_precision,
        "kind_accuracy_on_recalled": kind_accuracy >= args.min_kind_accuracy,
        "modality_accuracy_on_recalled": modality_accuracy >= args.min_modality_accuracy,
        "span_offset_validity": (
            total_spans > 0 and span_offset_validity >= args.min_offset_validity
            and offset_document_rate >= args.min_offset_validity
        ),
        "requirement_evidence_entailment": (
            total_requirements > 0 and evidence_entailment >= args.min_evidence_entailment
            and evidence_document_rate >= args.min_evidence_entailment
        ),
        "gold_exact_span_recall": exact_span_recall >= args.min_gold_span_recall,
        "schema_valid_rate": schema_valid_rate >= args.min_schema_valid_rate,
    }
    if args.mode == "intelligent":
        gates["intelligent_result_rate"] = intelligent_result_rate >= args.min_schema_valid_rate
    if args.max_p95_ms > 0:
        gates["p95_ms"] = p95_ms <= args.max_p95_ms

    by_category: dict[str, dict[str, float | int]] = {}
    for category in sorted({str(row.get("category") or "unknown") for row in rows}):
        category_rows = [row for row in rows if str(row.get("category") or "unknown") == category]
        category_gold = sum(int(row["gold_requirement_count"]) for row in category_rows)
        category_predicted = sum(int(row["requirement_count"]) for row in category_rows)
        category_matched = sum(int(row["matched_requirement_count"]) for row in category_rows)
        by_category[category] = {
            "documents": len(category_rows),
            "micro_precision": round(_ratio(category_matched, category_predicted), 6),
            "micro_recall": round(_ratio(category_matched, category_gold), 6),
            "macro_precision": round(
                sum(float(row["requirement_precision"]) for row in category_rows)
                / len(category_rows), 6,
            ),
            "macro_recall": round(
                sum(float(row["requirement_recall"]) for row in category_rows)
                / len(category_rows), 6,
            ),
        }

    report = {
        "set_version": SET_ALIASES[args.set],
        "set_kind": (
            "private_blind" if args.set == "blind" else "phenomena_development"
        ),
        "private_dataset_sha256": private_hash or None,
        "mode": args.mode,
        "cache_mode": "shared_reuse" if args.reuse_cache else "isolated_cold",
        "count": len(rows),
        "requirement_recall": round(recall, 6),
        "requirement_precision": round(requirement_precision, 6),
        "requirement_micro_recall": round(recall, 6),
        "requirement_micro_precision": round(requirement_precision, 6),
        "requirement_macro_recall": round(macro_recall, 6),
        "requirement_macro_precision": round(macro_precision, 6),
        "gold_exact_span_recall": round(exact_span_recall, 6),
        "gold_exact_span_on_matched": round(exact_span_on_matched, 6),
        "kind_accuracy_on_recalled": round(kind_accuracy, 6),
        "modality_accuracy_on_recalled": round(modality_accuracy, 6),
        "span_offset_validity": round(span_offset_validity, 6),
        "offset_valid_document_rate": round(offset_document_rate, 6),
        "requirement_evidence_entailment": round(evidence_entailment, 6),
        "evidence_entailment_document_rate": round(evidence_document_rate, 6),
        "keyword_grounding_rate": round(keyword_grounding, 6),
        "schema_valid_rate": round(schema_valid_rate, 6),
        "expected_field_accuracy": (
            round(expected_field_accuracy, 6) if expected_field_total else None
        ),
        "intelligent_result_rate": round(intelligent_result_rate, 6),
        "total_predicted_requirements": total_requirements,
        "total_gold_requirements": gold_requirements,
        "total_matched_requirements": matched_requirements,
        "total_predicted_keywords": total_keywords,
        "total_evidence_spans": total_spans,
        "p95_ms": round(p95_ms, 1),
        "latency_ms": {
            "p50": round(p50_ms, 1),
            "p95": round(p95_ms, 1),
            "p99": round(p99_ms, 1),
            "max": round(max(latencies), 1),
        },
        "timeout_count": timeout_count,
        "timeout_rate": round(timeout_count / len(rows), 6),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / len(rows), 6),
        "sources": dict(Counter(row["source"] for row in rows)),
        "models": dict(Counter(row["model"] or "<none>" for row in rows)),
        "schema_versions": dict(Counter(row["schema_version"] for row in rows)),
        "by_category": by_category,
        "gates": gates,
        "passed": all(gates.values()),
        "failure_count": sum(row["failed"] for row in rows),
    }
    if args.set != "blind":
        report["failures"] = [row for row in rows if row["failed"]]
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)
    if args.no_gate:
        return 0
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
