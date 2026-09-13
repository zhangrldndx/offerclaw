#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit the V2-A wave for question / answer-requirement / evidence consistency.

Every check here is deterministic and uses only the dataset plus the frozen
index — no LLM and no network, so the audit can be re-run and diffed.

Checks
------
``requirement_grounding``
    Each answer-requirement clause must be lexically grounded in the declared
    grade-3 evidence.  Only terms the knowledge base itself uses are judged, so
    a paraphrase is not punished for its connective tissue; what is punished is
    a clause that names domain terms the evidence never mentions.  A clause
    grounded in the full chunk but not in the excerpt means the excerpt markers
    are too narrow; a clause grounded in neither means the requirement
    over-claims and the gold is wrong.

    Known blind spot: this catches a requirement that reaches for terms the
    corpus attests *elsewhere* (the v2a-alg-004 case, where the encoder/decoder
    division of labour lives in the next chunk).  A requirement invented out of
    words no chunk uses at all scores 1.0 and passes — lexical grounding cannot
    see it, so evidence-first authoring remains the real guarantee.

``question_scope``
    A question must not demand content that is outside {topic, requirements,
    evidence}.  Conversational wording is separated from content by the corpus
    itself rather than by a hand-written stoplist: a term counts as content
    only if some anchor in the wave uses it in its topic, requirements, or
    evidence.  Anything else ("没法", "接算", background chit-chat) is invisible
    to the check, and a term that is content *somewhere else* but not here is
    exactly the drift signal we want.  ``long_noisy`` background is expected to
    reach outside scope, so its out-of-scope terms are reported but not failed.

``question_anchoring``
    Every question must share at least one content term with its own anchor's
    scope, otherwise it does not identify the knowledge point it is filed under.

``discriminativeness``
    Scoring a question against all anchors' (topic + requirements) by content
    term overlap must rank its own anchor first.  Losing to another anchor is
    the signature of scope drift or an ambiguous question.

``hard_negative_sanity``
    A hard negative whose requirement coverage approaches the gold's is the
    be02 failure mode (a heading digest that actually answers the question) and
    must be re-adjudicated rather than trained on.

``style_contract``
    Two objective bounds — ``long_noisy`` must actually be the longest form, and
    no style may recite the requirement (a soft answer-leak bound below the
    builder's verbatim one).  The two ``implicit_oral`` signals are reported at
    ``info`` only: colloquial Chinese cannot be enumerated by a marker list, and
    term overlap does not measure implicit reference either — "剪枝蒸馏量化之外
    还有一类是啥来着" uses every domain term and is still thoroughly oral.  They
    exist to route a human's eyes, not to judge.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_qrels import normalize_answer_span  # noqa: E402


AUDIT_SCHEMA_VERSION = "colloquial-v2a-semantic-consistency-v1"
DEFAULT_PUBLIC = ROOT / "docs/rag_eval/colloquial/rag_colloquial_v2a_train_dev_draft.json"
DEFAULT_JSON = ROOT / "docs/rag_eval/colloquial/V2A_SEMANTIC_CONSISTENCY_20260826.json"
DEFAULT_MD = ROOT / "docs/rag_eval/colloquial/V2A_SEMANTIC_CONSISTENCY_20260826.md"
DEFAULT_ADJUDICATION = (
    ROOT / "docs/rag_eval/colloquial/v2a_audit_adjudication_20260826.json"
)

# Requirement clause grounding thresholds, measured over corpus-attested terms.
CLAUSE_GROUNDED = 0.80
CLAUSE_WEAK = 0.60
# A term shared by more than this share of anchors is generic prose ("如何",
# "是什"), not a knowledge handle; only rarer terms carry scope information.
CONTENT_TERM_MAX_ANCHOR_SHARE = 0.10
# Drift is only claimed for terms that genuinely belong to a few knowledge
# points; a word used across a dozen anchors carries no ownership.
DRIFT_TERM_MAX_ANCHOR_DF = 3
# Another anchor must beat the true one by this IDF-weighted margin before it
# counts as a discriminativeness problem.
DISCRIMINATIVE_MARGIN = 1.0
# Boilerplate that every requirement starts with and no chunk ever contains.
_REQUIREMENT_PREFIX = re.compile(
    r"^答案必须(说明|列出|给出|覆盖|对比|区分|包含|指出|分别[^，]{0,4})?"
)
# A question reciting this much of a requirement is leaking the answer.
ANSWER_LEAK_COVERAGE = 0.70
# A hard negative this close to the gold's requirement coverage needs review.
NEGATIVE_REVIEW_RATIO = 0.75

_ORAL_MARKERS = (
    # particles and interjections
    "啊", "呗", "吧", "呢", "呀", "嘛", "哈", "咋", "啥", "来着",
    # demonstrative shorthands that stand in for a technical term
    "那个", "那套", "那种", "那事", "那招", "那版", "那几", "那条", "那块",
    "那步", "那栏", "这套", "这堆", "这玩意", "玩意", "事儿", "老办法",
    # colloquial verbs and figures of speech
    "整", "搞", "捞", "塞", "拧", "蹦", "掰扯", "懵", "卡住", "卡在", "翻车",
    "歇菜", "抬杠", "心虚", "背串", "记混", "凑齐", "接话", "别笑", "到底",
    "图个", "图的", "一堆", "咱", "俩", "仨", "多大", "谁管", "干嘛",
)
_QUESTION_CUES = ("？", "?", "吗", "什么", "哪", "如何", "怎么", "怎样", "为什么",
                  "为何", "多少", "是啥", "有啥", "咋", "呢")
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9+._/-]*")
_CJK_RUN = re.compile(r"[一-鿿]+")
# Bigrams are cut by a sliding window, so they routinely straddle word
# boundaries ("架构是" -> "构是").  A bigram counts as a knowledge handle only
# when neither character is grammatical glue; Chinese technical terms in this
# corpus ("虚拟内存", "布隆过滤", "缩放法则") never contain these characters.
_FUNCTION_CHARS = set(
    "的了是在和与或把被给让从到为着过就都也不没还又再才只吗呢吧啊呀嘛"
    "什么怎哪谁这那其之但而且能会要有对多少几很更最将得地上下前后里"
)


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def terms(text: str) -> set[str]:
    """Latin/numeric tokens plus Chinese bigrams, on normalized text."""

    normalized = normalize_answer_span(text)
    output = {
        token.lower() for token in _LATIN.findall(normalized) if len(token) >= 2
    }
    for run in _CJK_RUN.findall(normalized):
        output.update(run[index:index + 2] for index in range(len(run) - 1))
    return output


def is_content_term(term: str) -> bool:
    """True for Latin tokens and for Chinese bigrams free of grammatical glue."""

    if not term:
        return False
    if _LATIN.fullmatch(term):
        return True
    return all(char not in _FUNCTION_CHARS for char in term)


def coverage(needle: str, haystack_terms: set[str], *,
             vocabulary: set[str] | None = None) -> float:
    """Share of the needle's terms present in the haystack.

    ``vocabulary`` restricts scoring to terms the corpus itself attests, so a
    paraphrase is judged on the domain words it uses, not on its connectives.
    """

    values = terms(needle)
    if vocabulary is not None:
        values &= vocabulary
    if not values:
        return 1.0
    return len(values & haystack_terms) / len(values)


def requirement_clauses(requirement: str) -> list[str]:
    """Split a requirement into clauses without cutting inside brackets.

    "（短期保多轮连续，长期存画像）" must stay one clause: splitting it produced
    fragments like "保多轮连续）" whose coverage score is meaningless.
    """

    body = _REQUIREMENT_PREFIX.sub("", requirement.strip())
    parts: list[str] = []
    buffer: list[str] = []
    depth = 0
    index = 0
    while index < len(body):
        char = body[index]
        if char in "（(":
            depth += 1
        elif char in "）)":
            depth = max(0, depth - 1)
        if depth == 0 and body.startswith(("以及", "并且", "同时"), index):
            parts.append("".join(buffer))
            buffer = []
            index += 2
            continue
        if depth == 0 and char in "，、；;。":
            parts.append("".join(buffer))
            buffer = []
            index += 1
            continue
        buffer.append(char)
        index += 1
    parts.append("".join(buffer))
    return [part.strip() for part in parts if len(part.strip()) >= 6]


def question_core(question: str) -> str:
    """The interrogative part of a question, dropping long_noisy background.

    Only used for reporting; scope checks read the whole question, because the
    content of a colloquial question often sits in the clause *before* the
    interrogative one ("那个把句子概率拆成…的式子，为啥还是没法直接算？").
    """

    segments = [seg for seg in re.split(r"[，。；！]", question) if seg.strip()]
    cued = [seg for seg in segments if any(cue in seg for cue in _QUESTION_CUES)]
    return "，".join(cued) if cued else (segments[-1] if segments else question)


def _anchor_view(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    anchors: dict[str, dict[str, Any]] = {}
    for item in items:
        anchor = anchors.setdefault(item["anchor_id"], {
            "anchor_id": item["anchor_id"],
            "domain": item["domain"],
            "split": item["split"],
            "topic": (item.get("adjudication") or {}).get("topic", ""),
            "answer_requirements": item["answer_requirements"],
            "targets": item["relevant_targets"],
            "hard_negatives": item["hard_negatives"],
            "questions": {},
        })
        anchor["questions"][item["query_style"]] = {
            "query_id": item["query_id"], "question": item["question"],
        }
    return anchors


def _adjudication_key(finding: dict[str, Any]) -> tuple[str, str, str]:
    return (finding["check"], finding.get("anchor_id", ""),
            str(finding.get("clause") or finding.get("query_id") or ""))


def audit(
    payload: dict[str, Any], documents: dict[str, str],
    adjudication: dict[str, Any] | None = None,
) -> dict[str, Any]:
    anchors = _anchor_view(payload["items"])

    # Vocabulary the knowledge base itself uses: the yardstick for judging
    # whether a requirement names something the evidence never mentions.
    evidence_vocabulary: set[str] = set()
    for document in documents.values():
        evidence_vocabulary |= terms(document)

    scope_terms: dict[str, set[str]] = {}
    anchor_profile: dict[str, set[str]] = {}
    for anchor_id, anchor in anchors.items():
        scope = terms(anchor["topic"])
        for requirement in anchor["answer_requirements"]:
            scope |= terms(requirement)
        for target in anchor["targets"]:
            scope |= terms(target["evidence_excerpt"])
        scope_terms[anchor_id] = scope
        profile = terms(anchor["topic"])
        for requirement in anchor["answer_requirements"]:
            profile |= terms(requirement)
        anchor_profile[anchor_id] = profile

    # Drift is judged on *distinctive* vocabulary drawn from the knowledge base
    # (topics and evidence), never from the requirement prose I wrote myself —
    # otherwise my own connectives become "content" and every question drifts.
    # Rarity does the rest: a term shared by many anchors is not a handle.
    anchor_document_frequency: Counter = Counter()
    for anchor_id, anchor in anchors.items():
        evidence_side = terms(anchor["topic"])
        for target in anchor["targets"]:
            evidence_side |= terms(target["evidence_excerpt"])
        for term in evidence_side:
            anchor_document_frequency[term] += 1
    max_anchor_df = max(1, int(len(anchors) * CONTENT_TERM_MAX_ANCHOR_SHARE))
    content_vocabulary = {
        term for term, count in anchor_document_frequency.items()
        if count <= max_anchor_df and is_content_term(term)
    }
    inverse_document_frequency = {
        term: 1.0 / anchor_document_frequency[term] for term in content_vocabulary
    }

    def weighted(shared: Iterable[str]) -> float:
        return round(sum(inverse_document_frequency[term] for term in shared), 4)

    findings: list[dict[str, Any]] = []

    def record(kind: str, severity: str, anchor_id: str, detail: dict[str, Any]) -> None:
        findings.append({"check": kind, "severity": severity,
                         "anchor_id": anchor_id, **detail})

    for anchor_id, anchor in anchors.items():
        gold = next(target for target in anchor["targets"]
                    if int(target["relevance_grade"]) == 3)
        excerpt_terms = terms(gold["evidence_excerpt"])
        document_terms = terms(documents.get(gold["chunk_id"], ""))

        # --- requirement grounding -------------------------------------
        for requirement in anchor["answer_requirements"]:
            for clause in requirement_clauses(requirement):
                in_excerpt = coverage(clause, excerpt_terms,
                                      vocabulary=evidence_vocabulary)
                in_document = coverage(clause, document_terms,
                                       vocabulary=evidence_vocabulary)
                if in_excerpt >= CLAUSE_GROUNDED:
                    continue
                severity = ("excerpt_too_narrow" if in_document >= CLAUSE_GROUNDED
                            else "weak" if in_document >= CLAUSE_WEAK
                            else "ungrounded")
                record("requirement_grounding",
                       "info" if severity == "excerpt_too_narrow" else
                       "warn" if severity == "weak" else "fail",
                       anchor_id, {
                           "clause": clause,
                           "coverage_in_excerpt": round(in_excerpt, 3),
                           "coverage_in_chunk": round(in_document, 3),
                           "verdict": severity,
                           "gold_chunk_id": gold["chunk_id"],
                       })

        # --- hard negative sanity --------------------------------------
        requirement_text = " ".join(anchor["answer_requirements"])
        gold_requirement_coverage = coverage(requirement_text, document_terms)
        for negative in anchor["hard_negatives"]:
            negative_terms = terms(documents.get(negative["chunk_id"], ""))
            negative_coverage = coverage(requirement_text, negative_terms)
            ratio = (negative_coverage / gold_requirement_coverage
                     if gold_requirement_coverage else 0.0)
            if ratio >= NEGATIVE_REVIEW_RATIO:
                record("hard_negative_sanity", "warn", anchor_id, {
                    "chunk_id": negative["chunk_id"],
                    "negative_requirement_coverage": round(negative_coverage, 3),
                    "gold_requirement_coverage": round(gold_requirement_coverage, 3),
                    "ratio": round(ratio, 3),
                    "reason": negative["reason"],
                })

        # --- per-question checks ---------------------------------------
        lengths = {style: len(entry["question"])
                   for style, entry in anchor["questions"].items()}
        for style, entry in sorted(anchor["questions"].items()):
            question = entry["question"]
            question_terms = terms(question)
            # Anchoring asks "does this question touch its knowledge point at
            # all", so it reads the anchor's whole scope; drift asks "does it
            # reach for a *distinctive* term that belongs elsewhere".
            core_terms = question_terms & content_vocabulary
            out_of_scope = sorted(
                term for term in core_terms - scope_terms[anchor_id]
                if anchor_document_frequency[term] <= DRIFT_TERM_MAX_ANCHOR_DF
            )
            in_scope_hits = sorted(question_terms & scope_terms[anchor_id])

            if out_of_scope:
                record("question_scope",
                       "info" if style == "long_noisy" else "warn",
                       anchor_id, {
                           "query_id": entry["query_id"], "query_style": style,
                           "question": question,
                           "out_of_scope_terms": out_of_scope,
                           "out_of_scope_owners": sorted({
                               other_id for term in out_of_scope
                               for other_id, values in scope_terms.items()
                               if term in values and other_id != anchor_id
                           })[:5],
                           "in_scope_terms": in_scope_hits[:8],
                       })
            if not in_scope_hits:
                # A term-free question is the point of ``implicit_oral``; for a
                # standard or natural question it means the knowledge point is
                # not identifiable from the wording at all.
                record("question_anchoring",
                       "warn" if style == "implicit_oral" else "fail",
                       anchor_id, {
                           "query_id": entry["query_id"], "query_style": style,
                           "question": question,
                           "interrogative_core": question_core(question),
                           "reason": "问法与本锚点的主题/答案要求/证据没有任何共同实词",
                       })

            scores = sorted(
                ((weighted(core_terms & profile), other_id)
                 for other_id, profile in anchor_profile.items()),
                key=lambda pair: (-pair[0], pair[1]),
            )
            own = next(score for score, other_id in scores if other_id == anchor_id)
            best_score, best_id = scores[0]
            if best_id != anchor_id and best_score - own >= DISCRIMINATIVE_MARGIN:
                record("discriminativeness", "warn", anchor_id, {
                    "query_id": entry["query_id"], "query_style": style,
                    "question": question,
                    "own_score": own,
                    "best_other_anchor": best_id,
                    "best_other_score": best_score,
                    "best_other_topic": anchors[best_id]["topic"],
                })

            leak = max(coverage(requirement, terms(question))
                       for requirement in anchor["answer_requirements"])
            if leak >= ANSWER_LEAK_COVERAGE:
                record("style_contract", "warn", anchor_id, {
                    "query_id": entry["query_id"], "query_style": style,
                    "question": question, "issue": "answer_leak_soft",
                    "requirement_coverage_by_question": round(leak, 3),
                })
            if style == "implicit_oral":
                # Reported for reading only: naming the same terms as the
                # standard form does not by itself make a question un-oral.
                standard_hits = len(
                    terms(anchor["questions"].get("standard", {}).get("question", ""))
                    & content_vocabulary & scope_terms[anchor_id]
                )
                own_hits = len(set(in_scope_hits) & content_vocabulary)
                if standard_hits and own_hits >= standard_hits:
                    record("style_contract", "info", anchor_id, {
                        "query_id": entry["query_id"], "query_style": style,
                        "question": question,
                        "issue": "implicit_oral_as_explicit_as_standard",
                        "own_scope_terms": own_hits,
                        "standard_scope_terms": standard_hits,
                    })
                if not any(marker in question for marker in _ORAL_MARKERS):
                    # Lexicon heuristic with known incompleteness: colloquial
                    # Chinese cannot be enumerated, so a miss is a prompt to
                    # read the line, never a defect on its own.
                    record("style_contract", "info", anchor_id, {
                        "query_id": entry["query_id"], "query_style": style,
                        "question": question,
                        "issue": "implicit_oral_marker_not_matched",
                    })
            if style == "long_noisy" and lengths.get(style, 0) <= max(
                value for other, value in lengths.items() if other != style
            ):
                record("style_contract", "warn", anchor_id, {
                    "query_id": entry["query_id"], "query_style": style,
                    "question": question, "issue": "long_noisy_not_longest",
                    "lengths": lengths,
                })
            if style == "standard" and not in_scope_hits:
                record("style_contract", "warn", anchor_id, {
                    "query_id": entry["query_id"], "query_style": style,
                    "question": question, "issue": "standard_without_scope_term",
                })

    reviewed = {
        (entry["check"], entry["anchor_id"],
         str(entry.get("clause") or entry.get("query_id") or "")): entry
        for entry in (adjudication or {}).get("entries", [])
    }
    for finding in findings:
        entry = reviewed.get(_adjudication_key(finding))
        if entry is not None:
            finding["acknowledged"] = True
            finding["adjudication_verdict"] = entry["verdict"]
            finding["adjudication_reason"] = entry["reason"]
    counts = Counter((finding["check"], finding["severity"]) for finding in findings)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset_id": payload.get("dataset_id"),
        "thresholds": {
            "clause_grounded": CLAUSE_GROUNDED,
            "clause_weak": CLAUSE_WEAK,
            "answer_leak_coverage": ANSWER_LEAK_COVERAGE,
            "content_term_max_anchor_share": CONTENT_TERM_MAX_ANCHOR_SHARE,
            "drift_term_max_anchor_df": DRIFT_TERM_MAX_ANCHOR_DF,
            "negative_review_ratio": NEGATIVE_REVIEW_RATIO,
            "discriminative_margin": DISCRIMINATIVE_MARGIN,
        },
        "summary": {
            "anchors": len(anchors),
            "questions": sum(len(anchor["questions"]) for anchor in anchors.values()),
            "findings": len(findings),
            "by_check_severity": {f"{check}:{severity}": count
                                  for (check, severity), count in sorted(counts.items())},
            "acknowledged": sum(1 for finding in findings
                                if finding.get("acknowledged")),
            "unmatched_adjudications": sorted(
                "|".join(key) for key in
                set(reviewed) - {_adjudication_key(row) for row in findings}
            ),
            "blocking": sum(1 for finding in findings
                            if finding["severity"] == "fail"
                            and not finding.get("acknowledged")),
        },
        "findings": sorted(
            findings,
            key=lambda finding: ({"fail": 0, "warn": 1, "info": 2}[finding["severity"]],
                                 finding["check"], finding["anchor_id"],
                                 finding.get("query_id", "")),
        ),
    }


def render_markdown(report: dict[str, Any], *, source: str, sha: str) -> str:
    summary = report["summary"]
    lines = [
        "# V2-A 语义一致性审计（问题—答案要求—证据，2026-08-26）",
        "",
        "> 全部检查为确定性词面检查 + 数据集内部信号（无 LLM、无网络）。",
        "> `fail` 必须修；`warn` 需人工判读；`long_noisy` 的背景词出现在 scope 之外属设计预期，记为 `info`。",
        "",
        f"- 数据：`{source}`（`{sha}`）",
        f"- 锚点 {summary['anchors']} / 问法 {summary['questions']} / 发现 {summary['findings']} 条"
        f"（阻断级 {summary['blocking']}）",
        f"- 分布：{json.dumps(summary['by_check_severity'], ensure_ascii=False)}",
        "",
        "## 发现明细",
        "",
    ]
    for check in ("question_anchoring", "requirement_grounding", "discriminativeness",
                  "hard_negative_sanity", "style_contract", "question_scope"):
        entries = [row for row in report["findings"] if row["check"] == check]
        if not entries:
            continue
        lines.extend([f"### {check}（{len(entries)}）", ""])
        for row in entries:
            marker = "✔已复核 " if row.get("acknowledged") else ""
            head = f"- {marker}**[{row['severity']}]** `{row['anchor_id']}`"
            if row.get("query_id"):
                head += f" / `{row['query_id']}`（{row.get('query_style', '')}）"
            lines.append(head)
            for key, value in row.items():
                if key in {"check", "severity", "anchor_id", "query_id", "query_style"}:
                    continue
                rendered = (json.dumps(value, ensure_ascii=False)
                            if isinstance(value, (list, dict)) else value)
                lines.append(f"  - {key}: {rendered}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_PUBLIC,
                        help="V2-A dataset to audit (public draft by default)")
    parser.add_argument("--extra-cases", type=Path, action="append", default=[],
                        help="additional split files to audit together, e.g. sealed")
    parser.add_argument("--adjudication", type=Path, default=DEFAULT_ADJUDICATION,
                        help="reviewed findings to mark as acknowledged")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args(argv)

    import chromadb
    from rag_tools import get_collection_name

    cases_path = args.cases.expanduser().resolve()
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    for extra in args.extra_cases:
        other = json.loads(Path(extra).expanduser().resolve().read_text(encoding="utf-8"))
        payload = {**payload, "items": [*payload["items"], *other["items"]]}

    collection = chromadb.PersistentClient(
        path=str(ROOT / "chroma_db")
    ).get_collection(get_collection_name())
    chunk_ids = sorted(
        {target["chunk_id"] for item in payload["items"]
         for target in item["relevant_targets"]}
        | {negative["chunk_id"] for item in payload["items"]
           for negative in item["hard_negatives"]}
    )
    snapshot = collection.get(ids=chunk_ids, include=["documents"])
    documents = {
        str(chunk_id): str(document or "")
        for chunk_id, document in zip(snapshot.get("ids") or [],
                                      snapshot.get("documents") or [])
    }
    adjudication_path = args.adjudication.expanduser().resolve() if args.adjudication else None
    adjudication = (json.loads(adjudication_path.read_text(encoding="utf-8"))
                    if adjudication_path and adjudication_path.exists() else None)
    report = audit(payload, documents, adjudication)
    json_path = args.output_json.expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    args.output_md.expanduser().resolve().write_text(
        render_markdown(report, source=cases_path.name, sha=_sha256_file(cases_path)),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_json": str(json_path),
        "output_md": str(args.output_md.expanduser().resolve()),
        "summary": report["summary"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
