# -*- coding: utf-8 -*-
"""Grade whether a chunk *contains the answer*, as opposed to sharing its topic.

Every ranking intervention tried on 2026-08-27 failed the same way, and the
diagnosis was the same each time: ``bge-reranker-base`` scores topical
relatedness.  Its evidence, in one line -- on the 20 Dev-New queries where the
gold was in the pool and lost, the winning chunk covered the question's answer
requirement at 0.07-0.44 against the gold's 0.77-0.93, while being scored above
0.99.  Nothing downstream of a scorer like that can be fixed by giving it
better candidates, retraining its top layers, or breaking its ties.

The grade is deliberately ordinal with a *named* middle:

    0  片段与问题无关
    1  提到了这个话题，但没有回答所需的事实      <- the exact failure mode
    2  含部分事实，不足以完整回答
    3  含回答所需的事实

Grade 1 is the whole point.  A topical scorer cannot express it, which is why
"adjacent topic the corpus does not cover" both wins reranking and slips past
a confidence-based gate.

The grade alone cannot decide what to *do*, because it conflates two questions
that a false-premise query separates.  Asked "is MVCC used to deduplicate a
vector store?", a chunk explaining MVCC as MySQL concurrency control contains
everything needed -- to *refute* the premise.  Grading it 3 is right; treating
it as a case for abstention is not, and a guard built that way penalises the
system for correcting the user.  The judge therefore emits a structured
premise assessment; code, rather than the model, deterministically derives:

    entails          证据支持问题的前提      -> answer
    contradicts      证据明确反驳问题的前提  -> correct the premise, then answer
    not_established  既不支持也不反驳        -> abstain

Only ``not_established`` (or a low grade) is an abstention case.  A gate keyed
on the grade alone would train itself to go silent whenever a premise is false,
which is the opposite of the behaviour worth having.

Two properties this file must keep:

* **No label leakage.**  The prompt sees the question and the chunk, never the
  gold id, never ``answer_requirements`` (which were authored *from* the gold
  and would leak the answer straight into the feature).
* **Deterministic and cached.**  temperature 0 and an on-disk cache keyed by
  (model, question, chunk); a judge that drifts between runs makes every A/B
  built on it unreproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
CACHE_PATH = ROOT / ".offerclaw" / "answerability_cache.json"
SCHEMA = "answerability-v4"  # v4 derives relation from a strict premise contract
MAX_CHUNK_CHARS = 2000
GRADES = (0, 1, 2, 3)
RELATIONS = ("entails", "contradicts", "not_established")
QUESTION_FORMS = ("polar", "open")
DIRECT_ANSWERS = (
    "proposition_true", "proposition_false", "unknown", "not_applicable",
)
PREMISE_STATUSES = ("supported", "refuted", "not_established", "none")
# The action a gate should take, given (relation, grade).  Written out rather
# than derived so the policy is reviewable in one place.
ACTIONS = {"entails": "answer", "contradicts": "correct_premise",
           "not_established": "abstain"}

_PROMPT = """你是检索结果的评判者。判断【资料片段】是否包含回答【问题】所需的事实。

严格规则：
1. 只看片段本身。不要用你自己的知识补全，不要推测片段之外的内容。
2. "提到了话题"不等于"回答了问题"。片段只是谈论相关主题、列出小节标题、
   或给出练习题/目录/链接，都不算包含答案。
3. 分开判断“事实覆盖度”和“问题中命题/前提是否成立”。不要因为片段能回答问题，
   就把问题中的命题判为成立；完整答案可能是“不成立”。
4. 严格按指定枚举只输出一个 JSON 对象，不要输出 relation 字段；relation 由程序计算。
5. 【原始用户问题】是判断命题/前提真假的唯一来源。【本路由检索问题】只界定
   reference 路由负责检索的事实范围，不能替换、弱化或改写原始问题中的命题。

grade（片段含有多少本路由事实范围所需的事实）：
0 = 与问题无关
1 = 提到了这个话题，但不含回答所需的事实
2 = 含部分事实，不足以完整回答
3 = 含回答所需的事实

question_form：
polar = “是否/是不是/能否/可否/有无/X负责Y吗”等真假询问
open  = “什么/为何/如何”等开放问句

本题 question_form 硬约束：{question_form_hint}

direct_answer（只对 polar 使用）：
proposition_true  = 问号前的待检验命题成立
proposition_false = 问号前的待检验命题不成立
unknown           = 片段不足以判断命题真假
not_applicable    = 仅用于 open
这里判断的是“命题真/假”，不是对含否定词问句口语回答“是/不是”。

premise_status：
supported       = 片段明确支持问题中的命题或预设
refuted         = 片段明确反驳问题中的命题或预设
not_established = 片段既不支持也不反驳；只谈相邻话题也属于此项
none            = open 问句没有需要单独检验的预设；仅用于 open

一致性要求：
- polar 只允许 proposition_true+supported、proposition_false+refuted、
  unknown+not_established 三种组合。
- open 的 direct_answer 必须是 not_applicable；premise_status 可按片段取值。
- 没有证据不能判 refuted；一般类别的事实不能推出某个特定子类也具有该事实。
- grade 只表示事实覆盖度。grade=3 不自动等于 supported。

易错示例：
1. 问“闭源 LLM 是否依靠压缩加速”，片段只说压缩可提升一般 LLM 推理速度，
   没说闭源模型或“依靠”关系：unknown + not_established，不是 supported/refuted。
2. 问“Planner 的职责是执行子任务而非设计路径吗”，片段明确说 Planner 设计路径、
   不执行子任务：proposition_false + refuted；若事实完整，grade 可以是 3。
3. 问“向量数据库与传统数据库相互替代吗”，片段明确说二者互补而非替代：
   proposition_false + refuted；若事实完整，grade 可以是 3。

输出格式：
{{"grade": 0|1|2|3, "question_form": "polar|open", "direct_answer": "proposition_true|proposition_false|unknown|not_applicable", "premise_status": "supported|refuted|not_established|none", "reason": "不超过30字"}}

【原始用户问题（唯一的命题/前提真值来源）】
{question}

【本路由检索问题（只界定事实覆盖范围，不得改写原命题）】
{retrieval_question}

【资料片段】
{chunk}"""
PROMPT_SHA256 = hashlib.sha256(_PROMPT.encode("utf-8")).hexdigest()
# Frozen only for an exact same-protocol stability audit of the 889 Pilot rows
# labelled before the explicit question-form hint was added.  Production never
# selects this template.
LEGACY_PROMPT_V1 = _PROMPT.replace(
    '\n\n本题 question_form 硬约束：{question_form_hint}\n\n', '\n\n',
)
LEGACY_PROMPT_V1_SHA256 = hashlib.sha256(
    LEGACY_PROMPT_V1.encode("utf-8")
).hexdigest()

# --- v5 candidate prompt (opt-in via RAG_ANSWERABILITY_PROMPT=v5) -----------
# One rule, one measured blind spot: questions with an unresolved referent
# ("这俩哪个好" / "上次那个") carried no checkable entity, yet chunks still got
# graded 3/entails -- every genuine false accept the consensus gate let through
# on Final v3 had this shape (or missing-data).  The targeted screen fixed 4/4
# of them with zero damage to corrections (3/3) and golds (8/9, the miss being
# judge sampling noise on an unrelated case).
#
# A second candidate rule -- "grade 3 requires the fact in THIS chunk, not its
# document" -- was screened the same day and ABORTED: it down-graded only 2 of
# the 9 same-source inflated incumbents it targeted.  Same-source inflation
# resists prompt-level fixes; do not re-add wording for it without a new
# mechanism.
_V5_EDITS = [
    ('5. 【原始用户问题】是判断命题/前提真假的唯一来源。',
     '5. 问题里若含未解析指代（如"这俩""那个东西""上次说的"），且问题文本自身\n   没有给出指代对象，则没有任何片段能与之建立对应关系：grade 只能是 0 或 1，\n   polar 时只能 unknown + not_established。注意：若问题自己点名了对象\n   （如"那个结构校验器""咱那个 LangGraph 工作流"），指代已解析，本条不适用。\n6. 【原始用户问题】是判断命题/前提真假的唯一来源。'),
    ('proposition_false + refuted；若事实完整，grade 可以是 3。\n\n输出格式',
     'proposition_false + refuted；若事实完整，grade 可以是 3。\n4. 问"这俩哪个更适合我"，问题没有说明"这俩"指什么：任何片段都只能\n   unknown + not_established，grade 不超过 1。\n5. 问"那个改写模块是怎么触发的"，"那个改写模块"已点名对象：按正常规则判，\n   不因带"那个"而降档。\n\n输出格式'),
]
PROMPT_V5 = _PROMPT
for _old, _new in _V5_EDITS:
    assert _old in PROMPT_V5, f"v5 edit anchor missing: {_old[:30]}"
    PROMPT_V5 = PROMPT_V5.replace(_old, _new, 1)
PROMPT_V5_SHA256 = hashlib.sha256(PROMPT_V5.encode("utf-8")).hexdigest()


def active_prompt() -> tuple[str, str]:
    """Return (template, sha) for the prompt the judge should use right now.

    The sha participates in the cache key, so switching prompts switches cache
    namespaces instead of poisoning v4's entries with v5 verdicts.
    """
    # v5 default since the Final v4 verdict (2026-08-30): the quality package
    # that includes it beat the v4-prompt default on every ranking metric with
    # true false-accepts 1/40.  Honest cost, measured on the same blind set:
    # one contextual-referent colloquial positive ("题里那三个地方...") was
    # downgraded by the no-referent rule -- ~1/80 misfire rate.  "v4" opts out.
    choice = os.environ.get("RAG_ANSWERABILITY_PROMPT", "").strip().lower()
    if choice == "v4":
        return _PROMPT, PROMPT_SHA256
    return PROMPT_V5, PROMPT_V5_SHA256

_LOCK = threading.Lock()
_CACHE: dict[str, Any] | None = None

# This is deliberately only a high-precision guard.  Less explicit forms such
# as alternative questions remain the judge's responsibility; treating every
# trailing question mark as polar would reject ordinary open questions.
_OBVIOUS_POLAR = re.compile(
    r"(?:是否|是不是|能否|可否|有无|要不要|会不会|应不应该|吗\s*[？?]?\s*$)"
)


class AnswerabilityError(RuntimeError):
    """Raised when the judge cannot be used as a deterministic feature."""


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def mode() -> str:
    """Resolve the unified runtime mode without freezing env at import time.

    The legacy boolean retains its old meaning (teacher reranking).  Invalid
    new values fail closed to ``off`` rather than accidentally loading either
    a local checkpoint or a paid judge.
    """

    explicit = os.environ.get("RAG_ANSWERABILITY_MODE", "").strip().lower()
    if explicit:
        return explicit if explicit in {"off", "student", "balanced", "teacher"} else "off"
    # Default ON since the Final v3 verdict (2026-08-29): teacher reranking
    # improved every ranking metric of a fresh blind set at zero false-accept
    # cost.  ``RAG_ANSWERABILITY=0`` is the explicit opt-out; an *unset*
    # variable now means production default, so harnesses that need the judge
    # off must say "0" rather than deleting the variable -- deleting it used to
    # mean "off" and silently means "on" after this flip.
    from rag_mode import mode_env
    raw = mode_env("RAG_ANSWERABILITY")        # 显式 env 优先,fast 模式预设补缺省
    if raw is None or not raw.strip():
        return "teacher"
    return "teacher" if _truthy(raw) else "off"


def enabled() -> bool:
    return mode() != "off"


def resolve_model(model: str | None = None) -> str:
    """Resolve the exact requested judge model before cache lookup.

    An empty model name is not valid cache lineage: ``rag_gate._chat`` would
    later resolve it from ``.env.local``, so changing provider could otherwise
    reuse a verdict produced by a different model under the same empty key.
    """
    from day1_api_starter import get_llm_config, load_local_env

    load_local_env()
    explicit = (model or os.environ.get("RAG_ANSWERABILITY_MODEL", "")).strip()
    if explicit:
        return explicit
    cfg = get_llm_config()
    if cfg.get("is_zhipu"):
        return str(cfg.get("model") or "").strip()
    return (
        os.environ.get("RAG_SYNTH_MODEL", "").strip()
        or os.environ.get("LLM_MODEL", "").strip()
        or str(cfg.get("model") or "").strip()
    )


def cache_key(model: str, question: str, chunk: str,
              retrieval_question: str | None = None,
              variant: str = "") -> str:
    """``variant`` exists for consensus voting and must reach the hash.

    A vote that shares the standard key would be answered from the first
    vote's cache entry, so a "2-of-3 panel" would always be unanimous by
    construction -- consensus as decoration, the same silent-no-op shape as a
    dead env knob.  Salting the key makes each extra vote a real independent
    sample (cached thereafter under its own identity).
    """
    route_scope = retrieval_question or question
    # active_prompt() so a v5 experiment writes to its own cache namespace
    # instead of poisoning v4 entries (and vice versa).
    payload = json.dumps(
        [SCHEMA, active_prompt()[1], model, question, route_scope,
         chunk[:MAX_CHUNK_CHARS]] + ([variant] if variant else []),
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cache() -> dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        if CACHE_PATH.is_file():
            try:
                _CACHE = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                _CACHE = {}
        else:
            _CACHE = {}
    return _CACHE


def flush_cache() -> None:
    with _LOCK:
        if _CACHE is None:
            return
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps(_CACHE, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8")


def _derive_relation(grade: int, question_form: str, direct_answer: str,
                     premise_status: str) -> str | None:
    """Derive relation from the validated v4 contract, never model prose."""
    if question_form == "polar":
        return {
            ("proposition_true", "supported"): "entails",
            ("proposition_false", "refuted"): "contradicts",
            ("unknown", "not_established"): "not_established",
        }.get((direct_answer, premise_status))
    if question_form != "open" or direct_answer != "not_applicable":
        return None
    if premise_status == "supported":
        return "entails"
    if premise_status == "refuted":
        return "contradicts"
    if premise_status == "not_established":
        return "not_established"
    if premise_status == "none":
        # ``relation`` is retained as the stable downstream policy interface.
        # An answerable open question has no proposition to contradict, so a
        # full answer follows the ordinary answer branch; partial/empty
        # evidence remains an abstention case.
        return "entails" if grade == 3 else "not_established"
    return None


def parse_grade(text: str | None, *, question: str = "") -> dict[str, Any] | None:
    """Parse v4 and derive relation, failing closed on any inconsistency.

    A judge that silently returns prose, or a grade of 7, must not be coerced
    into a number -- the caller has to be able to tell "the model said 1" from
    "the model failed", because those mean opposite things for a gate.
    """
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    grade = payload.get("grade")
    if isinstance(grade, bool) or not isinstance(grade, int) or grade not in GRADES:
        return None
    question_form = payload.get("question_form")
    direct_answer = payload.get("direct_answer")
    premise_status = payload.get("premise_status")
    if question_form not in QUESTION_FORMS:
        return None
    if direct_answer not in DIRECT_ANSWERS:
        return None
    if premise_status not in PREMISE_STATUSES:
        return None
    if question and _OBVIOUS_POLAR.search(question) and question_form != "polar":
        return None
    relation = _derive_relation(
        grade, question_form, direct_answer, premise_status,
    )
    if relation is None:
        return None
    return {"grade": grade, "question_form": question_form,
            "direct_answer": direct_answer,
            "premise_status": premise_status, "relation": relation,
            "reason": str(payload.get("reason", ""))[:80]}


# Grade 2 is "partial facts, not enough to answer in full", and it is not a
# licence to speak.  Two adversarial negatives graded 2/entails were compound
# questions the corpus only half-covers, and col-neg-056 was graded 2 precisely
# because its refutation is incomplete -- a caller that forgot to pass the
# threshold would have turned that into a confident correction.  The default is
# therefore the strict side.
MIN_GRADE_TO_ACT = 3


def action(verdict: dict[str, Any] | None, *,
           min_grade: int = MIN_GRADE_TO_ACT) -> str:
    """Map a verdict to what the system should do.

    ``abstain`` is the answer for an unavailable judge, a low grade, and an
    unestablished relation alike -- but only the last of those is a statement
    about the corpus.  Callers that need to tell them apart should read the
    verdict, not this.
    """
    if not verdict or verdict.get("grade", -1) < min_grade:
        return "abstain"
    return ACTIONS.get(verdict.get("relation") or "", "abstain")


def confirm_action(question: str, chunk: str, first_verdict: dict,
                   *, votes: int = 3, grader: Callable | None = None) -> dict:
    """Confirm an actionable verdict with extra independent votes.

    The gate was rejected in its single-verdict form for one measured reason:
    across cold runs, verdicts occasionally flip, and both observed flips moved
    toward acceptance.  So confirmation is deliberately asymmetric -- only an
    *accepting* verdict is ever escalated here; a refusal stands on one vote,
    because the failure mode worth insuring against is speaking falsely, not
    staying silent.

    Vote 1 is the verdict the reranker already produced.  Votes 2..N are real
    resamples (their cache keys carry a vote salt), and a vote that fails to
    parse counts as an abstention rather than being dropped: an outage must
    make the panel more conservative, never smaller.  Majority of actions
    decides; a tie is an abstention.
    """
    call = grader or grade
    actions = [action(first_verdict)]
    for vote in range(2, max(2, votes) + 1):
        try:
            verdict = call(question, chunk, variant=f"vote{vote}")
        except Exception:
            verdict = None
        actions.append(action(verdict))
    counts: dict[str, int] = {}
    for act in actions:
        counts[act] = counts.get(act, 0) + 1
    best = max(counts, key=lambda a: counts[a])
    majority = counts[best] * 2 > len(actions)
    return {
        "action": best if majority else "abstain",
        "votes": actions,
        "unanimous": len(counts) == 1,
    }


def grade(question: str, chunk: str, *, model: str | None = None,
          caller: Callable[..., str | None] | None = None,
          use_cache: bool = True,
          retrieval_question: str | None = None,
          variant: str = "") -> dict[str, Any] | None:
    """Return a v4 verdict, or ``None`` if the judge contract failed.

    ``question`` is always the original user wording and is the sole source of
    premise truth.  ``retrieval_question`` may narrow the reference route's
    factual responsibility, but never replaces that original proposition.

    ``None`` is not a zero.  Callers must decide explicitly what an unavailable
    judge means for them; for a gate, that decision is "refuse", and silently
    substituting 0 would make an outage look like a corpus miss.
    """
    resolved_model = resolve_model(model)
    route_scope = retrieval_question or question
    key = cache_key(
        resolved_model, question, chunk, retrieval_question=route_scope,
        variant=variant,
    )
    if use_cache:
        cached = _load_cache().get(key)
        if cached is not None:
            return dict(cached)

    prompt = active_prompt()[0].format(
        question=question,
        retrieval_question=route_scope,
        chunk=(chunk or "")[:MAX_CHUNK_CHARS],
        question_form_hint=(
            "程序已由显式‘是否/是不是/能否/可否/有无/吗’句式判定为 polar；"
            "必须输出 polar，并只用 polar 允许的 direct_answer/premise_status 组合。"
            if _OBVIOUS_POLAR.search(question)
            else "按上述定义判断；不要因问题较长或口语化而改变问句类型。"
        ),
    )
    if caller is None:
        from rag_gate import _chat as caller  # noqa: PLC0415 - avoids import cycle
    text = caller(
        [{"role": "user", "content": prompt}],
        max_tokens=3000,      # reasoning models share this budget with content
        temperature=0.0,      # a drifting judge is not a feature
        model=resolved_model or None,
    )
    parsed = parse_grade(text, question=question)
    if parsed is None:
        return None
    if use_cache:
        with _LOCK:
            _load_cache()[key] = parsed
    return parsed




# --------------------------------------------------------------------------
# retrieval integration


_TIEBREAK_PROMPT = """下面是若干份资料片段，它们都含有回答【问题】所需的事实。
请选出**最直接、最完整**回答这个问题的那一份。

规则：
1. 只比较这些片段本身，不要用你自己的知识。
2. 谁把问题问到的那件事讲得最具体、最完整，就选谁；只是相关、或者只讲了一部分，都不选。
3. 只输出一个 JSON 对象：{{"best": 序号}}。序号是下面片段前面的数字，不要输出别的。

【问题】
{question}

{candidates}"""

_TIEBREAK_SCHEMA = "answerability-tiebreak-v1"


def tiebreak(question: str, chunks: list, *, model: str | None = None,
             caller: Callable[..., str | None] | None = None,
             use_cache: bool = True) -> int | None:
    """Pick which of several equally-graded chunks best answers the question.

    The four-point scale runs out of resolution exactly where the remaining
    Dev-New misses live: on 23 of 80 queries two or more head candidates are all
    ``grade 3 + entails``, and the cross-encoder score then decides.  A
    deterministic screen of those ties showed the gold covering the question's
    answer requirement 3-5x better than the winner (0.23-0.58 against
    0.00-0.21), so the ties are genuine ranking defects rather than unlabelled
    equivalent evidence -- the pointwise judge simply cannot express "both
    contain the answer, but this one contains more of it".

    This asks one *listwise* question instead of adding a finer grade, because a
    finer grade is another scale to calibrate while a comparison is not.  It
    returns a position in ``chunks`` or ``None``; ``None`` must leave the
    existing order alone, never impose a default winner.
    """
    if len(chunks) < 2:
        return None
    resolved_model = resolve_model(model)
    body = "\n\n".join(
        f"【片段 {index}】\n{(chunk or '')[:MAX_CHUNK_CHARS]}"
        for index, chunk in enumerate(chunks)
    )
    key = hashlib.sha256(json.dumps(
        [_TIEBREAK_SCHEMA, resolved_model, question, body],
        ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    if use_cache:
        cached = _load_cache().get(key)
        if cached is not None:
            return cached.get("best")

    if caller is None:
        from rag_gate import _chat as caller  # noqa: PLC0415 - avoids import cycle
    text = caller(
        [{"role": "user", "content": _TIEBREAK_PROMPT.format(
            question=question, candidates=body)}],
        max_tokens=3000,
        temperature=0.0,
        model=resolved_model or None,
    )
    match = re.search(r'"best"\s*:\s*(\d+)', text or "")
    if not match:
        return None
    best = int(match.group(1))
    if not 0 <= best < len(chunks):
        return None                      # an out-of-range pick is not a pick
    if use_cache:
        with _LOCK:
            _load_cache()[key] = {"best": best}
    return best


# 12 since the Final v3 default flip: the validated production arm judges the
# top 12 of a 28-candidate pool.  6 was the pool-20 era default.
ANSWERABILITY_DEPTH = 12


def rerank_by_answerability(question: str, docs: list, metas: list,
                            dists: list, scores: list,
                            *, depth: int = ANSWERABILITY_DEPTH,
                            grader=None, workers: int | None = None,
                            early_exit: bool = False, stats: dict | None = None,
                            tiebreak_ties: bool = False):
    """Reorder the top ``depth`` candidates by answer containment.

    Ordering key is ``(grade, existing_rerank_score)``: the judge decides which
    chunks actually contain the answer, and the cross-encoder breaks ties inside
    a grade.  There is no threshold to tune, which is why this cannot be
    overfitted to a development set the way a cut point can.

    ``early_exit`` grades the incumbent first and stops when it already scores
    the top grade.  That is exact for rank 1 -- the incumbent holds the highest
    reranker score, so a top grade makes its sort key minimal and nothing below
    can displace it -- but it forfeits any reordering *below* rank 1, which is
    a real cost to R@3/R@5 and to the evidence handed to generation.  It trades
    depth for latency, not correctness for latency.

    ``workers`` follows ``depth`` so that latency is counted in *rounds*, not
    calls: the whole head is one round, and the early-exit tail is a second one.
    A fixed smaller pool would make the early-exit path (1 + depth-1) cost more
    rounds than judging everything at once, which is the opposite of the point,
    and would silently re-introduce that cost the moment ``depth`` is raised.

    Failure is a no-op, never a reordering: a chunk the judge could not grade
    keeps its original position rather than being pushed down as if it had
    scored zero.  An unavailable judge must degrade to today's behaviour, not
    to a silently different ranking.
    """
    if not docs:
        return docs, metas, dists, scores
    from concurrent.futures import ThreadPoolExecutor

    call = grader or grade
    head = list(range(min(depth, len(docs))))
    workers = depth if workers is None else workers
    verdicts: dict[int, dict[str, Any] | None] = {}

    def work(index: int):
        try:
            return index, call(question, docs[index])
        except Exception:
            return index, None

    def run(indexes: list):
        if len(indexes) == 1:
            verdicts.update(dict([work(indexes[0])]))
            return
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for index, verdict in pool.map(work, indexes):
                verdicts[index] = verdict

    # The two are mutually exclusive by construction: early exit stops as soon
    # as the incumbent holds the top grade, so the rest of the head is never
    # graded and a tie can never be observed.  Silently combining them yields a
    # tiebreaker that simply never fires -- measured on Dev-New, four of the six
    # in-pool misses were exactly this shape, with the gold sitting at rank 2-6
    # and no verdict at all.  Depth wins over the latency shortcut.
    early_exit = early_exit and not tiebreak_ties
    if early_exit and len(head) > 1:
        run(head[:1])
        incumbent = verdicts.get(head[0])
        if not (incumbent and incumbent["grade"] == max(GRADES)):
            run(head[1:])
    else:
        run(head)

    graded = sum(1 for v in verdicts.values() if v)
    if stats is not None:
        # ``calls`` alone cannot tell a clean run from a degraded one: a judge
        # that 503s still counts as called and silently returns to the
        # reranker's order.  ``graded`` is what makes that visible afterwards.
        stats["calls"] = len(verdicts)
        stats["graded"] = graded

    if not graded:
        return docs, metas, dists, scores          # judge unavailable: no-op

    def sort_key(index: int):
        verdict = verdicts.get(index)
        # An ungraded row keeps the reranker's own opinion and is placed
        # between grade 1 and grade 2 rather than at the bottom.
        grade_value = verdict["grade"] if verdict else 1.5
        base = scores[index] if index < len(scores) and scores[index] is not None else 0.0
        return (-float(grade_value), -float(base))

    order = sorted(head, key=sort_key) + list(range(len(head), len(docs)))

    if tiebreak_ties:
        # Only the leading block of exact ties is contestable.  Anything the
        # grade already separates stays separated -- the listwise call is a
        # tiebreaker, not a second ranker.
        top = [i for i in order[:len(head)]
               if (verdicts.get(i) or {}).get("grade") == max(GRADES)
               and ACTIONS.get((verdicts.get(i) or {}).get("relation") or "") != "abstain"]
        if len(top) >= 2:
            if stats is not None:
                stats["tiebreak_candidates"] = len(top)
            picked = None
            try:
                picked = tiebreak(question, [docs[i] for i in top])
            except Exception:
                picked = None
            if picked is not None:
                if stats is not None:
                    stats["tiebreak_moved"] = picked != 0
                winner = top[picked]
                order = [winner] + [i for i in order if i != winner]

    take = lambda seq: [seq[i] for i in order] if len(seq) == len(docs) else seq
    if stats is not None:
        # Keyed by the position the chunk ends up in, so a caller can never pair
        # one chunk's grade with another chunk's identity -- the exact mix-up the
        # evidence gate's anchor handling exists to prevent.
        stats["grades"] = {new: verdicts[old]["grade"]
                           for new, old in enumerate(order)
                           if verdicts.get(old)}
        # The relation travels with the grade because grade alone cannot decide
        # what to do: ``not_established`` is an abstention case at any grade,
        # and a consumer that reads only the grade silently re-implements the
        # two-state policy this module exists to replace.
        stats["relations"] = {new: verdicts[old].get("relation")
                              for new, old in enumerate(order)
                              if verdicts.get(old)}
    return take(docs), take(metas), take(dists), take(scores)


def rerank_by_mode(question: str, docs: list, metas: list, dists: list,
                   scores: list, *, depth: int = ANSWERABILITY_DEPTH,
                   early_exit: bool = False, stats: dict | None = None,
                   tiebreak_ties: bool = False):
    """Dispatch the frozen ``off/student/balanced/teacher`` runtime contract.

    ``balanced`` always runs the local student first.  It may call the teacher
    only when a validation-frozen policy for the exact checkpoint recommends
    fallback.  If the student is missing, the function returns the BGE order;
    it never turns a local failure into an unbudgeted LLM request.
    """

    selected = mode()
    if selected == "off":
        if stats is not None:
            stats.update({"applied": False, "mode": "off", "reason": "disabled"})
        return docs, metas, dists, scores
    if selected == "teacher":
        teacher_stats: dict[str, Any] = {}
        result = rerank_by_answerability(
            question, docs, metas, dists, scores, depth=depth,
            early_exit=early_exit, stats=teacher_stats,
            tiebreak_ties=tiebreak_ties,
        )
        if stats is not None:
            stats.update(teacher_stats)
            stats.update({
                "applied": bool(teacher_stats.get("graded")),
                "mode": "teacher", "source": "teacher",
                "fallback_used": False,
            })
        return result

    from rag_answerability_student import rerank_by_student

    student_stats: dict[str, Any] = {}
    result = rerank_by_student(
        question, docs, metas, dists, scores, depth=depth,
        balanced=selected == "balanced", stats=student_stats,
    )
    if selected == "student" or not student_stats.get("fallback_recommended"):
        if stats is not None:
            stats.update(student_stats)
            stats.update({"mode": selected, "fallback_used": False})
        return result

    # The policy exists and marked this exact checkpoint/query uncertain.
    teacher_stats: dict[str, Any] = {}
    judged = rerank_by_answerability(
        question, *result, depth=depth, early_exit=early_exit,
        stats=teacher_stats, tiebreak_ties=tiebreak_ties,
    )
    teacher_available = bool(teacher_stats.get("graded"))
    if stats is not None:
        stats.update(teacher_stats if teacher_available else student_stats)
        stats.update({
            "applied": bool(student_stats.get("applied")),
            "mode": "balanced",
            "source": "teacher" if teacher_available else "student",
            "fallback_used": True,
            "fallback_available": teacher_available,
            "student": student_stats,
        })
    return judged if teacher_available else result


__all__ = ["ANSWERABILITY_DEPTH", "LEGACY_PROMPT_V1", "LEGACY_PROMPT_V1_SHA256", "PROMPT_SHA256", "rerank_by_answerability", "rerank_by_mode", "ACTIONS", "AnswerabilityError", "DIRECT_ANSWERS", "GRADES",
           "MIN_GRADE_TO_ACT", "PREMISE_STATUSES", "QUESTION_FORMS",
           "RELATIONS", "action",
           "cache_key", "enabled", "flush_cache", "grade", "mode", "parse_grade",
           "resolve_model"]
