# -*- coding: utf-8 -*-
"""Machine-readable domain enums and exact legacy adapters."""
from __future__ import annotations

from enum import StrEnum


class MatchStatusCode(StrEnum):
    SUITABLE = "suitable"
    STRETCH = "stretch"
    NOT_RECOMMENDED = "not_recommended"
    UNKNOWN = "unknown"


MATCH_STATUS_LABELS = {
    MatchStatusCode.SUITABLE: "当前适合投递",
    MatchStatusCode.STRETCH: "中长期可转向",
    MatchStatusCode.NOT_RECOMMENDED: "当前暂不建议投递",
    MatchStatusCode.UNKNOWN: "信息不足，建议补充后再判断",
}
_LEGACY_MATCH_STATUS = {
    "当前适合投递": MatchStatusCode.SUITABLE,
    "适合": MatchStatusCode.SUITABLE,
    "中长期可转向": MatchStatusCode.STRETCH,
    "当前暂不建议投递": MatchStatusCode.NOT_RECOMMENDED,
    "暂不建议投递": MatchStatusCode.NOT_RECOMMENDED,
    "不适合": MatchStatusCode.NOT_RECOMMENDED,
    "信息不足，建议补充后再判断": MatchStatusCode.UNKNOWN,
}


def match_status_code(value: object) -> MatchStatusCode:
    if isinstance(value, MatchStatusCode):
        return value
    raw = str(value or "").strip()
    try:
        return MatchStatusCode(raw)
    except ValueError:
        return _LEGACY_MATCH_STATUS.get(raw, MatchStatusCode.UNKNOWN)


class ApplicationStatusCode(StrEnum):
    EVALUATED = "evaluated"
    PREPARING = "preparing"
    SKIPPED = "skipped"
    APPLIED = "applied"
    WAITING = "waiting"
    INTERVIEWING = "interviewing"
    OFFERED = "offered"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


APPLICATION_STATUS_LABELS = {
    ApplicationStatusCode.EVALUATED: "已评估",
    ApplicationStatusCode.PREPARING: "准备投递",
    ApplicationStatusCode.SKIPPED: "不投递",
    ApplicationStatusCode.APPLIED: "已投递",
    ApplicationStatusCode.WAITING: "等待反馈",
    ApplicationStatusCode.INTERVIEWING: "面试中",
    ApplicationStatusCode.OFFERED: "已 Offer",
    ApplicationStatusCode.REJECTED: "已拒绝",
    ApplicationStatusCode.WITHDRAWN: "主动放弃",
}
_APPLICATION_BY_LABEL = {label: code for code, label in APPLICATION_STATUS_LABELS.items()}


def application_status_code(value: object) -> ApplicationStatusCode:
    if isinstance(value, ApplicationStatusCode):
        return value
    raw = str(value or "").strip()
    try:
        return ApplicationStatusCode(raw)
    except ValueError as exc:
        if raw in _APPLICATION_BY_LABEL:
            return _APPLICATION_BY_LABEL[raw]
        raise ValueError(f"未知投递状态: {raw}") from exc


class HardGateStatusCode(StrEnum):
    MET = "met"
    UNMET = "unmet"
    UNKNOWN = "unknown"


class SoftConditionStatusCode(StrEnum):
    MET = "met"
    PARTIAL = "partial"
    UNMET = "unmet"
    UNKNOWN = "unknown"


_HARD_STATUS = {"✓": HardGateStatusCode.MET, "✗": HardGateStatusCode.UNMET,
                "?": HardGateStatusCode.UNKNOWN}
_SOFT_STATUS = {"命中": SoftConditionStatusCode.MET,
                "部分命中": SoftConditionStatusCode.PARTIAL,
                "未命中": SoftConditionStatusCode.UNMET,
                "?": SoftConditionStatusCode.UNKNOWN}


def requirement_status_code(value: object) -> str:
    """Exact compatibility adapter for legacy hard/soft display labels."""
    raw = str(value or "").strip()
    if raw in _HARD_STATUS:
        return _HARD_STATUS[raw].value
    if raw in _SOFT_STATUS:
        return _SOFT_STATUS[raw].value
    raise ValueError(f"未知条件判断状态: {raw}")


__all__ = [
    "ApplicationStatusCode", "APPLICATION_STATUS_LABELS", "HardGateStatusCode",
    "MatchStatusCode", "MATCH_STATUS_LABELS", "SoftConditionStatusCode",
    "application_status_code", "match_status_code", "requirement_status_code",
]
