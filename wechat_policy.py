# -*- coding: utf-8 -*-
"""Auditable WeChat exposure policy for every OfferClaw read and action capability."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from action_capabilities import ACTION_CAPABILITIES
from rag_route_registry import READ_SOURCE_REGISTRY


POLICY_VERSION = "offerclaw-wechat-policy-v1"


@dataclass(frozen=True)
class WeChatCapabilityPolicy:
    capability_id: str
    mode: str
    privacy: str
    formatter: str
    model_allowed: bool = False
    confirmation_required: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


_READ_MODES = {
    "product_help": ("direct_read", "public"),
    "system_diagnostics": ("direct_read", "personal"),
    "application_state": ("direct_read", "sensitive"),
    "application_jd": ("direct_read", "sensitive"),
    "profile_plan": ("direct_read", "sensitive"),
    "reflection_memory": ("direct_read", "sensitive"),
    "project_memory": ("local_evidence", "sensitive"),
    "application_experience": ("local_evidence", "sensitive"),
    "resume_rules": ("local_evidence", "sensitive"),
    "reference_kb": ("local_evidence", "personal"),
    "paper_kb": ("local_evidence", "personal"),
}

READ_POLICIES = {}
for item in READ_SOURCE_REGISTRY:
    mode, privacy = _READ_MODES[item.source]
    READ_POLICIES[item.key] = WeChatCapabilityPolicy(
        item.key, mode, privacy,
        "deterministic_evidence" if mode == "local_evidence" else "deterministic_structured",
        model_allowed=False,
    )


_CONFIRMED_ACTIONS = {
    "application.update_status",
    "daily_log.create",
    "profile.approve_suggestion",
    "profile.reject_suggestion",
}
_MODEL_BLOCKED_ACTIONS = {
    "plan.generate", "plan.update", "plan.approve", "plan.reject",
    "resume.generate_project", "resume.generate_full", "resume.update",
    "resume.approve", "resume.reject", "reflection.generate",
}
_UNAVAILABLE_ACTIONS = {"application.delete"}


def _action_policy(capability_id: str) -> WeChatCapabilityPolicy:
    if capability_id in _CONFIRMED_ACTIONS:
        return WeChatCapabilityPolicy(
            capability_id, "preview_confirm", "sensitive", "field_preview",
            model_allowed=False, confirmation_required=True,
        )
    if capability_id in _MODEL_BLOCKED_ACTIONS:
        return WeChatCapabilityPolicy(
            capability_id, "model_blocked", "sensitive", "blocked_reason"
        )
    if capability_id in _UNAVAILABLE_ACTIONS:
        return WeChatCapabilityPolicy(
            capability_id, "unavailable", "sensitive", "unavailable_reason"
        )
    return WeChatCapabilityPolicy(
        capability_id, "safe_guidance", "sensitive", "action_guidance"
    )


ACTION_POLICIES = {
    item.capability_id: _action_policy(item.capability_id)
    for item in ACTION_CAPABILITIES
}


def policy_report() -> dict:
    read_ids = {item.key for item in READ_SOURCE_REGISTRY}
    action_ids = {item.capability_id for item in ACTION_CAPABILITIES}
    missing_reads = sorted(read_ids - set(READ_POLICIES))
    missing_actions = sorted(action_ids - set(ACTION_POLICIES))
    extra_reads = sorted(set(READ_POLICIES) - read_ids)
    extra_actions = sorted(set(ACTION_POLICIES) - action_ids)
    return {
        "version": POLICY_VERSION,
        "status": "ok" if not any((missing_reads, missing_actions, extra_reads, extra_actions)) else "error",
        "read_count": len(READ_POLICIES),
        "action_count": len(ACTION_POLICIES),
        "missing_reads": missing_reads,
        "missing_actions": missing_actions,
        "extra_reads": extra_reads,
        "extra_actions": extra_actions,
    }


__all__ = [
    "ACTION_POLICIES", "POLICY_VERSION", "READ_POLICIES",
    "WeChatCapabilityPolicy", "policy_report",
]
