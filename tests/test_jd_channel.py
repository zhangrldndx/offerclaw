# -*- coding: utf-8 -*-
"""JD data boundaries: business bindings stay direct and candidate files stay offline."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_candidate_jd_file_is_not_a_default_rag_source(tmp_path):
    from rag_ingest import DEFAULT_FILES, ingest_file

    assert "jd_candidates.md" not in {path for path, _ in DEFAULT_FILES}
    assert "jd" not in {source_type for _, source_type in DEFAULT_FILES}

    source = tmp_path / "manual_candidate.md"
    source.write_text("# Candidate JD\n\n岗位要求：Python", encoding="utf-8")

    class Collection:
        def __getattr__(self, name):
            raise AssertionError(f"excluded source must not access collection.{name}")

    out = ingest_file(str(source), Collection(), source_type="jd")
    assert out == {
        "file": "manual_candidate.md",
        "status": "excluded_from_rag",
        "chunks": 0,
        "tokens": 0,
    }


def test_candidate_and_business_jds_are_rejected_as_vector_evidence():
    from rag_source_policy import evidence_allowed, rag_source_excluded

    for source_type, source in (
        ("jd", "candidate.md"),
        ("application_jd", "jdv_123.md"),
        ("doc", "jd_candidates.md"),
    ):
        assert rag_source_excluded(source_type, source)
        for route in ("reference_kb", "project_memory", "unknown"):
            assert not evidence_allowed(route, source_type, "internal", source)


def test_retrieval_profile_has_no_candidate_jd_channel():
    import rag_gate
    from rag_retrieval_trace import RetrievalProfile, retrieval_profile_registry

    assert "enable_jd_channel" not in RetrievalProfile.__dataclass_fields__
    assert not hasattr(rag_gate, "_JD_SIGNAL")
    assert not hasattr(rag_gate, "_jd_channel_wanted")
    for profile in retrieval_profile_registry().values():
        assert "enable_jd_channel" not in profile.to_dict()


def test_direct_application_jd_routes_remain_registered():
    from rag_route_registry import get_route_definition

    for operation in (
        "get_bound_jd", "search_bound_jd", "get_match_snapshot", "compare_versions",
    ):
        route = get_route_definition("application_jd", operation)
        assert route is not None
        assert route.personal is True
        assert "application" in route.required_entities


def test_legacy_candidate_jd_chunks_can_be_purged():
    from rag_ingest import purge_non_rag_sources

    class Collection:
        rows = {
            "legacy_type": {"source_type": "jd", "source": "other.md"},
            "legacy_name": {"source_type": "doc", "source": "jd_candidates.md"},
            "business_jd": {"source_type": "application_jd", "source": "jdv_123.md"},
            "keep": {"source_type": "doc", "source": "guide.md"},
        }

        def get(self, *, where):
            key, value = next(iter(where.items()))
            return {"ids": [item_id for item_id, meta in self.rows.items()
                            if meta.get(key) == value]}

        def delete(self, *, ids):
            for item_id in ids:
                self.rows.pop(item_id)

    collection = Collection()
    assert purge_non_rag_sources(collection) == 3
    assert collection.rows == {
        "keep": {"source_type": "doc", "source": "guide.md"},
    }
