# -*- coding: utf-8 -*-


def test_route_eval_set_meets_registered_acceptance_thresholds():
    from eval_rag_routes import evaluate

    report = evaluate()
    assert report["count"] == 120
    assert report["application_state_recall"] == 1.0, report["failures"][:10]
    assert report["macro_f1"] >= 0.95, report["failures"][:10]
