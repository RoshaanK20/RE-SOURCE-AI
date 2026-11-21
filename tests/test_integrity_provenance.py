"""Fast integrity regression tests; full real-data evaluation is validation/real_data_pilot.py."""
from types import SimpleNamespace
from resource_ai.verify import METHOD_CHANGED, NO_EVIDENCE, verify


def test_reject_fabricated_prediction_with_matching_score():
    X = {"seq_digest": "X_test"}
    y = {"seq_digest": "Y_test"}
    base = {"events": [
        {"i": 0, "depth": 0, "type": "fit", "cls": "RF", "obj": "RF#1", "params": {},
         "X": {"shape": [3, 2], "set_digest": "train_X"},
         "y": {"set_digest": "train_y"}, "xy_pairs": "pairsA"},
        {"i": 1, "depth": 0, "type": "predict", "cls": "RF", "obj": "RF#1", "X": X,
         "output": {"seq_digest": "model_predictions"}},
        {"i": 2, "depth": 0, "type": "metric", "name": "accuracy_score", "options": {},
         "y_true": y["seq_digest"], "y_pred": "model_predictions", "value": 0.9},
    ]}
    bad = {"events": [dict(e) for e in base["events"]]}
    bad["events"][2] = dict(bad["events"][2], y_pred="fabricated_predictions")
    result = verify(base, SimpleNamespace(ok=True, trace=bad))
    assert result.verdict == METHOD_CHANGED
    assert any("not linked" in v for v in result.violations)


def test_reject_reassigned_labels_even_if_individual_hashes_equal():
    fit = {"i": 0, "depth": 0, "type": "fit", "cls": "RF", "obj": "RF#1", "params": {},
           "X": {"shape": [3, 2], "set_digest": "features"},
           "y": {"set_digest": "same_labels"}, "xy_pairs": "correct_pairs"}
    metric = {"i": 1, "depth": 0, "type": "metric", "name": "silhouette_score",
              "options": {}, "value": 0.9, "y_true": "some_data"}
    ref = {"events": [fit, metric]}
    changed = {"events": [dict(fit, xy_pairs="wrong_pairs"), metric]}
    result = verify(ref, SimpleNamespace(ok=True, trace=changed))
    assert result.verdict == METHOD_CHANGED
    assert any("pairings" in v for v in result.violations)


def test_missing_evidence_is_not_reproduced():
    ref = {"events": [{"i": 0, "depth": 0, "type": "fit", "cls": "RF", "obj": "RF#1", "params": {},
                        "X": {"shape": [2, 1], "set_digest": "X"},
                        "y": {"set_digest": "y"}, "xy_pairs": "pairs"},
                       {"i": 1, "depth": 0, "type": "metric", "name": "accuracy_score", "options": {},
                        "y_true": "y_test", "y_pred": "unknown", "value": 0.9}]}
    rep = verify(ref, SimpleNamespace(ok=True, trace=ref))
    assert rep.verdict == NO_EVIDENCE  # neither trace demonstrates observable model predictions
