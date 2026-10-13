import pytest

from backend.evaluation.metrics import (event_matches, file_level, intersect_len, iou, length, pairwise_ranking, prf,
                                        segment_level, union)


def test_interval_ops():
    assert union([(0, 2), (1, 3), (5, 6)]) == [(0, 3), (5, 6)]
    assert length([(0, 2), (1, 3)]) == 3
    assert intersect_len([(0, 4)], [(1, 2), (3, 6)]) == pytest.approx(2.0)
    assert iou((0, 2), (1, 3)) == pytest.approx(1 / 3)


def test_prf_edge_cases():
    assert prf(2, 2, 0) == {"precision": 0.5, "recall": 1.0, "f1": pytest.approx(2 / 3)}
    assert prf(0, 0, 0)["f1"] is None
    assert prf(0, 3, 0)["f1"] == 0.0


def test_segment_and_file_level():
    files = [{"pred": {"a": [(0, 5)]}, "truth": {"a": [(0, 10)]}},      # half covered
             {"pred": {"a": [(0, 4)]}, "truth": {}},                      # false positive file
             {"pred": {}, "truth": {"a": [(0, 3)]}}]                      # missed file
    s = segment_level(files, "a")
    assert s["tp_s"] == 5 and s["fp_s"] == 4 and s["fn_s"] == 8
    f = file_level(files, "a")
    assert (f["tp"], f["fp"], f["fn"], f["tn"]) == (1, 1, 1, 0)


def test_event_matching_boundaries():
    m = event_matches([(5.0, 10.0), (20, 21)], [(5.5, 10.5)])
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 0)
    assert m["mean_start_err_s"] == pytest.approx(0.5) and m["mean_iou"] == pytest.approx(4.5 / 5.5)
    assert event_matches([], [(0, 1)])["fn"] == 1


def test_pairwise_ranking():
    r = pairwise_ranking({"a": {"clean": 90, "flawed": [70, 95]}, "b": {"clean": None, "flawed": [10]}})
    assert r["pairs"] == 2 and r["flawed_below_clean"] == 1 and r["fraction"] == 0.5
