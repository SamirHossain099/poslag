"""Every claim in FINDINGS.md that the paper will use, checked against results/ (standing rule 4: never a literal)."""
import json
import os
from pathlib import Path

import pytest

R = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parents[1] / "results"))
OFFSET = ["2", "3", "5", "6"]


def load(name):
    f = R / name
    if not f.exists():
        pytest.skip(f"{name} not produced yet")
    return json.loads(f.read_text())


def test_gate_gain_and_consistent_fold_choice_in_the_offset_scenarios():
    g = load("gate.json")
    for s in OFFSET:
        e = g[s]["E1"]
        assert e["gain"] >= 0.03, (s, e["gain"])
        folds = e["s_star_folds"]
        assert all(f != 0 for f in folds) and len({f > 0 for f in folds}) == 1, (s, folds)


def test_go_rule_met_by_at_least_four_scenarios():
    g = load("gate.json")
    hits = [s for s, v in g.items() if "E1" in v and v["E1"]["gain"] >= 0.03
            and v["lag_beam_gps"] and v["lag_beam_gps"]["lag"] * sum(v["E1"]["s_star_folds"]) > 0]
    assert len(hits) >= 4, hits


def test_beam_and_camera_aligned_gps_offset_from_raw_files():
    r = load("raw_lags.json")
    for s in ["1", "2", "3", "4", "5", "6", "7", "9"]:
        assert abs(r[s]["beam~cam"]["lag"]) == 0, (s, r[s]["beam~cam"])
    assert abs(r["8"]["beam~cam"]["lag"]) <= 1
    for s in ["2", "3", "4", "5", "6"]:
        q = r[s]["gps"]
        assert abs(q["cam~gps"]["lag"] - q["beam~gps"]["lag"]) <= 1, (s, q["cam~gps"], q["beam~gps"])


def test_offsets_in_seconds_and_under_one_second():
    r = load("raw_lags.json")
    for s in OFFSET:
        sec = abs(r[s]["gps"]["beam~gps"]["lag"] * r[s]["row_dt_s"])
        assert 0.3 <= sec < 1.0, (s, sec)


def test_calibrated_column_removes_scenario_8_offset():
    r = load("raw_lags.json")["8"]
    assert abs(r["gps"]["beam~gps"]["lag"]) >= 5 and abs(r["gps_cal"]["beam~gps"]["lag"]) <= 2
    assert r["gps_cal"]["E1"]["acc_s0"] > 3 * r["gps"]["E1"]["acc_s0"]


def test_published_baseline_reproduced_within_three_points():
    m = load("morais.json")
    for s, v in m.items():
        assert abs(v["sample"]["top1_uncorrected"] - v["published_top1"]) <= 3.0, (s, v["sample"]["top1_uncorrected"])


def test_camera_correction_raises_the_published_baseline_in_offset_scenarios():
    m = load("morais.json")
    for s in OFFSET:
        assert m[s]["sample"]["top1_corrected"] - m[s]["sample"]["top1_uncorrected"] >= 3.0, s


def test_scenario_7_is_reported_as_a_failure():
    m = load("morais.json")["7"]
    assert m["sample"]["top1_corrected"] < m["sample"]["top1_uncorrected"] - 3.0


def test_scenario_41_is_aligned():
    r = load("s41.json")
    for k in ("array1", "array2", "array3"):
        assert abs(r[k]["gain"]) < 0.01, (k, r[k]["gain"])
    assert r["unit1_pwr1_minus_row_s"]["median"] < -0.7          # the file names alone would suggest an offset


def test_scenario_23_offset():
    r = load("s23.json")
    assert r["gps"]["gain"] >= 0.05 and set(r["gps"]["s_star_folds"]) == {-2}
    assert r["camera"]["beam~box(xy)_E1"]["s_star_folds"] == [0, 0, 0, 0, 0]


def test_timestamps_assigned_in_testbed1_but_measured_in_9():
    t = load("timestamp_mechanism.json")
    for s in ["1", "2", "5", "7", "8"]:
        assert t[s]["frac_ms_equal_k_over_n"] >= 0.75 and t[s]["distinct_ms_values"] <= 40, s
    assert t["9"]["frac_ms_equal_k_over_n"] < 0.05 and t["9"]["distinct_ms_values"] > 500


def test_rule_gains_in_offset_scenarios_and_costs_little_elsewhere():
    r = load("rule.json")
    for s in OFFSET + ["23"]:
        for split in ("sample", "sequence"):
            assert r[s][split]["top1_rule"] - r[s][split]["top1_uncorrected"] >= 1.5, (s, split)
    for s in ["1", "4", "7", "8", "9"]:
        for split in ("sample", "sequence"):
            assert r[s][split]["top1_rule"] - r[s][split]["top1_uncorrected"] >= -3.5, (s, split)


def test_01_geometric_methods_gain_and_supft_does_not():
    f = load("fix01.json")
    a, c = f["as_published"], f["corrected"]
    for m in ("calib+supft", "calib+gate+norm"):
        assert c[m]["mean"]["top1"] - a[m]["mean"]["top1"] >= 0.05, m
        assert a[m]["mean"]["ploss_db"] - c[m]["mean"]["ploss_db"] >= 0.5, m
    assert abs(c["supft"]["mean"]["top1"] - a["supft"]["mean"]["top1"]) < 0.02


def test_01_gain_holds_in_every_seed():
    for name in ("fix01.json", "fix01_s1.json", "fix01_s2.json"):
        f = load(name)
        a, c = f["as_published"], f["corrected"]
        assert c["calib+supft"]["mean"]["top1"] - a["calib+supft"]["mean"]["top1"] >= 0.05, name
        assert c["calib+gate+norm"]["mean"]["top1"] - a["calib+gate+norm"]["mean"]["top1"] >= 0.05, name
        assert a["supft"]["mean"]["top1"] > a["calib+supft"]["mean"]["top1"], name          # as published, supft led
        assert c["calib+supft"]["mean"]["top1"] > c["supft"]["mean"]["top1"], name          # corrected, it does not
