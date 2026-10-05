"""Every number the letter quotes, computed from results/ into results/manuscript_numbers.json.

Each entry is {"text": <as printed>, "value": <raw>}. PAPER-DRAFT.md carries `{name}` tokens and render_manuscript.py
substitutes them, so no result is typed into the prose by hand (standing rule 4). Table 1 is built here too.

    python manuscript_numbers.py && python render_manuscript.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
MINUS = "−"


def j(name):
    return json.loads((R / name).read_text())


def f(v, d=1, signed=False):
    v = round(v, d) + 0.0                                    # no negative zero
    s = f"{abs(v):.{d}f}"
    if signed:
        return ("+" if v > 0 else MINUS if v < 0 else "") + s
    return (MINUS if v < 0 else "") + s


def change(runs, a="top1_0", b="top1_rule"):
    """Mean paired change over seeds and the half-width of its 95% t interval."""
    d = np.array([r[b] - r[a] for r in runs])
    return float(d.mean()), float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))


def main():
    gate, raw, mor, rule = j("gate.json"), j("raw_lags.json"), j("morais.json"), j("rule.json")
    s23, s41, mech = j("s23.json"), j("s41.json"), j("timestamp_mechanism.json")
    fix = [j(n) for n in ("fix01.json", "fix01_s1.json", "fix01_s2.json")]
    curves, offs_seq = j("fig1_curves.json"), j("offsets.json")

    def resid_rows(s):                                    # minimum of the beam residual over +-9 rows (Figure 1)
        c = curves[s]
        return int(c["shifts"][int(np.argmin(c["rms_beams"]))])
    N = {}

    def put(k, v, text):
        N[k] = {"value": v, "text": text}

    # offsets in seconds (beam~GPS lag x row interval); Testbed 1 positions; 8 raw column
    for s in ("2", "3", "5", "6"):
        q = raw[s]["gps"]
        rows = resid_rows(s)
        assert rows == q["beam~gps"]["lag"], s                  # the +-6 and +-9 searches agree where the text uses both
        sec = rows * raw[s]["row_dt_s"]
        put(f"off{s}", sec, f(sec, 2, signed=True))
        put(f"rows{s}", rows, f(rows, 0, signed=True))
        put(f"camrows{s}", q["cam~gps"]["lag"], f(q["cam~gps"]["lag"], 0, signed=True))
    q8 = raw["8"]
    put("off8raw", q8["gps"]["beam~gps"]["lag"] * q8["row_dt_s"], f(q8["gps"]["beam~gps"]["lag"] * q8["row_dt_s"], 2, True))
    put("rows8raw", q8["gps"]["beam~gps"]["lag"], f(q8["gps"]["beam~gps"]["lag"], 0, True))
    put("rows8cal", q8["gps_cal"]["beam~gps"]["lag"], f(q8["gps_cal"]["beam~gps"]["lag"], 0, True))
    put("acc8raw", q8["gps"]["E1"]["acc_s0"], f(q8["gps"]["E1"]["acc_s0"], 2))
    put("acc8cal", q8["gps_cal"]["E1"]["acc_s0"], f(q8["gps_cal"]["E1"]["acc_s0"], 2))
    sec23 = np.median(s23["gps"]["s_star_folds"]) * s23["row_dt_s"]
    put("off23", sec23, f(sec23, 2, True))
    put("rows23", int(np.median(s23["gps"]["s_star_folds"])), f(int(np.median(s23["gps"]["s_star_folds"])), 0, True))
    put("gain23", s23["gps"]["gain"] * 100, f(s23["gps"]["gain"] * 100, 1))
    put("acc23box", s23["camera"]["beam~box(xy)_E1"]["acc_s0"], f(s23["camera"]["beam~box(xy)_E1"]["acc_s0"], 2))
    dts = [raw[s]["row_dt_s"] * 1000 for s in ("1", "2", "3", "4", "5", "7", "8", "9")]
    put("dt_min", min(dts), f(min(dts), 0))
    put("dt_max", max(dts), f(max(dts), 0))
    offs = [abs(N[f"off{s}"]["value"]) for s in ("2", "3", "5", "6")]
    put("off_min", min(offs), f(min(offs), 2))
    put("off_max", max(offs), f(max(offs), 2))

    # held-out gain of the shift chosen on training sequences (gate.json, kNN)
    for s in ("2", "3", "5", "6"):
        put(f"knn{s}", gate[s]["E1"]["gain"] * 100, f(gate[s]["E1"]["gain"] * 100, 1))

    # scenario 41: file names vs data
    put("s41name", s41["unit1_pwr1_minus_row_s"]["median"], f(s41["unit1_pwr1_minus_row_s"]["median"], 2))
    put("s41gainmax", max(abs(s41[k]["gain"]) for k in ("array1", "array2", "array3")) * 100,
        f(max(abs(s41[k]["gain"]) for k in ("array1", "array2", "array3")) * 100, 1))
    put("s41dup", s41["unit1_pwr1_dup_prev"] * 100, f(s41["unit1_pwr1_dup_prev"] * 100, 1))

    # mechanism
    tb1 = ["1", "2", "3", "4", "5", "7", "8", "23"]
    fr = [mech[s]["frac_ms_equal_k_over_n"] * 100 for s in tb1]
    put("mech_min", min(fr), f(min(fr), 0))
    put("mech_max", max(fr), f(max(fr), 0))
    dv = [mech[s]["distinct_ms_values"] for s in tb1]
    put("mech_distinct_max", max(dv), str(max(dv)))
    put("mech9_distinct", mech["9"]["distinct_ms_values"], str(mech["9"]["distinct_ms_values"]))
    spd = {s: mech[s]["speed"]["median_kmh"] for s in ("2", "3", "5", "6")}
    put("spd_aff_min", min(spd.values()), f(min(spd.values()), 0))
    put("spd_aff_max", max(spd.values()), f(max(spd.values()), 0))
    m = {s: abs(N[f"off{s}"]["value"]) * spd[s] / 3.6 for s in spd}
    put("m_aff_min", min(m.values()), f(min(m.values()), 1))
    put("m_aff_max", max(m.values()), f(max(m.values()), 1))
    sign = [offs_seq[s]["per_seq_frac_same_sign"] * 100 for s in ("2", "3", "5", "6")]
    put("seq_sign_min", min(sign), f(min(sign), 0))
    put("seq_sign_max", max(sign), f(max(sign), 0))
    g3 = [gate[s]["E1"]["gain"] * 100 for s in ("31", "32", "33", "34", "35")]
    put("s3135_gain_max", max(g3), f(max(g3), 1))

    # published baseline: reproduction (all rows with a camera-shifted position) and the camera-lag correction
    rep = [abs(v["sample"]["top1_uncorrected"] - v["published_top1"]) for v in mor.values()]
    put("rep_max", max(rep), f(max(rep), 1))
    for s in ("2", "3", "5", "6", "7"):
        for sp, tag in (("sample", ""), ("sequence", "seq")):
            a_ = mor[s][sp]
            put(f"cam{s}{tag}", a_["top1_corrected"] - a_["top1_uncorrected"],
                f(a_["top1_corrected"] - a_["top1_uncorrected"], 1, True))
    cam = [N[f"cam{s}{t}"]["value"] for s in ("2", "3", "5", "6") for t in ("", "seq")]
    put("cam_aff_min", min(cam), f(min(cam), 1, True))
    put("cam_aff_max", max(cam), f(max(cam), 1, True))

    # the correction rule: paired change over ten seeds with its 95% interval
    aff, clean = ["2", "3", "5", "6"], ["1", "4", "7", "8", "9"]
    ch = {(s, sp): change(rule[s][sp]["runs"]) for s in rule for sp in ("sample", "sequence")}
    g_aff = [ch[(s, sp)][0] for s in aff for sp in ("sample", "sequence")]
    g_cln = [ch[(s, sp)][0] for s in clean for sp in ("sample", "sequence")]
    put("rule_aff_min", min(g_aff), f(min(g_aff), 1))
    put("rule_aff_max", max(g_aff), f(max(g_aff), 1))
    put("rule_clean_min", min(g_cln), f(min(g_cln), 1, True))
    put("rule_clean_max", max(g_cln), f(max(g_cln), 1, True))
    put("rule_clean_cost", -min(g_cln), f(-min(g_cln), 1))
    for sp, tag in (("sample", ""), ("sequence", "seq")):
        put(f"rule23{tag}", ch[("23", sp)][0], f(ch[("23", sp)][0], 1, True))
        put(f"rule23{tag}_h", ch[("23", sp)][1], f(ch[("23", sp)][1], 1))
    pl = {(s, sp): change(rule[s][sp]["runs"], "pl_0", "pl_rule") for s in aff for sp in ("sample", "sequence")}
    falls = sum(1 for v in pl.values() if v[0] < 0)
    put("pl_aff_falls", falls, ["none", "one", "two", "three", "four", "five", "six", "seven", "all eight"][falls])
    put("pl_aff_min", min(v[0] for v in pl.values()), f(min(v[0] for v in pl.values()), 2, True))
    put("pl_aff_max", max(v[0] for v in pl.values()), f(max(v[0] for v in pl.values()), 2, True))
    shifts = {s: [x for sp in ("sample", "sequence") for x in rule[s][sp]["shifts"]] for s in rule}
    nz = {s: float(np.mean([x != 0 for x in shifts[s]])) for s in clean}
    put("clean_nonzero_min", min(nz.values()) * 100, f(min(nz.values()) * 100, 0))
    put("clean_nonzero_max", max(nz.values()) * 100, f(max(nz.values()) * 100, 0))
    s4 = shifts["4"]
    put("s4_shift_lo", min(s4), f(min(s4), 0, True))
    put("s4_shift_hi", max(s4), f(max(s4), 0, True))
    # the camera guard for the rule (Section 4): keep a run's shift only if the camera lag has its sign and is >= 3 rows
    camlag = {s: (raw[s]["gps_cal"] if "gps_cal" in raw[s] else raw[s]["gps"])["cam~gps"]["lag"] for s in aff + clean}

    def kept(s, guard):
        return [guard(r["shift"], camlag[s]) for sp in ("sample", "sequence") for r in rule[s][sp]["runs"]]
    g3 = lambda k, c: k != 0 and np.sign(k) == np.sign(c) and abs(c) >= 3          # noqa: E731
    g1 = lambda k, c: k != 0 and np.sign(k) == np.sign(c) and abs(k - c) <= 1      # noqa: E731
    put("guard_aff", float(np.mean([x for s in aff for x in kept(s, g3)])) * 100,
        f(float(np.mean([x for s in aff for x in kept(s, g3)])) * 100, 0))
    put("guard_clean", float(np.mean([x for s in clean for x in kept(s, g3)])) * 100,
        f(float(np.mean([x for s in clean for x in kept(s, g3)])) * 100, 0))
    put("guard1_s5", float(np.mean(kept("5", g1))) * 100, f(float(np.mean(kept("5", g1))) * 100, 0))
    put("n_seeds", len(rule["2"]["sample"]["seeds"]), str(len(rule["2"]["sample"]["seeds"])))

    # 01, three seeds
    def mean(tag, m, k="top1"):
        return float(np.mean([x[tag][m]["mean"][k] for x in fix]))
    for m, key in (("calib+supft", "cs"), ("calib+gate+norm", "cgn"), ("supft", "sft")):
        put(f"o1_{key}_a", mean("as_published", m), f(mean("as_published", m), 3))
        put(f"o1_{key}_c", mean("corrected", m), f(mean("corrected", m), 3))
        put(f"o1_{key}_pa", mean("as_published", m, "ploss_db"), f(mean("as_published", m, "ploss_db"), 2))
        put(f"o1_{key}_pc", mean("corrected", m, "ploss_db"), f(mean("corrected", m, "ploss_db"), 2))

    put("acc9raw", raw["9"]["gps"]["E1"]["acc_s0"], f(raw["9"]["gps"]["E1"]["acc_s0"], 2))
    put("shift_lo", -6, f(-6, 0, True))
    put("shift_hi", 6, f(6, 0, True))
    top = max(max(rule[s][sp]["top1_rule"], rule[s][sp]["top1_uncorrected"]) for s in rule for sp in ("sample", "sequence"))
    put("top1_corr_max", top, f(int(np.ceil(top / 10) * 10), 0))

    # Table 1
    rows = ["| Scenario | Offset (s) | Shift (rows) | Morais et al. [2] | Reproduced | Sample split | Sequence split |",
            "|---|---|---|---|---|---|---|"]
    for s in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "23"]:
        off = f(sec23, 2, True) if s == "23" else f(resid_rows(s) * raw[s]["row_dt_s"], 2, True)
        pub = f"{mor[s]['published_top1']:.1f}" if s in mor else "n/a"
        rp = f"{mor[s]['sample']['top1_uncorrected']:.1f}" if s in mor else "n/a"
        cells = []
        for sp in ("sample", "sequence"):
            r = rule[s][sp]
            m_, h = ch[(s, sp)]
            cells.append(f"{r['top1_uncorrected']:.1f} → {r['top1_rule']:.1f} ({f(m_, 1, True)} ± {h:.1f})")
        sh = f(int(np.median(shifts[s])), 0, True)
        rows.append(f"| {s}{' (cal.)' if s in ('8', '9') else ''} | {off} | {sh} | {pub} | {rp} | {cells[0]} | {cells[1]} |")
    put("table1", None, "\n".join(rows))                     # true minus signs via f()
    (R / "manuscript_numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(N)} numbers")


if __name__ == "__main__":
    main()
