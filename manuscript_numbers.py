"""Every number the letter quotes, computed from results/ into results/manuscript_numbers.json.

Each entry is {"text": <as printed>, "value": <raw>}. PAPER-DRAFT.md carries `{name}` tokens and render_manuscript.py
substitutes them, so no result is typed into the prose by hand (standing rule 4). Table 1 is built here too.

    python manuscript_numbers.py && python render_manuscript.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
MINUS = "−"


def j(name):
    return json.loads((R / name).read_text())


def f(v, d=1, signed=False):
    s = f"{abs(v):.{d}f}"
    if signed:
        return ("+" if v > 0 else MINUS if v < 0 else "") + s
    return (MINUS if v < 0 else "") + s


def main():
    gate, raw, mor, rule = j("gate.json"), j("raw_lags.json"), j("morais.json"), j("rule.json")
    s23, s41, mech = j("s23.json"), j("s41.json"), j("timestamp_mechanism.json")
    fix = [j(n) for n in ("fix01.json", "fix01_s1.json", "fix01_s2.json")]
    N = {}

    def put(k, v, text):
        N[k] = {"value": v, "text": text}

    # offsets in seconds (beam~GPS lag x row interval); Testbed 1 positions; 8 raw column
    for s in ("2", "3", "5", "6"):
        q = raw[s]["gps"]
        sec = q["beam~gps"]["lag"] * raw[s]["row_dt_s"]
        put(f"off{s}", sec, f(sec, 2, signed=True))
        put(f"rows{s}", q["beam~gps"]["lag"], f(q["beam~gps"]["lag"], 0, signed=True))
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
    tb1 = ["1", "2", "3", "4", "5", "7", "8"]
    fr = [mech[s]["frac_ms_equal_k_over_n"] * 100 for s in tb1]
    put("mech_min", min(fr), f(min(fr), 0))
    put("mech_max", max(fr), f(max(fr), 0))
    dv = [mech[s]["distinct_ms_values"] for s in tb1]
    put("mech_distinct_max", max(dv), str(max(dv)))
    put("mech9_distinct", mech["9"]["distinct_ms_values"], str(mech["9"]["distinct_ms_values"]))

    # published baseline: reproduction and correction
    rep = [abs(v["sample"]["top1_uncorrected"] - v["published_top1"]) for v in mor.values()]
    put("rep_max", max(rep), f(max(rep), 1))
    for s in ("2", "3", "5", "6", "7"):
        a = mor[s]["sample"]
        put(f"cam{s}", a["top1_corrected"] - a["top1_uncorrected"], f(a["top1_corrected"] - a["top1_uncorrected"], 1, True))
    b6 = mor["6"]["sequence"]
    put("cam6seq_from", b6["top1_uncorrected"], f(b6["top1_uncorrected"], 1))
    put("cam6seq_to", b6["top1_corrected"], f(b6["top1_corrected"], 1))
    put("cam7seq", mor["7"]["sequence"]["top1_corrected"] - mor["7"]["sequence"]["top1_uncorrected"],
        f(mor["7"]["sequence"]["top1_corrected"] - mor["7"]["sequence"]["top1_uncorrected"], 1, True))
    aff = ["2", "3", "5", "6", "23"]
    clean = ["1", "4", "7", "8", "9"]
    g_aff = [rule[s][sp]["top1_rule"] - rule[s][sp]["top1_uncorrected"] for s in aff for sp in ("sample", "sequence")]
    g_cln = [rule[s][sp]["top1_rule"] - rule[s][sp]["top1_uncorrected"] for s in clean for sp in ("sample", "sequence")]
    put("rule_aff_min", min(g_aff), f(min(g_aff), 1))
    put("rule_aff_max", max(g_aff), f(max(g_aff), 1))
    put("rule_clean_worst", min(g_cln), f(min(g_cln), 1))

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
    put("meters_half_s", 40 / 3.6 * 0.5, f(40 / 3.6 * 0.5, 1))           # 40 km/h for 0.5 s
    top = max(max(rule[s][sp]["top1_rule"], rule[s][sp]["top1_uncorrected"]) for s in rule for sp in ("sample", "sequence"))
    put("top1_corr_max", top, f(int(np.ceil(top / 10) * 10), 0))

    # Table 1
    rows = ["| Scenario | Offset (s) | Morais et al. [2] | Reproduced | Corrected, sample split | Corrected, sequence split |",
            "|---|---|---|---|---|---|"]
    for s in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "23"]:
        if s == "23":
            off = f(sec23, 2, True)
        else:
            st = "gps_cal" if "gps_cal" in raw[s] else "gps"
            off = f(raw[s][st]["beam~gps"]["lag"] * raw[s]["row_dt_s"], 2, True)
        pub = f"{mor[s]['published_top1']:.1f}" if s in mor else "n/a"
        rp = f"{mor[s]['sample']['top1_uncorrected']:.1f}" if s in mor else "n/a"
        a, b = rule[s]["sample"], rule[s]["sequence"]
        rows.append(f"| {s}{' (cal.)' if s in ('8', '9') else ''} | {off} | {pub} | {rp} | "
                    f"{a['top1_uncorrected']:.1f} to {a['top1_rule']:.1f} | {b['top1_uncorrected']:.1f} to {b['top1_rule']:.1f} |")
    put("table1", None, "\n".join(rows))                     # offsets already carry a true minus sign via f()
    (R / "manuscript_numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(N)} numbers")


if __name__ == "__main__":
    main()
