"""Scenario 41 (three 60 GHz arrays, distributed cameras, 11 Hz): file-name clock offsets, and whether the beams
actually fit the GPS better after the shift those offsets imply.

Everything comes from `scenario41.csv` (best beam per array and GPS lat/lon are inline), so no archive is unpacked.
E1 as in gate.py (held-out kNN position -> beam within +-1, shift chosen by inner CV on training groups only), with
groups = 20 s blocks (220 rows) because the scenario has only 4 sequences. Shifts up to +-15 rows.

    python s41.py
Writes results/s41.json.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import utm

import gate
from raw import RAW

ROOT = Path(__file__).resolve().parent
CSV = RAW / "Scenario41 Dataset" / "scenario41.csv"


def t(v):
    m = re.findall(r"(\d{2})-(\d{2})-(\d{2})\.(\d+)", str(v))
    if not m:
        return np.nan
    h, mi, s, f = m[-1]
    return int(h) * 3600 + int(mi) * 60 + int(s) + float("0." + f)


def main():
    df = pd.read_csv(CSV)
    row = df["timestamp"].map(t).values
    out = {"rows": int(len(df)), "sequences": int(df["seq_index"].nunique()),
           "row_dt_s": float(np.median(np.diff(row)[np.diff(df["seq_index"].values) == 0]))}
    for c in ("unit1_rgb1", "unit1_pwr1", "unit1_pwr2", "unit1_pwr3", "unit2_gps1"):
        x = df[c].map(t).values - row
        out[f"{c}_minus_row_s"] = {"median": float(np.nanmedian(x)), "p5": float(np.nanpercentile(x, 5)),
                                   "p95": float(np.nanpercentile(x, 95))}
        out[f"{c}_dup_prev"] = float((df[c].values[1:] == df[c].values[:-1]).mean())
    e, n = zip(*[utm.from_latlon(a, b)[:2] for a, b in zip(df["unit2_gps1_lat"], df["unit2_gps1_lon"])])
    gps = np.c_[np.asarray(e) - e[0], np.asarray(n) - n[0]] / 30.0
    block = (np.arange(len(df)) // 220) + 1000 * df["seq_index"].values      # 20 s groups inside each sequence
    gate.S = 15
    gate.SHIFTS = list(range(-15, 16))
    pos, L = gate.index(block)
    base = np.flatnonzero((pos - 15 >= 0) & (pos + 15 < L))
    for k in (1, 2, 3):
        y = df[f"unit1_pwr{k}_best-beam"].values.astype(int)
        r = gate.e1({"gps": gps, "y": y, "seq": block}, base)
        out[f"array{k}"] = r
        print(f"array {k}: E1 {r['acc_s0']:.3f} -> {r['acc_s_star']:.3f} (gain {r['gain']:+.3f}), s* rows {r['s_star_folds']} "
              f"= {np.median(r['s_star_folds']) * out['row_dt_s']:+.2f} s", flush=True)
    print({k: v for k, v in out.items() if k.endswith("_minus_row_s") or k.endswith("dup_prev")})
    (ROOT / "results" / "s41.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
