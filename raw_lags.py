"""Lags from the raw dataset files (raw.py): beam~GPS, beam~camera box, camera box~GPS; raw vs calibrated GPS for 8, 9.

Same estimators as gate.py (residual of a pooled cubic fit; nested-CV kNN gain), with the dataset's own Tx boxes as
the camera reference instead of 01's detector. Lags in rows and in seconds (median within-sequence row interval).

    python raw_lags.py
Writes results/raw_lags.json.
"""
import json
from pathlib import Path

import numpy as np

from gate import S, e1, index, pooled_lag
from raw import load_raw

ROOT = Path(__file__).resolve().parent


def bearing(g):
    return np.degrees(np.unwrap(np.arctan2(g[:, 1], g[:, 0])))


def main():
    res = {}
    for s in (1, 2, 3, 4, 5, 6, 7, 8, 9, 14):
        d = load_raw(s)
        pos, L = index(d["seq"])
        ok_y = np.isfinite(d["y"])
        base = np.flatnonzero((pos - S >= 0) & (pos + S < L) & ok_y)
        t = d["t"]
        if np.isfinite(t).all() and np.all(np.abs(t - np.round(t)) < 1e-6):   # whole seconds only (scenario 6); no rtol
            per = [(t[d["seq"] == q].max() - t[d["seq"] == q].min() + 1) / (d["seq"] == q).sum()
                   for q in np.unique(d["seq"]) if (d["seq"] == q).sum() > 5]
            row_dt = float(np.median(per))
        else:
            dt = np.diff(t)[np.diff(d["seq"]) == 0]
            dt = dt[(dt > 0) & (dt < 5)]
            row_dt = float(np.median(dt)) if len(dt) else float("nan")
        r = {"rows": int(len(d["seq"])), "row_dt_s": row_dt}
        streams = {"gps": d["gps"]} | ({"gps_cal": d["gps_cal"]} if "gps_cal" in d else {})
        for name, g in streams.items():
            brg = bearing(g)
            dd = {"gps": g, "y": d["y"].astype(int), "seq": d["seq"]}
            r[name] = {"E1": e1(dd, base),
                       "beam~gps": pooled_lag(d["y"], brg, d["seq"], base),
                       "cam~gps": pooled_lag(d["box_x"], brg, d["seq"], base)}
        r["beam~cam"] = pooled_lag(d["y"], d["box_x"], d["seq"], base)
        res[s] = r
        f = lambda x: "  --" if x is None else f"{x['lag']:+d}"  # noqa: E731
        line = f"s{s:2d} dt {row_dt*1000:4.0f} ms | beam~cam {f(r['beam~cam'])}"
        for name in streams:
            q = r[name]
            sec = "" if np.isnan(row_dt) else f" = {q['beam~gps']['lag'] * row_dt:+.2f} s"
            line += (f" | {name}: beam~gps {f(q['beam~gps'])}{sec}, cam~gps {f(q['cam~gps'])}, "
                     f"E1 {q['E1']['acc_s0']:.3f}->{q['E1']['acc_s_star']:.3f} ({q['E1']['gain']:+.3f}, s* {q['E1']['s_star_folds']})")
        print(line, flush=True)
    (ROOT / "results" / "raw_lags.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
