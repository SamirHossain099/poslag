"""Offsets in seconds, and whether they are constant within a scenario or vary by sequence.

1. Row interval per scenario from the index CSV's timestamp column (formats differ by testbed; scenario 6 has whole
   seconds only, so its interval is the scenario's duration over its rows within sequences).
2. Per sequence: with the scenario-level map fixed (cubic of the reference stream on GPS bearing, fitted on all
   sequences at the scenario's pooled lag), the shift s in [-S, S] minimising that sequence's residual. Reference =
   camera x of the vehicle where it exists (label-free), else the beam index.

    python offsets.py
Writes results/offsets.json.
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from gate import S, SCEN, SHIFTS, index, load

ROOT = Path(__file__).resolve().parent
from raw import RAW  # noqa: E402


def parse_ts(v):
    """'['03-20-31-142']' (Testbed 1, ms), '['20-20-21']' (s6, s), '01:30:31-839374' (Testbed 5, us) -> seconds."""
    m = re.findall(r"\d+", str(v))
    if len(m) < 3:
        return np.nan
    h, mi, se = map(int, m[:3])
    frac = 0.0
    if len(m) >= 4:
        f = m[3]
        frac = int(f) / (1e6 if len(f) == 6 else 1e3)
    return h * 3600 + mi * 60 + se + frac


def row_interval(s):
    csvs = sorted((p for p in glob.glob(str(RAW / f"scenario{s}" / "**" / "*.csv"), recursive=True)
                   if "resources" not in Path(p).parts), key=lambda p: (len(Path(p).parts), p))
    df = pd.read_csv(csvs[0])
    tc = [c for c in df.columns if "time" in c.lower()][0]
    sc = [c for c in df.columns if "seq" in c.lower()]
    t = df[tc].map(parse_ts).values
    seq = df[sc[0]].values if sc else np.zeros(len(df))
    same = np.diff(seq) == 0
    dt = np.diff(t)[same]
    whole_seconds = all(len(re.findall(r"\d+", str(v))) == 3 for v in df[tc].iloc[:50])
    if whole_seconds:  # duration of each sequence over its rows
        per = [(t[seq == q].max() - t[seq == q].min() + 1) / (seq == q).sum() for q in np.unique(seq) if (seq == q).sum() > 5]
        return float(np.median(per)), "whole seconds only; interval = sequence duration / rows"
    dt = dt[(dt > 0) & (dt < 5)]
    return float(np.median(dt)), f"median of {len(dt)} within-sequence intervals"


def per_sequence(ref, brg, seq, base, lag0, deg=3):
    ok = np.isfinite(ref[base])
    for s in SHIFTS:
        ok &= np.isfinite(brg[base + s])
    b = base[ok]
    z0 = brg[b + lag0]
    mu, sd = z0.mean(), z0.std() + 1e-12
    coef = np.polyfit((z0 - mu) / sd, ref[b], deg)
    out = {}
    for q in np.unique(seq[b]):
        m = b[seq[b] == q]
        if len(m) < 15:
            continue
        r = {s: np.sqrt(np.mean((np.polyval(coef, (brg[m + s] - mu) / sd) - ref[m]) ** 2)) for s in SHIFTS}
        out[int(q)] = min(SHIFTS, key=lambda s: (r[s], abs(s)))
    return out


def main():
    gate = json.loads((ROOT / "results" / "gate.json").read_text())
    res = {}
    for s in SCEN:
        g = gate[str(s)]
        if "skipped" in g:
            continue
        dt, how = row_interval(s)
        d = load(s)
        pos, L = index(d["seq"])
        base = np.flatnonzero((pos - S >= 0) & (pos + S < L))
        brg = np.degrees(np.unwrap(np.radians(np.degrees(np.arctan2(d["gps"][:, 1], d["gps"][:, 0])))))
        use_cam = g.get("lag_cam_gps") is not None
        ref = d["cam_x"] if use_cam else d["y"].astype(float)
        lag0 = (g["lag_cam_gps"] if use_cam else g["lag_beam_gps"])["lag"]
        ps = per_sequence(ref, brg, d["seq"], base, lag0)
        v = np.array(list(ps.values()))
        res[s] = {"row_dt_s": round(dt, 4), "dt_note": how, "reference": "camera" if use_cam else "beam",
                  "pooled_lag_rows": lag0, "pooled_lag_s": round(lag0 * dt, 3),
                  "e1_gain": g["E1"]["gain"], "e1_s_star_rows": g["E1"]["s_star_folds"],
                  "per_seq_n": int(len(v)), "per_seq_median_rows": float(np.median(v)) if len(v) else None,
                  "per_seq_iqr_rows": [float(np.percentile(v, 25)), float(np.percentile(v, 75))] if len(v) else None,
                  "per_seq_frac_at_pooled": float((v == lag0).mean()) if len(v) else None,
                  "per_seq_frac_same_sign": float((np.sign(v) == np.sign(lag0)).mean()) if len(v) and lag0 else None}
        r = res[s]
        print(f"s{s:2d} dt {dt*1000:5.0f} ms | pooled {lag0:+d} rows = {lag0*dt:+.2f} s ({r['reference']}) | "
              f"E1 gain {g['E1']['gain']:+.3f} | per-seq n={r['per_seq_n']} median {r['per_seq_median_rows']} "
              f"IQR {r['per_seq_iqr_rows']} at-pooled {r['per_seq_frac_at_pooled']}", flush=True)
    (ROOT / "results" / "offsets.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
