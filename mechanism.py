"""How were the row timestamps made, and how fast do the vehicles move?

Timestamps: for every whole second of a recording, the k-th of its n rows (k = 0..n-1) is tested for a millisecond
field equal to floor(1000 k / n) (within 1 ms). Evenly spread milliseconds of this form are what a writer produces
when it assigns times to the rows of a second after the fact; measured capture times would scatter. Also the number
of distinct millisecond values and the median rows per second.

Speed: per sequence, the path length of the position track divided by its time span (calibrated column for 8 and 9),
median over sequences; and the distance covered in 0.5 s at that speed.

    python mechanism.py
Writes results/timestamp_mechanism.json.
"""
import json
from pathlib import Path

import numpy as np

import s23
from raw import load_raw

ROOT = Path(__file__).resolve().parent


def assigned(t):
    if np.all(np.abs(t - np.round(t)) < 1e-6):
        return "whole seconds only"
    sec = np.floor(t + 1e-9)
    ms = np.round((t - sec) * 1000)
    hit, per = [], []
    for v in np.unique(sec):
        m = ms[sec == v]
        n = len(m)
        k = np.arange(n)
        hit.append(np.abs(m - np.floor(1000 * k / n)) <= 1)
        per.append(n)
    return {"frac_ms_equal_k_over_n": float(np.concatenate(hit).mean()),
            "rows_per_second_median": float(np.median(per)), "distinct_ms_values": int(len(np.unique(ms)))}


def speed(gps, t, seq):
    v = []
    for q in np.unique(seq):
        i = np.flatnonzero(seq == q)
        span = t[i[-1]] - t[i[0]]
        if len(i) >= 5 and span >= 1:
            v.append(np.linalg.norm(np.diff(gps[i], axis=0), axis=1).sum() / span)
    v = np.asarray(v) * 3.6
    return {"median_kmh": float(np.median(v)), "iqr_kmh": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
            "n_sequences": int(len(v)), "m_in_half_second": float(np.median(v) / 3.6 * 0.5)}


def main():
    res = {}
    for s in [1, 2, 3, 4, 5, 6, 7, 8, 9, 23]:
        d = s23.load() if s == 23 else load_raw(s)
        t = d["t"]
        r = assigned(t)
        gps = d.get("gps_cal", d["gps"])
        r = r if isinstance(r, dict) else {"note": r}
        if s != 23:
            r["speed"] = speed(gps, t, d["seq"])
        res[str(s)] = r
        print(s, json.dumps(r), flush=True)
    (ROOT / "results" / "timestamp_mechanism.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
