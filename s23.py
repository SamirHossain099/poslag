"""Scenario 23 (drone, 60 GHz BS on the ground): is the drone's position offset in time from the beam rows?

Features: GPS east/north of the BS (m / 30) and the drone's reported height (m / 30), shifted together; and GPS alone.
Beam = argmax of the 64-beam power file. E1 as in gate.py (held-out kNN, within +-1 beam, shift chosen by inner CV on
training sequences only), shifts up to +-10 rows.

    python s23.py
Writes results/s23.json and data/raw_scenario23.npz.
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
BASE = RAW / "scenario23_dev_w_resources" / "scenario23_dev"


def load():
    f = ROOT / "data" / "raw_scenario23.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    df = pd.read_csv(BASE / "scenario23.csv")
    rd = lambda p: np.loadtxt(BASE / str(p).lstrip("./"))  # noqa: E731
    bs = rd(df["unit1_loc"].iloc[0])[:2]
    e0, n0, *_ = utm.from_latlon(bs[0], bs[1])
    gps = []
    for p in df["unit2_loc"]:
        lat, lon = rd(p)[:2]
        e, n, *_ = utm.from_latlon(lat, lon)
        gps.append([(e - e0) / 30, (n - n0) / 30])
    h = np.array([float(np.atleast_1d(rd(p))[0]) for p in df["unit2_height"]]) / 30
    y = []
    for p in df["unit1_pwr_60ghz"]:
        v = rd(p).astype(float)
        y.append(-1 if np.isnan(v).all() else int(np.nanargmax(v)))
    ts = df["time_stamp[UTC]"].map(lambda v: [int(x) for x in re.findall(r"\d+", str(v))])
    t = np.array([a[0] * 3600 + a[1] * 60 + a[2] + (a[3] / 1e3 if len(a) > 3 else 0) for a in ts], float)
    d = {"gps": np.asarray(gps), "h": h, "y": np.asarray(y), "seq": df["seq_index"].values.astype(np.int64), "t": t}
    f.parent.mkdir(exist_ok=True)
    np.savez_compressed(f, **d)
    return d


def main():
    d = load()
    dt = np.diff(d["t"])[np.diff(d["seq"]) == 0]
    row_dt = float(np.median(dt[(dt > 0) & (dt < 5)]))
    gate.S = 10
    gate.SHIFTS = list(range(-10, 11))
    pos, L = gate.index(d["seq"])
    base = np.flatnonzero((pos - 10 >= 0) & (pos + 10 < L) & (d["y"] >= 0))
    out = {"rows": int(len(d["y"])), "sequences": int(len(np.unique(d["seq"]))), "row_dt_s": row_dt,
           "usable_frames": int(len(base))}
    for name, X in (("gps+height", np.c_[d["gps"], d["h"]]), ("gps", d["gps"])):
        r = gate.e1({"gps": X, "y": d["y"], "seq": d["seq"]}, base)
        out[name] = r
        print(f"s23 {name:10s}: E1 {r['acc_s0']:.3f} -> {r['acc_s_star']:.3f} (gain {r['gain']:+.3f}), s* {r['s_star_folds']} "
              f"= {np.median(r['s_star_folds']) * row_dt:+.2f} s  [rows {out['rows']}, seqs {out['sequences']}, dt {row_dt*1000:.0f} ms]",
              flush=True)
    out["camera"] = camera(d)
    (ROOT / "results" / "s23.json").write_text(json.dumps(out, indent=1))


def box_xy():
    """Centre (x, y) of the dataset's own drone box per row (class 0 in resources/bbox_labels_final; NaN if absent
    or ambiguous)."""
    df = pd.read_csv(BASE / "scenario23.csv")
    bdir = BASE / "resources" / "bbox_labels_final"
    out = []
    for p in df["unit1_rgb"]:
        f = bdir / (Path(str(p)).stem + ".txt")
        rows = [r.split() for r in f.read_text().splitlines() if r.strip()] if f.exists() else []
        rows = [r for r in rows if float(r[0]) == 0]
        out.append((float(rows[0][1]), float(rows[0][2])) if len(rows) == 1 else (np.nan, np.nan))
    return np.asarray(out)


def camera(d):
    """Which stream is offset: beam against the box (E1 on box x, y) and box against the GPS (fit residual)."""
    bx = box_xy()
    pos, L = gate.index(d["seq"])
    base = np.flatnonzero((pos - 10 >= 0) & (pos + 10 < L) & (d["y"] >= 0) & np.isfinite(bx[:, 0]))
    base = base[[np.isfinite(bx[i - 10:i + 11, 0]).all() for i in base]]
    brg = np.degrees(np.unwrap(np.arctan2(d["gps"][:, 1], d["gps"][:, 0])))
    elev = np.degrees(np.arctan2(d["h"], np.hypot(d["gps"][:, 0], d["gps"][:, 1])))
    r = {"frames": int(len(base)), "box_present": float(np.isfinite(bx[:, 0]).mean()),
         "boxx~gps_bearing": gate.pooled_lag(bx[:, 0], brg, d["seq"], base),
         "boxy~gps_elev": gate.pooled_lag(bx[:, 1], elev, d["seq"], base),
         "beam~box(xy)_E1": gate.e1({"gps": bx, "y": d["y"], "seq": d["seq"]}, base),
         "beam~gps_E1_same_frames": gate.e1({"gps": d["gps"], "y": d["y"], "seq": d["seq"]}, base)}
    print(f"s23 camera: beam~box s* {r['beam~box(xy)_E1']['s_star_folds']} acc {r['beam~box(xy)_E1']['acc_s0']:.3f}; "
          f"box x ~ GPS bearing lag {r['boxx~gps_bearing']['lag']}; beam~gps s* {r['beam~gps_E1_same_frames']['s_star_folds']}",
          flush=True)
    return r


if __name__ == "__main__":
    main()
