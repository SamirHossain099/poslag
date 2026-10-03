"""Read Testbed-1 DeepSense scenarios (1-9, 14) straight from the dataset files, independently of 01's cache.

Per row, in CSV order: time (s, NaN if the scenario has no timestamp column), sequence id, UE position east/north of
the BS in metres from `unit2_loc` (and from `unit2_loc_cal` where it exists, scenarios 8 and 9), best beam (argmax of
the 64-beam power file, NaN rows -> NaN), and the x-centre of the dataset's own "Tx" bounding box (class 0 in
`resources/.../bbox`, YOLO format, scaled to [0, 1]; NaN when there is no box or more than one Tx box).

    from raw import load_raw
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import utm

RAW = Path(os.environ.get("DEEPSENSE_ROOT", r"N:\Datasets\DeepSense 6G\extracted"))
CACHE = Path(__file__).resolve().parent / "data"


def _csv(s):
    cs = sorted((p for p in (RAW / f"scenario{s}").rglob("*.csv") if "resources" not in p.parts),
                key=lambda p: (len(p.parts), p.name))
    return cs[0]


def _ts(v):
    m = re.findall(r"\d+", str(v))
    if len(m) < 3:
        return np.nan
    h, mi, se = map(int, m[:3])
    frac = int(m[3]) / (1e6 if len(m[3]) == 6 else 1e3) if len(m) >= 4 else 0.0
    return h * 3600 + mi * 60 + se + frac


def _enu(paths, base, ref):
    out = []
    for p in paths:
        lat, lon = np.loadtxt(base / str(p).lstrip("./"))[:2]
        e, n, *_ = utm.from_latlon(lat, lon)
        out.append([e - ref[0], n - ref[1]])
    return np.asarray(out, float)


def _box_x(path):
    if not path.exists():
        return np.nan
    rows = [r.split() for r in path.read_text().splitlines() if r.strip()]
    tx = [float(r[1]) for r in rows if len(r) >= 5 and float(r[0]) == 0]
    return tx[0] if len(tx) == 1 else np.nan


def load_raw(s, refresh=False):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"raw_scenario{s}.npz"
    if f.exists() and not refresh:
        z = np.load(f)
        return {k: z[k] for k in z.files}
    csv = _csv(s)
    base = csv.parent
    df = pd.read_csv(csv)
    col = lambda *subs: next((c for c in df.columns if all(x in c.lower() for x in subs)), None)  # noqa: E731
    bs = np.loadtxt(base / str(df[col("unit1_loc")].iloc[0]).lstrip("./"))[:2]
    e0, n0, *_ = utm.from_latlon(bs[0], bs[1])
    d = {"seq": df[col("seq")].values.astype(np.int64)}
    tcol = col("time")
    d["t"] = df[tcol].map(_ts).values.astype(float) if tcol else np.full(len(df), np.nan)
    d["gps"] = _enu(df[col("unit2_loc")], base, (e0, n0))
    cal = col("unit2_loc_cal")
    if cal:
        d["gps_cal"] = _enu(df[cal], base, (e0, n0))
    pw = []
    for p in df[col("pwr")]:
        v = np.loadtxt(base / str(p).lstrip("./")).astype(float)
        pw.append(np.nan if np.isnan(v).all() else float(np.nanargmax(v)))
    d["y"] = np.asarray(pw)
    bcol = col("bbox")
    if bcol:                                            # scenario 14: the CSV names the box file
        d["box_x"] = np.asarray([_box_x(base / str(p).lstrip("./")) for p in df[bcol]])
    else:                                               # 1-9: boxes keyed by the image file name
        bdir = next((p for p in base.rglob("bbox") if p.is_dir()), None)
        d["box_x"] = np.asarray([_box_x(bdir / (Path(str(p)).stem + ".txt")) if bdir else np.nan
                                 for p in df[col("rgb")]])
    np.savez_compressed(f, **d)
    return d


if __name__ == "__main__":
    for s in (1, 2, 3, 4, 5, 6, 7, 8, 9, 14):
        d = load_raw(s)
        print(f"s{s:2d} rows {len(d['seq'])}  box_x present {np.isfinite(d['box_x']).mean():.2f}  "
              f"t present {np.isfinite(d['t']).mean():.2f}  cal {'gps_cal' in d}", flush=True)
