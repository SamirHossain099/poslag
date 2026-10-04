"""The correction rule, fixed before this run: the offset is a hyperparameter of the training data.

For each scenario, split and seed (protocol of morais.py): score every shift s in [-6, 6] with a kNN (k = 5, accuracy
within +-1 beam) fitted on the training rows and scored on the validation rows; take the best (ties -> smaller |s|,
so s = 0 unless a shift wins); then train the Morais et al. network at s = 0 and at the chosen s and score both on the
test rows. Only training and validation labels are used to choose. Scenarios 1-9 and 23.

    python rule.py [seed ...]            (default seeds 0-9)
Writes results/rule.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from gate import index
from morais import PUBLISHED, train_eval
from raw import _csv, load_raw
import s23

ROOT = Path(__file__).resolve().parent
SHIFTS = list(range(-6, 7))


def data(s):
    if s == 23:
        d = s23.load()
        df = pd.read_csv(s23.BASE / "scenario23.csv")
        paths = [s23.BASE / str(p).lstrip("./") for p in df["unit1_pwr_60ghz"]]
        y = d["y"].astype(float)
        y[y < 0] = np.nan
        return d["gps"], y, d["seq"], paths
    d = load_raw(s)
    g = d["gps_cal"] if "gps_cal" in d else d["gps"]
    csv = _csv(s)
    df = pd.read_csv(csv)
    col = next(c for c in df.columns if "pwr" in c.lower())
    return g, d["y"], d["seq"], [csv.parent / str(p).lstrip("./") for p in df[col]]


def knn_val(X, y, tr, va):
    clf = KNeighborsClassifier(n_neighbors=5).fit(X[tr], y[tr])
    return float((np.abs(clf.predict(X[va]) - y[va]) <= 1).mean())


def main(seeds=tuple(range(10))):
    res = {}
    for s in [1, 2, 3, 4, 5, 6, 7, 8, 9, 23]:
        g, yf, seq_all, paths = data(s)
        pos, L = index(seq_all)
        rows = np.flatnonzero(np.isfinite(yf) & (pos - 6 >= 0) & (pos + 6 < L))
        y = yf[rows].astype(int)
        seq = seq_all[rows]
        pw = np.stack([np.loadtxt(paths[i]) for i in rows]).astype(float)
        pw = np.nan_to_num(pw, nan=np.nanmin(pw))
        pl = 10 * np.log10(pw.max(1, keepdims=True) / np.maximum(pw, 1e-12))
        Xs = {k: g[rows + k] for k in SHIFTS}
        out = {"n": int(len(rows)), "published_top1": PUBLISHED.get(s)}
        for split in ("sample", "sequence"):
            runs, v_all = [], []
            for sd in seeds:
                rng = np.random.default_rng(sd)
                if split == "sample":
                    p = rng.permutation(len(rows))
                    tr, va, te = np.split(p, [int(0.6 * len(p)), int(0.8 * len(p))])
                else:
                    u = rng.permutation(np.unique(seq))
                    a, b, _ = np.split(u, [int(0.6 * len(u)), int(0.8 * len(u))])
                    tr, va, te = (np.flatnonzero(np.isin(seq, a)), np.flatnonzero(np.isin(seq, b)),
                                  np.flatnonzero(~np.isin(seq, np.r_[a, b])))
                v = {k: knn_val(Xs[k], y, tr, va) for k in SHIFTS}
                k_star = max(SHIFTS, key=lambda k: (v[k], -abs(k)))
                v_all.append({str(k): round(x, 4) for k, x in v.items()})
                t0, p0 = train_eval(Xs[0], y, pl, tr, va, te, sd)
                t1, p1 = (t0, p0) if k_star == 0 else train_eval(Xs[k_star], y, pl, tr, va, te, sd)
                runs.append({"shift": k_star, "top1_0": t0 * 100, "top1_rule": t1 * 100, "pl_0": p0, "pl_rule": p1})
            out[split] = {"shifts": [r["shift"] for r in runs],
                          "top1_uncorrected": float(np.mean([r["top1_0"] for r in runs])),
                          "top1_rule": float(np.mean([r["top1_rule"] for r in runs])),
                          "pl_db_uncorrected": float(np.mean([r["pl_0"] for r in runs])),
                          "pl_db_rule": float(np.mean([r["pl_rule"] for r in runs])), "runs": runs,
                          "val_knn": v_all, "seeds": list(seeds)}
        res[s] = out
        a, b = out["sample"], out["sequence"]
        print(f"s{s:2d} | sample: shifts {a['shifts']} top1 {a['top1_uncorrected']:.1f} -> {a['top1_rule']:.1f} "
              f"(PL {a['pl_db_uncorrected']:.2f} -> {a['pl_db_rule']:.2f}) | sequence: shifts {b['shifts']} top1 "
              f"{b['top1_uncorrected']:.1f} -> {b['top1_rule']:.1f} (PL {b['pl_db_uncorrected']:.2f} -> {b['pl_db_rule']:.2f})",
              flush=True)
    (ROOT / "results" / "rule.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(tuple(int(a) for a in sys.argv[1:]) or tuple(range(10)))
