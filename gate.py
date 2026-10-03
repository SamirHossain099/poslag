"""Gate for 06: are DeepSense GPS rows misaligned in time with the beam (power) rows, and does correcting it pay?

Per scenario, on 01's caches (`../01-tta-beam-prediction/data/cache/scenarioNN.npz`, `_cam_x.npz`):

E1  held-out position -> beam kNN, accuracy within +-1 beam, with the position of row t+s paired with the beam of
    row t. Nested, grouped by sequence: 5 outer folds; the shift s* is chosen by 4-fold CV inside the training folds
    only; the held-out fold is scored at s = 0 and at s*. Gain = mean over outer folds. No test-set selection.
E2  label-free pairwise lags from pooled within-sequence correlation (each sequence demeaned):
      beam vs GPS bearing, beam vs camera x of the vehicle, camera x vs GPS bearing.
    lag(a, b) = s maximising |corr(a[t], b[t+s])|; s > 0 means stream b's row t+s describes the moment of a's row t,
    i.e. b is recorded s rows late relative to a. Closure check: lag(beam,gps) ~ lag(beam,cam) + lag(cam,gps).

Every s is evaluated on the same frames (those with t + s inside the sequence for all |s| <= S).

    python gate.py
Writes results/gate.json (local; derived from DeepSense, CC BY-NC-ND).
"""
import json
import os
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parent
# parsed DeepSense caches of the companion package beamrecal (github.com/SamirHossain099/beamrecal)
CACHE = Path(os.environ.get("BEAMRECAL_CACHE", ROOT.parent / "01-tta-beam-prediction" / "data" / "cache"))
SCEN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 31, 32, 33, 34, 35]
S = 6
SHIFTS = list(range(-S, S + 1))


def load(s):
    z = np.load(CACHE / f"scenario{s}.npz")
    d = {k: z[k] for k in ("gps", "y", "seq")}
    f = CACHE / f"scenario{s}_cam_x.npz"
    d["cam_x"] = np.load(f)["cam_x"].astype(float) if f.exists() else np.full(len(d["y"]), np.nan)
    return d


def index(seq):
    """Row position inside its sequence and sequence length (rows are in time order within a sequence)."""
    pos, L = np.zeros(len(seq), int), np.zeros(len(seq), int)
    for v in np.unique(seq):
        m = np.flatnonzero(seq == v)
        pos[m], L[m] = np.arange(len(m)), len(m)
    return pos, L


def knn_acc(Xtr, ytr, Xte, yte, k=5):
    clf = KNeighborsClassifier(n_neighbors=min(k, len(ytr))).fit(Xtr, ytr)
    return float((np.abs(clf.predict(Xte) - yte) <= 1).mean())


def e1(d, base, seed=0):
    g, y, seq = d["gps"], d["y"], d["seq"]
    seqs = np.unique(seq[base])
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    folds = np.array_split(seqs, 5)
    X = {s: g[base + s] for s in SHIFTS}
    Y = y[base]
    sb = seq[base]
    gains, chosen, acc0, accs = [], [], [], []
    for i, f in enumerate(folds):
        te = np.isin(sb, f)
        trs = np.concatenate([folds[j] for j in range(5) if j != i])
        inner = np.array_split(trs, 4)
        cv = {}
        for s in SHIFTS:
            a = []
            for j, h in enumerate(inner):
                v = np.isin(sb, h)
                t = np.isin(sb, trs) & ~v
                a.append(knn_acc(X[s][t], Y[t], X[s][v], Y[v]))
            cv[s] = np.mean(a)
        s_star = max(SHIFTS, key=lambda s: (cv[s], -abs(s)))
        tr = np.isin(sb, trs)
        a0 = knn_acc(X[0][tr], Y[tr], X[0][te], Y[te])
        a1 = knn_acc(X[s_star][tr], Y[tr], X[s_star][te], Y[te])
        chosen.append(s_star)
        acc0.append(a0)
        accs.append(a1)
        gains.append(a1 - a0)
    return {"acc_s0": float(np.mean(acc0)), "acc_s_star": float(np.mean(accs)), "gain": float(np.mean(gains)),
            "gain_folds": [round(x, 3) for x in gains], "s_star_folds": chosen}


def pooled_lag(a, b, seq, base, deg=3):
    """argmin over s of the residual RMS of a[t] ~ cubic(b[t+s]), pooled over the scenario (the map from bearing or
    image x to beam is fixed by the site geometry). Level correlation saturates near 1 for monotone passes, so the
    residual of a smooth fit is used instead. Returned rms is in a's units; NaNs dropped."""
    ok = np.isfinite(a[base])
    for s in SHIFTS:
        ok &= np.isfinite(b[base + s])
    if ok.sum() < 50:
        return None
    x0 = a[base][ok]
    out = {}
    for s in SHIFTS:
        z = b[base + s][ok]
        z = (z - z.mean()) / (z.std() + 1e-12)
        coef = np.polyfit(z, x0, deg)
        out[s] = float(np.sqrt(np.mean((np.polyval(coef, z) - x0) ** 2)))
    best = min(SHIFTS, key=lambda s: (out[s], abs(s)))
    return {"lag": best, "rms_at_lag": round(out[best], 3), "rms_at_0": round(out[0], 3), "n": int(ok.sum())}


def main():
    res = {}
    for s in SCEN:
        d = load(s)
        pos, L = index(d["seq"])
        base = np.flatnonzero((pos - S >= 0) & (pos + S < L))
        brg = np.degrees(np.arctan2(d["gps"][:, 1], d["gps"][:, 0]))
        brg = np.degrees(np.unwrap(np.radians(brg)))
        y = d["y"].astype(float)
        r = {"n": int(len(base)), "seqs": int(len(np.unique(d["seq"][base])))}
        if r["seqs"] < 10:
            res[s] = {**r, "skipped": "fewer than 10 usable sequences"}
            print(f"s{s:2d} skipped ({r['seqs']} sequences)")
            continue
        r["E1"] = e1(d, base)
        r["lag_beam_gps"] = pooled_lag(y, brg, d["seq"], base)
        r["lag_beam_cam"] = pooled_lag(y, d["cam_x"], d["seq"], base)
        r["lag_cam_gps"] = pooled_lag(d["cam_x"], brg, d["seq"], base)
        res[s] = r
        e, lb, lc, lg = r["E1"], r["lag_beam_gps"], r["lag_beam_cam"], r["lag_cam_gps"]
        fmt = lambda x: "  -- " if x is None else f"{x['lag']:+d} ({x['rms_at_lag']:.2f} vs {x['rms_at_0']:.2f})"  # noqa: E731
        print(f"s{s:2d} n={r['n']:5d} | E1 acc {e['acc_s0']:.3f} -> {e['acc_s_star']:.3f} gain {e['gain']:+.3f} "
              f"s* {e['s_star_folds']} | beam~gps {fmt(lb)} | beam~cam {fmt(lc)} | cam~gps {fmt(lg)}", flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "gate.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
