"""Does the published position-aided baseline move once the GPS offset is corrected?

Re-implements the neural network of Morais, Behboodi, Pezeshki and Alkhateeb (arXiv 2205.09054, Table I): input the
UE position min-max normalised, 3 hidden layers of 256 ReLU, 64-way output, batch 32, lr 1e-2 reduced x0.2 at epochs
20 and 40, 60 epochs, split 60/20/20, best validation epoch kept. Their published top-1 (Table II) is the target
for the uncorrected run.

The correction is the camera-to-GPS lag from the dataset's own Tx boxes (`results/raw_lags.json`, `cam~gps`), fixed
before any training: the label of row t is paired with the position of row t + lag. No shift is chosen on accuracy.

Splits: "sample" (random rows, as the paper appears to do: neighbouring frames of one pass fall on both sides) and
"sequence" (whole passes held out). Corrected and uncorrected runs use the same frames (rows whose t + lag stays
inside the sequence) and the same split.

    python morais.py
Writes results/morais.json.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from gate import index
from raw import load_raw

ROOT = Path(__file__).resolve().parent
PUBLISHED = {1: 55.57, 2: 48.86, 3: 31.09, 4: 29.14, 5: 43.12, 6: 41.51, 7: 27.82, 8: 43.65, 9: 38.73}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def mlp():
    return nn.Sequential(nn.Linear(2, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 64)).to(DEV)


def train_eval(X, y, pwr_loss, tr, va, te, seed):
    torch.manual_seed(seed)
    lo, hi = X[tr].min(0), X[tr].max(0)
    Xn = torch.as_tensor((X - lo) / (hi - lo + 1e-12), dtype=torch.float32, device=DEV)
    Y = torch.as_tensor(y, dtype=torch.long, device=DEV)
    m = mlp()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [20, 40], 0.2)
    trt = torch.as_tensor(tr, device=DEV)
    best, state = -1.0, None
    g = torch.Generator().manual_seed(seed)
    for ep in range(60):
        m.train()
        perm = trt[torch.randperm(len(trt), generator=g).to(DEV)]
        for i in range(0, len(perm), 32):
            j = perm[i:i + 32]
            loss = nn.functional.cross_entropy(m(Xn[j]), Y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            acc = (m(Xn[va]).argmax(1) == Y[va]).float().mean().item()
        if acc > best:
            best, state = acc, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state)
    with torch.no_grad():
        pred = m(Xn[te]).argmax(1).cpu().numpy()
    top1 = float((pred == y[te]).mean())
    pl = float(np.mean(pwr_loss[te, pred]))
    return top1, pl


def main(seeds=(0, 1, 2)):
    lags = json.loads((ROOT / "results" / "raw_lags.json").read_text())
    res = {}
    for s in range(1, 10):
        d = load_raw(s)
        stream = "gps_cal" if "gps_cal" in d else "gps"          # 8, 9: raw positions are unusable (accuracy < 0.15)
        d["gps"] = d[stream]
        lag = lags[str(s)][stream]["cam~gps"]["lag"]
        pos, L = index(d["seq"])
        ok = np.isfinite(d["y"]) & (pos + lag >= 0) & (pos + lag < L)
        rows = np.flatnonzero(ok)
        y = d["y"][rows].astype(int)
        # power loss of predicting beam b instead of the best, in dB, from the 64-beam power vectors
        pw = np.stack([np.loadtxt(p) for p in _pwr_paths(s, rows)])
        pw = np.nan_to_num(pw, nan=np.nanmin(pw))
        pl = 10 * np.log10(pw.max(1, keepdims=True) / np.maximum(pw, 1e-12))
        X0, X1 = d["gps"][rows], d["gps"][rows + lag]
        seq = d["seq"][rows]
        out = {"lag_rows": lag, "position_stream": stream, "n": int(len(rows)), "published_top1": PUBLISHED[s]}
        for split in ("sample", "sequence"):
            r0, r1 = [], []
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
                r0.append(train_eval(X0, y, pl, tr, va, te, sd))
                r1.append(train_eval(X1, y, pl, tr, va, te, sd))
            r0, r1 = np.array(r0), np.array(r1)
            out[split] = {"top1_uncorrected": r0[:, 0].mean() * 100, "top1_corrected": r1[:, 0].mean() * 100,
                          "pl_db_uncorrected": r0[:, 1].mean(), "pl_db_corrected": r1[:, 1].mean(),
                          "top1_runs": [[round(a * 100, 2), round(b * 100, 2)] for a, b in zip(r0[:, 0], r1[:, 0])]}
        res[s] = out
        a, b = out["sample"], out["sequence"]
        print(f"s{s} lag {lag:+d} | published {PUBLISHED[s]:.1f} | sample split {a['top1_uncorrected']:.1f} -> "
              f"{a['top1_corrected']:.1f} (PL {a['pl_db_uncorrected']:.2f} -> {a['pl_db_corrected']:.2f} dB) | sequence split "
              f"{b['top1_uncorrected']:.1f} -> {b['top1_corrected']:.1f} (PL {b['pl_db_uncorrected']:.2f} -> "
              f"{b['pl_db_corrected']:.2f} dB)", flush=True)
    (ROOT / "results" / "morais.json").write_text(json.dumps(res, indent=1))


def _pwr_paths(s, rows):
    import pandas as pd
    from raw import _csv
    csv = _csv(s)
    df = pd.read_csv(csv)
    col = next(c for c in df.columns if "pwr" in c.lower())
    return [csv.parent / str(p).lstrip("./") for p in df[col].values[rows]]


if __name__ == "__main__":
    main()
