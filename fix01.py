"""Do 01's Track B results move when the target scenarios' GPS offsets are corrected?

Runs 01's own stream evaluator (sweep every 20 frames, batch 16, seed 0) on Track B (source scenario 1, targets 2, 5,
6, 7) twice: as published, and with each target's GPS row t replaced by row t + lag inside its sequence (edge rows
take the nearest valid row). Lags are the camera-to-GPS estimates from the dataset's own boxes (raw_lags.json):
2 -> -3, 5 -> -5, 6 -> +3; 7 is left uncorrected (unresolved). The correction is injected when 01's loader returns a
scenario; nothing in 01's folder is written.

    python fix01.py
Writes results/fix01.json.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
P01 = Path(os.environ.get("BEAMRECAL_ROOT", ROOT.parent / "01-tta-beam-prediction"))  # beamrecal checkout
sys.path.insert(0, str(P01 / "src"))
import tta_beam.data as data  # noqa: E402
from tta_beam.geo import attach_geo  # noqa: E402
from tta_beam.geo_methods import GEO_METHODS  # noqa: E402
from tta_beam.models import build_model  # noqa: E402
from tta_beam.stream import make_stream, run  # noqa: E402
from tta_beam.tta import METHODS as TTA  # noqa: E402

METHODS = {**TTA, **GEO_METHODS}
LAGS = {"scenario2": -3, "scenario5": -5, "scenario6": 3}
STREAM = ["scenario2", "scenario5", "scenario6", "scenario7"]
RUNS = [("plain", "source", {}), ("plain", "supft", {"lr": 1e-3, "params": "all"}),
        ("anchored", "calib+supft", {}), ("anchored", "calib+gate+norm", {}), ("anchored", "gps-selfcal", {})]
_orig = data.load_scenario


def shifted(name, seed=0):
    d = _orig(name, seed)
    lag = LAGS.get(name, 0) if CORRECT else 0
    if lag:
        g, seq = d["gps"].copy(), d["seq"]
        for v in np.unique(seq):
            m = np.flatnonzero(seq == v)                 # rows of one pass, in time order
            src = np.clip(np.arange(len(m)) + lag, 0, len(m) - 1)
            g[m] = d["gps"][m[src]]
        d = {**d, "gps": g}
    return d


def load_ckpt(kind, dev, seed=0):
    ck = torch.load(P01 / "checkpoints" / f"src_scenario1_gps-cam{'_anchored' if kind == 'anchored' else ''}_s{seed}.pt",
                    map_location=dev)
    model = build_model(ck.get("arch", "plain"), ck["modalities"], cam_ch=ck["cam_ch"]).to(dev)
    missing, unexpected = model.load_state_dict(ck["state"], strict=False)
    if "prior.K_ref" in missing:
        model.prior.K_ref.copy_(model.prior.K)
    return model.eval()


def main(seed=0):
    global CORRECT
    dev = "cuda"
    out = {}
    src = attach_geo(_orig("scenario1"), "scenario1")
    for CORRECT in (False, True):
        tag = "corrected" if CORRECT else "as_published"
        parts = make_stream([attach_geo(shifted(s, seed=seed + 1), s) for s in STREAM], STREAM)
        out[tag] = {}
        for kind, name, kw in RUNS:
            torch.manual_seed(seed)
            if getattr(METHODS[name], "needs_source", False):
                kw = {**kw, "src": src}
            r = run(METHODS[name](load_ckpt(kind, dev, seed), **kw), parts, 16, dev, sweep_every=20)
            out[tag][name] = {d: {k: r[d][k] for k in ("top1", "top3", "ploss_db")} for d in STREAM} | {"mean": r["mean"]}
            print(f"{tag:12s} {name:16s} top1 {r['mean']['top1']:.3f}  ploss {r['mean']['ploss_db']:.2f} dB  | "
                  + "  ".join(f"{d[8:]}:{r[d]['top1']:.3f}" for d in STREAM), flush=True)
    (ROOT / "results" / ("fix01.json" if seed == 0 else f"fix01_s{seed}.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
