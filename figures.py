"""Figures for the paper. Curve data are computed once into results/ and plotted from there (standing rule 4).

  Fig. 1  residual of the beam on GPS bearing against the GPS shift, one panel per scenario (single hue, minimum marked)
  Fig. 2  the three pairwise lags per scenario: beam~camera, camera~GPS, beam~GPS (palette slots 1-3, distinct markers)

    python figures.py
Writes results/fig1_curves.json and figures/fig{1,2}.{png,pdf} at 600 dpi.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gate import index  # noqa: E402
from raw import load_raw  # noqa: E402
from raw_lags import bearing  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]          # reference palette slots 1-3, validated all-pairs (light)
SCEN = [1, 2, 3, 4, 5, 6, 7, 8, 9]
S = 9                                              # wider than the gate's +-6 so every minimum is interior
SHIFTS = list(range(-S, S + 1))
MINUS = "−"


def signed(v):
    return f"{'+' if v > 0 else MINUS if v < 0 else ''}{abs(v)}"

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def curves():
    f = ROOT / "results" / "fig1_curves.json"
    out = {}
    for s in SCEN:
        d = load_raw(s)
        g = d["gps_cal"] if "gps_cal" in d else d["gps"]
        brg = bearing(g)
        pos, L = index(d["seq"])
        base = np.flatnonzero((pos - S >= 0) & (pos + S < L) & np.isfinite(d["y"]))
        y = d["y"][base]
        rms = []
        for k in SHIFTS:
            z = brg[base + k]
            z = (z - z.mean()) / (z.std() + 1e-12)
            rms.append(float(np.sqrt(np.mean((np.polyval(np.polyfit(z, y, 3), z) - y) ** 2))))
        out[s] = {"shifts": SHIFTS, "rms_beams": rms, "stream": "gps_cal" if "gps_cal" in d else "gps"}
    f.write_text(json.dumps(out, indent=1))
    return out


def fig1(c):
    fig, axs = plt.subplots(3, 3, figsize=(7.0, 4.6), sharex=True)
    for ax, s in zip(axs.flat, SCEN):
        r = c[str(s)] if str(s) in c else c[s]
        x, v = np.array(r["shifts"]), np.array(r["rms_beams"])
        ax.plot(x, v, color=SLOT[0], lw=2, solid_capstyle="round")
        k = int(np.argmin(v))
        ax.plot([x[k]], [v[k]], "o", ms=5, color=SLOT[0], mec="white", mew=1.5, zorder=3)
        ax.axvline(0, color=GRID, lw=1, zorder=0)
        rows = "row" if abs(int(x[k])) == 1 else "rows"
        ax.set_title(f"Scenario {s}" + (" (cal.)" if r["stream"] == "gps_cal" else "")
                     + f": minimum at {signed(int(x[k]))} {rows}", color=INK, fontsize=8, pad=3)
        ax.grid(axis="y", color=GRID, lw=0.5)
    for ax in axs[-1]:
        ax.set_xlabel("GPS shift (rows)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Residual (beams)")
    fig.tight_layout()
    return fig


def fig2():
    r = json.loads((ROOT / "results" / "raw_lags.json").read_text())
    rows = [s for s in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    series = [("beam vs camera", lambda q: q["beam~cam"], "o"),
              ("camera vs GPS", lambda q: q["gps_cal" if "gps_cal" in q else "gps"]["cam~gps"], "s"),
              ("beam vs GPS", lambda q: q["gps_cal" if "gps_cal" in q else "gps"]["beam~gps"], "^")]
    for i, (name, get, mk) in enumerate(series):
        ys = np.arange(len(rows)) + (i - 1) * 0.22
        xs = [get(r[s])["lag"] * r[s]["row_dt_s"] for s in rows]
        ax.plot(xs, ys, mk, ms=6, color=SLOT[i], mec="white", mew=1.0, ls="none", label=name)
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"Scenario {s}" + (" (cal.)" if "gps_cal" in r[s] else "") for s in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Lag (s)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, labelcolor=INK,
              handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout()
    return fig


def main():
    OUT.mkdir(exist_ok=True)
    f = ROOT / "results" / "fig1_curves.json"
    c = json.loads(f.read_text()) if f.exists() else curves()
    for name, fig in (("fig1", fig1(c)), ("fig2", fig2())):
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=600, facecolor="white")
        plt.close(fig)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
