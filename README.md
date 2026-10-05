# poslag

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23149346.svg)](https://doi.org/10.5281/zenodo.23149346)

Time offsets between the position rows and the beam and camera rows of the DeepSense 6G dataset: how to measure
them, which stream carries them, and what correcting them does to position-aided beam prediction.

In scenarios 2, 3, 5 and 6 of DeepSense 6G the GPS rows are offset from the beam-power and camera rows by 0.4 to
0.85 s, and in the drone recording (scenario 23) by 0.28 s, with the same sign in nearly every pass. The dataset's
own bounding boxes align with the beam rows and reproduce the offset against the GPS rows, so the position stream
is the displaced one. Choosing a shift on training data raises the position-aided network of Morais et al. (IEEE
ICC 2023), reproduced with its 200-bin input quantisation, by 2.4 to 7.9 points of top-1 accuracy in the four
recordings (means over ten seeds; on held-out sequences the 95% intervals of scenarios 3 and 5 include zero).

## What is here

| File | What it does |
|---|---|
| `raw.py` | Reads scenarios 1 to 9 and 14 from the dataset files: positions, best beam, timestamps, the dataset's "Tx" box |
| `gate.py` | First estimates on parsed caches: held-out kNN gain with the shift chosen on training sequences, and lag estimates from a cubic-fit residual |
| `raw_lags.py` | The three pairwise lags (beam and camera, camera and position, beam and position) from the raw files |
| `offsets.py` | Offsets in seconds and their spread across sequences |
| `s23.py`, `s41.py` | The drone recording (23) and the three-array recording (41) |
| `mechanism.py` | How the row timestamps were made, and vehicle speeds from the position tracks |
| `morais.py` | Re-implementation of the Morais et al. position-aided network, uncorrected and corrected with the camera lag |
| `rule.py` | The correction rule: shift chosen on validation rows only, ten seeds per split |
| `fix01.py` | The Track B streams of the companion package `beamrecal`, rerun with corrected positions |
| `figures.py`, `manuscript_numbers.py` | Figures and the numbers quoted in the letter, computed from `results/` |
| `results/` | Aggregated results (lags, accuracies, curves); no per-frame data |
| `tests/` | Every claim checked against `results/`, and the packaging guard |

## Running it

The DeepSense 6G data are not redistributed here (licence CC BY-NC-ND 4.0). Register and download them from
https://www.deepsense6g.net, extract the scenarios, and point the scripts at them:

```
set DEEPSENSE_ROOT=D:\DeepSense 6G\extracted
pip install -r requirements.txt
python raw.py && python raw_lags.py && python offsets.py && python s23.py && python s41.py && python mechanism.py
python morais.py && python rule.py && python figures.py && python manuscript_numbers.py
python -m pytest tests -q
```

`gate.py` and `fix01.py` reuse the parsed caches, checkpoints and stream evaluator of `beamrecal`
(https://github.com/SamirHossain099/beamrecal); set `BEAMRECAL_ROOT` to its checkout and `BEAMRECAL_CACHE` to
its `data/cache`.

The claim tests and the packaging guard run in CI on Python 3.11, 3.12 and 3.13; they read `results/` and need
only `pytest`, `numpy` and `scipy`. The analysis itself was run on Python 3.12.

## Checking another recording

Shift the positions within each pass by s rows, fit a position-to-beam predictor on training passes, score passes
that were not used to choose s, and adopt the shift only if it wins on training data. Where bounding boxes exist,
the camera-to-position lag confirms which stream moved. A file-name or timestamp difference is not evidence on
its own: scenario 41's power files are named 0.82 s from their rows and the data are aligned.

## Licence

MIT for the code. The dataset belongs to its authors and is used under its own terms.
