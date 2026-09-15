# Where these numbers came from

Accuracy columns (MAE, RMSE, bad-N, density) are deterministic: the synthetic
pair is seeded and OpenCV's matchers are not stochastic, so every machine
produces bit-identical values. The `elapsed_ms` column is not, and the
difference is large enough to change a conclusion — see the note in the README.

## `sweep.csv` — reference run

The original sweep. Machine not recorded at the time, which is exactly the
mistake `apple-m1/` exists to stop repeating: a timing column without a machine
attached is not a measurement.

## `apple-m1/sweep.csv`

    machine:  Apple M1, 8 cores, macOS 15.6.1, arm64
    python:   Python 3.9.6
    opencv:   5.0.0
    command:  python cli.py sweep --out results/apple-m1
    date:     2026-09-15T03:52:10Z
    note:     laptop under normal desktop load, not an isolated core.

## Scope note

`sweep.csv` covers one synthetic pair (`...-s0`); `apple-m1/sweep.csv` covers two
(`-s0` and `-s1`), which is why it has 288 rows for the same 144 settings. Every
comparison below is restricted to the `-s0` pair the two runs share.

## What matched-configuration comparison shows

Same two frontier settings, both machines:

| | 5-path (block 9, P1 392, P2 800, uniq 5) | 8-path (block 3, P1 392, P2 1568, uniq 10) | time premium | MAE gain |
|---|---|---|---|---|
| reference | 33.9 ms, MAE 0.280 | 45.3 ms, MAE 0.220 | +34% | 21% |
| Apple M1 | 75.7 ms, MAE 0.280 | 88.7 ms, MAE 0.220 | +17% | 21% |

Across all 144 shared settings the largest MAE difference between the two
machines is exactly **0.0** -- not close, identical. The time premium is not: it
halves. Accuracy conclusions from this sweep transfer; latency
conclusions have to be re-measured on the target.
