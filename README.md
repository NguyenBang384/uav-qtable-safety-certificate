# Exhaustive safety certification of learned Q-tables for UAV path planning

Simulation code and result files for the paper "Exhaustive safety certification of learned
Q-tables for UAV path planning: a sound cell-level abstraction and certified action masking",
submitted to the Journal of Science and Technique (manuscript 1280).

Author: Nguyen Van Bang, Faculty of Missiles, Air Defence-Air Force Academy, Ha Noi, Viet Nam
(banghvpkkq@gmail.com, ORCID 0000-0002-9354-9283).

## Contents

- `code/paper3_sim.py`: Complete simulation in one file (Python 3, NumPy, Matplotlib): the
  discrete unicycle model, the sound cell-to-cell abstraction (Lemmas 1 to 3), the two greatest
  fixed points and the certified action mask, learning inside the mask, the baseline schemes
  (no mask, one-step filter, backup-orbit filter), the unsound abstractions, the resolution
  sweep, the three-dimensional environments E1 to E3, the Monte Carlo checks and all figures.
- `code/make_tables.py`: Prints the numbers of Tables 5 to 10 of the paper directly from the
  stored result file, without running any simulation.
- `figs/p3/results.json`: Result file of the full run. Every number, table and figure of the
  paper is generated from it.
- `figs/p3/plotdata.pkl`: Stored curves and maps, so that the figures can be redrawn without
  retraining.
- `figs/p3/en`, `figs/p3/vi`: Figures as produced by the code, with English and Vietnamese
  labels.

The scripts read and write relative to their own location (`../figs/p3`), so keep the folder
layout above.

## Requirements

Python 3.10 or newer, NumPy and Matplotlib (`pip install -r requirements.txt`). The results in
the paper were produced with Python 3.13 and NumPy 2.5 on one core of an Intel Xeon processor at
2.10 GHz; timings depend on the machine, all other numbers are deterministic for the fixed seed
2026.

## Usage

```
cd code
python3 make_tables.py              # print Tables 5 to 10 from the stored results (seconds)
python3 paper3_sim.py               # full run, about 11 minutes, regenerates results and figures
python3 paper3_sim.py --quick       # reduced budget for a smoke test
python3 paper3_sim.py --replot      # redraw all figures from stored data, no retraining
python3 paper3_sim.py --only3d      # only the three-dimensional environments of Section 5.7
```

## Mapping to the paper

| Paper | Where in the code or result file |
|---|---|
| Tables 5 and 7, Fig. 9 | `grid`, `inv`, `lemma_check`, `mc` in `results.json` |
| Table 6, Figs. 4 to 8 | `train`, `ver`, `repair`, `checkpoints`, `tracks` |
| Table 8 | `unsound` |
| Table 9, Fig. 10 | `res` |
| Table 10, Fig. 11 | `threeD` |

The figure files keep the names used during development (`Fig2_hull.png`, `Fig_ckpt.png`, and so
on); the correspondence with the numbering of the paper is given by the captions of the paper.

## License

MIT License, see `LICENSE`.
