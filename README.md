# FWNO-TDAEM

Fourier–wavelet neural operator for three-dimensional time-domain airborne electromagnetic forward modeling. The archived model maps a `48 × 48 × 20` log10-conductivity volume to positive dBz/dt amplitudes for seven flight heights and 16 time gates.

Zhiyuan Ke, Yongxin Gao, and Yunhe Liu. Corresponding author: Yongxin Gao (gaoyongx@ustc.edu.cn).

## Install

Use Python 3.10 or later. The archived experiment used PyTorch 2.8.0 with CUDA 12.6. Choose the corresponding PyTorch build for your device.

```sh
python -m pip install -r requirements.txt
```

The `FWNO-TDAEM.zip` Release asset includes the code and model data. The archived-statistics and window-rebuilding commands run using these files. To run fresh FWNO inference, obtain the epoch-82 checkpoint separately and place it in `03_results/synthetic/`. The checksums are recorded in `03_results/synthetic/copy_provenance.json`.

## Reproduce the forward results

Run from the repository root.

```sh
python -B 04_reproduction/fwno/analyze_archive.py
python -B 04_reproduction/musgrave/reproduce.py --metrics-only
python -B 04_reproduction/musgrave/rebuild_windows.py
python -B 04_reproduction/musgrave/predict_50m.py --device cuda:0
```

The first command recomputes the synthetic test metrics from archived predictions and finite-difference labels. The second checks the Musgrave archive and recomputes the 50 m error statistics. The third rebuilds the 396 overlapping input volumes from the resistivity model. The last uses the frozen checkpoint to run fresh forward predictions. See the `fwno` and `musgrave` READMEs under `04_reproduction` for training, data format, and figure commands.

## Included data

| Directory | Contents |
|---|---|
| `03_results/synthetic` | Test split, archived predictions and FD labels, training settings and log |
| `03_results/musgrave` | Survey-derived Musgrave resistivity model, cropped and resampled volumes, numbered FD inputs and reference amplitudes, FWNO predictions at 50 m, and geologic context |
| `03_results/inversion_examples` | Two 400-sounding synthetic examples with truth, observations, geometry, and final deterministic inversion results |

The synthetic inversion examples use FWNO-generated observations. The Musgrave comparison uses the supplied finite-difference responses. The external MAT training inputs, t-SNE refit cache, and FD solver are separate dependencies. To redraw the flight-line panel, use the original study's `iso.imp` file and set `FWNO_MUSGRAVE_FLIGHTLINES` to its path. See [DATA_SOURCES.md](DATA_SOURCES.md) for data attribution and conventions.
