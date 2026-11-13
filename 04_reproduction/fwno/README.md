# Frozen FWNO reproduction package

This directory contains the FWNO architecture, training module, and normalization utilities used by the epoch-82 checkpoint. The release source omits comments and docstrings. The split, FD labels, predictions, training configuration, and training log are retained in ../../03_results/synthetic. Obtain the checkpoint separately for frozen inference.

## Data and physical conventions

- Exclude test0-3000TD.mat. Supply the remaining nine files, test3000-6000TD.mat through test27000-30000TD.mat, using --raw-dir.
- There are 27,000 models. The fixed split seed is 1234, giving 21,600 training, 2,700 validation, and 2,700 test models.
- Input is **log10 conductivity**, in the stored conductivity convention, with shape [B,48,48,20]. It is not linear conductivity or log10 resistivity. The frozen input normalizer runs on the CPU before transfer.
- All calls predict seven heights, 30–90 m at 10 m intervals, and 16 receiver gates. Ordering is height-major and time-minor. Outputs are log10 response amplitude or positive amplitude in nT/s.
- Archived test predictions and labels are decoded **log10 amplitudes**, not linear amplitudes. All metrics use the complete test set without clipping.
- The configuration uses a 48×48×20 input and a shared temporal decoder.

## Commands

Run from this directory using the Python environment recorded in environment_versions.json, or install compatible versions of PyTorch, NumPy, SciPy, h5py, PyYAML, and Matplotlib. Times New Roman is required for the reproduced figures.

Recompute statistics and figures from the saved arrays, without raw data, training, or GPU inference:

    python -B analyze_archive.py

Validate raw-data paths and HDF5 shapes without inference:

    python -B evaluate.py --raw-dir "PATH_TO_mat_for_training_v2" --dry-run

Evaluate all 2,700 models with the archived weights and frozen normalizers:

    python -B evaluate.py --raw-dir "PATH_TO_mat_for_training_v2" --batch-size 8 --device cuda:0

Batch sizes 1 and 32 are also supported. They may introduce small mixed-precision differences. Each new run saves its own differences from the archived predictions. This command is an accuracy evaluation, not a latency benchmark.

Train a fresh model with the original protocol:

    python -B train.py --raw-dir "PATH_TO_mat_for_training_v2" --dry-run
    python -B train.py --raw-dir "PATH_TO_mat_for_training_v2" --epochs 100

New runs default to a timestamped folder under ../../05_runs. --output-dir accepts another new or empty folder. Runtime outputs cannot be placed in 03_results or 04_reproduction, so evidence and source files are not overwritten. Training may stop early. A fresh stochastic training run is not guaranteed to reproduce the historical weights bit for bit.

For a freshly trained same-architecture checkpoint:

    python -B evaluate.py --raw-dir "PATH_TO_mat_for_training_v2" --checkpoint "PATH_TO_NEW_BEST.pkl" --allow-other-checkpoint

## Programmatic inference

Add this directory to the Python module path and use:

    from frozen_runtime import FrozenFWNO
    model = FrozenFWNO(device="cuda:0")
    log_amplitude = model.predict_log10(raw_log10_sigma)
    amplitude = model.predict_amplitude(raw_log10_sigma)

No normalizer is fitted to new data.

## Evidence boundaries

The package preserves the split, test predictions, and 94-epoch training log. The checkpoint retains the original weights and frozen normalizers and uses portable metadata. The recomputed subdirectory retains recalculated statistics and the training-curve CSV. Running analyze_archive.py generates new figures in its separate output folder. The log's elapsed times are historical training-epoch durations.

The reported_comparisons.json file distinguishes the author-supplied HDF-TD accuracy values and historical FD/HDF-TD timing summaries from the FWNO statistics recomputed here. The earlier 30-repeat FWNO latency benchmark records and batch-size accuracy-validation files were not found in the current workspace. Their published summary values are labeled as reported summaries, not raw measured records. No replacement benchmark or HDF prediction arrays have been fabricated. New inference results never substitute for historical timing evidence.
