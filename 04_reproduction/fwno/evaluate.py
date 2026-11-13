import argparse
import sys
from contextlib import ExitStack
from pathlib import Path
import time
import numpy as np
import h5py
from package_common import CHECKPOINT, EVIDENCE, json_write, output_path, raw_files

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description='Reinfer the fixed 2700 test models from raw HDF5 MAT files, preserving order.')
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    p.add_argument("--allow-other-checkpoint", action="store_true",
                   help="Evaluate a new same-architecture training run; never replace evidence.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate raw shapes and archived order without inference or writing.")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("--batch-size must be positive")
    files = raw_files(args.raw_dir)
    out = output_path(args.output_dir, "evaluation")
    with np.load(EVIDENCE / "v6_test_predictions_and_labels.npz", allow_pickle=False) as z:
        ids = z["test_idx"].copy()
        reference = z["test_truth"].copy()
        archived = z["test_predictions"][0].copy()
        times_ms, heights = z["times_ms"].copy(), z["heights"].copy()
    assert ids.shape == (2700,) and reference.shape == archived.shape == (2700, 112)
    with ExitStack() as stack:
        datasets = [stack.enter_context(h5py.File(path, "r"))["sig"] for path in files]
        for ds in datasets:
            if ds.ndim != 4 or ds.shape[0] < 20 or ds.shape[1:3] != (48, 48) or ds.shape[3] != 3000:
                raise ValueError(f"Unexpected sig layout {ds.shape}; expected [Z,Y,X,3000].")
        print("All 9 HDF5 files and 2700 archived model identifiers validated.", flush=True)
        if args.dry_run:
            print("Dry run passed. No model loaded, inference, or files written.")
            return
        from frozen_runtime import FrozenFWNO
        model = FrozenFWNO(args.checkpoint, args.device, not args.allow_other_checkpoint)
        np.testing.assert_allclose(model.times_ms, times_ms, rtol=0, atol=2e-6)
        np.testing.assert_array_equal(model.heights_m, heights)
        predictions = []
        last_report = time.monotonic()
        for start in range(0, len(ids), args.batch_size):
            model_ids = ids[start:start + args.batch_size]
            raw = np.stack([np.asarray(datasets[int(k)//3000][:20, :, :, int(k)%3000],
                                       dtype=np.float32).transpose(2, 1, 0) for k in model_ids])
            predictions.append(model.predict_log10(raw).reshape(len(raw), 112))
            if time.monotonic() - last_report >= 30 or start + len(raw) == len(ids):
                print(f"Inferred {start + len(raw)}/{len(ids)} models", flush=True)
                last_report = time.monotonic()
    prediction = np.concatenate(predictions)
    assert prediction.shape == reference.shape and np.isfinite(prediction).all()
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "v6_test_predictions_and_labels.npz",
                        test_predictions=prediction[None], test_truth=reference, test_idx=ids,
                        times_ms=times_ms, heights=heights, checkpoint_epoch=np.asarray([model.epoch]))
    delta = prediction.astype(np.float64) - archived.astype(np.float64)
    json_write(out / "inference_audit.json", {
        "checkpoint_sha256": model.checkpoint_sha256, "checkpoint_epoch": model.epoch,
        "batch_size": args.batch_size, "device": args.device,
        "raw_file_order": [str(path) for path in files], "all_test_models": 2700,
        "test_order_preserved": True, "labels": "Unchanged archived log10 FD labels",
        "normalization": "Frozen checkpoint statistics; CPU input encoding",
        "maximum_abs_log10_difference_from_archive": float(np.abs(delta).max()),
        "maximum_relative_amplitude_change_from_archive_percent":
            float(100 * np.abs(np.expm1(np.log(10.0) * delta)).max()),
        "timing": "This evaluation is not a benchmark; raw-file access is included in elapsed execution.",
    })
    from analyze_archive import generate_report
    generate_report(out / "v6_test_predictions_and_labels.npz", out / "statistics")
    print(f"Saved fresh inference and statistics: {out}", flush=True)

if __name__ == "__main__":
    main()
