import argparse
import sys
import csv
from pathlib import Path
import re
import numpy as np
from package_common import EVIDENCE, json_write, output_path, sha256

def setup_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 10,
        "axes.labelsize": 10.5, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#687580", "axes.axisbelow": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "savefig.facecolor": "white",
    })
    return plt

def save_figure(fig, stem):
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(Path(stem).with_suffix("." + extension), dpi=360)

def error_summary(values):
    x = np.asarray(values, dtype=np.float64)
    return {"mean_percent": float(x.mean()), "median_percent": float(np.median(x)),
            "p90_percent": float(np.quantile(x, .90)),
            "p95_percent": float(np.quantile(x, .95)),
            "p99_percent": float(np.quantile(x, .99)), "maximum_percent": float(x.max())}

def plot_precision(error_percent, out):
    plt = setup_plotting()
    fwno = [100 * np.mean(error_percent < t) for t in (5, 10)]
    hdf = [63.27, 75.42]
    fig, (cdf, compare) = plt.subplots(1, 2, figsize=(7.05, 2.85))
    fig.subplots_adjust(left=.091, bottom=.19, right=.989, top=.88, wspace=.35)
    blue, gray = "#0074A7", "#687683"
    ordered = np.sort(error_percent, axis=None)
    rank = np.arange(1, len(ordered) + 1)
    cdf.plot(np.r_[0, ordered], np.r_[0, 100 * rank / len(rank)], color=blue, lw=1.65)
    for t, coverage, color, pattern in zip([5, 10], fwno, ["#B36C2D", "#378C71"], ["--", ":"]):
        cdf.axvline(t, color=color, ls=pattern, lw=1)
        cdf.scatter(t, coverage, color=blue, s=15, zorder=5)
    cdf.set(xlim=(0, max(32, float(error_percent.max()) * 1.03)),
            xticks=(0, 5, 10, 15, 20, 25, 30),
            xlabel="Relative error (%)", ylabel="Cumulative fraction (%)")
    cdf.set_title("(a) Full test set", loc="left", fontweight="bold")
    x = np.arange(2)
    bars_a = compare.bar(x - .16, hdf, .32, color=gray, label="HDF-TD")
    bars_b = compare.bar(x + .16, fwno, .32, color=blue, label="FWNO")
    for bars in [bars_a, bars_b]:
        compare.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
    compare.set(xticks=x, xticklabels=["Error < 5%", "Error < 10%"],
                ylabel="Response points (%)")
    compare.legend(loc="lower center", bbox_to_anchor=(.5, -.27), ncol=2,
                   frameon=False, handlelength=1.25, columnspacing=1.35)
    compare.set_title("(b) Comparison on the same split", loc="left", fontweight="bold")
    for ax in [cdf, compare]:
        ax.set(ylim=(0, 111), yticks=np.arange(0, 101, 20))
        ax.grid(axis="y", color="#E3E7EB", lw=.65)
    save_figure(fig, out / "figure6_precision")
    plt.close(fig)

def parse_training_log(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(\d+)\s+([0-9.]+)\s+stage=(\d+)\s", line)
        if not match:
            continue
        row = {"epoch": int(match[1]), "seconds": float(match[2]), "stage": int(match[3])}
        for key in ["trainRawMeanRel", "valRawMeanRel", "trainP5", "valP5",
                    "trainP10", "valP10", "valComposite"]:
            value = re.search(r"\b" + key + r"=([0-9.eE+-]+)%?", line)
            if not value:
                raise ValueError(f"Missing {key} in training log epoch {row['epoch']}")
            row[key] = float(value[1])
        rows.append(row)
    if not rows:
        raise ValueError("No epoch records in training log.")
    return rows

def plot_convergence(path, out):
    rows = parse_training_log(path)
    with (out / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt = setup_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.7), sharex=True)
    fig.subplots_adjust(left=.10, right=.98, bottom=.12, top=.92, hspace=.30, wspace=.33)
    epochs = [r["epoch"] for r in rows]
    for ax, key, label, panel in zip(axes.flat,
            ["RawMeanRel", "P5", "P10", "Composite"],
            ["Mean relative error (%)", "Points with error < 5% (%)",
             "Points with error < 10% (%)", "Validation score"],
            ["(a)", "(b)", "(c)", "(d)"]):
        for lo, hi, shade in [(1, 19.5, "#ECF1F4"), (19.5, 39.5, "#F3F1EB"),
                              (39.5, max(epochs), "#EEF3EE")]:
            ax.axvspan(lo, hi, color=shade, zorder=0)
        if key != "Composite":
            ax.plot(epochs, [r["train" + key] for r in rows],
                    color="#687683", lw=1.15, label="Training")
        ax.plot(epochs, [r["val" + key] for r in rows],
                color="#0074A7", lw=1.2, label="Validation")
        ax.axvline(82, color="#B4592C", ls="--", lw=.9)
        ax.set_ylabel(label)
        ax.set_title(panel, loc="left", fontweight="bold")
        ax.set_xlim(1, max(epochs))
        ax.grid(axis="y", color="#DCE1E5", lw=.55)
    axes[0, 0].legend(frameon=False, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    save_figure(fig, out / "training_convergence")
    plt.close(fig)
    return {"epochs_logged": len(rows), "last_epoch": rows[-1]["epoch"],
            "minimum_logged_validation_score_epoch":
                min(rows, key=lambda r: r["valComposite"])["epoch"],
            "sum_logged_epoch_seconds": sum(r["seconds"] for r in rows),
            "sum_logged_epoch_hours": sum(r["seconds"] for r in rows) / 3600,
            "log_sha256": sha256(path),
            "note": "Values retain printed log precision. Training statistics aggregate changing minibatch states; validation uses end-of-epoch parameters."}

def generate_report(archive, out, training_log=None, bootstrap_repeats=2000):
    archive, out = Path(archive), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(archive, allow_pickle=False) as z:
        pred = z["test_predictions"][0].astype(np.float64)
        truth = z["test_truth"].astype(np.float64)
        ids, times, heights = z["test_idx"], z["times_ms"], z["heights"]
        epoch = int(z["checkpoint_epoch"][0])
    if pred.shape != truth.shape or pred.shape != (2700, 112):
        raise ValueError("Expected one prediction and label vector for each of 2700 test models.")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("Nonfinite archived predictions or labels.")
    e = pred - truth
    r = np.abs(np.expm1(np.log(10.0) * e))
    percent = 100 * r
    p_amp, t_amp = np.power(10.0, pred), np.power(10.0, truth)
    curve_p, curve_t = p_amp.reshape(2700, 7, 16), t_amp.reshape(2700, 7, 16)
    curve_l2 = 100 * np.linalg.norm(curve_p-curve_t, axis=2) / np.linalg.norm(curve_t, axis=2)
    model_l2 = 100 * np.linalg.norm(p_amp-t_amp, axis=1) / np.linalg.norm(t_amp, axis=1)
    report = {
        "source_archive": str(archive.resolve()), "source_sha256": sha256(archive),
        "checkpoint_epoch": epoch, "test_models": len(ids), "response_curves": 18900,
        "response_points": int(r.size), "strict_thresholds": True,
        "pointwise_relative_error": error_summary(percent),
        "curve_relative_l2": error_summary(curve_l2),
        "model_relative_l2": error_summary(model_l2),
        "log10_amplitude_mae_dex": float(np.abs(e).mean()),
        "log10_amplitude_rmse_dex": float(np.sqrt(np.mean(e**2))),
        "P5_percent": float(100 * np.mean(r < .05)),
        "P10_percent": float(100 * np.mean(r < .10)),
        "points_below_5_percent": int(np.count_nonzero(r < .05)),
        "points_below_10_percent": int(np.count_nonzero(r < .10)),
        "Q5_percent": float(100 * np.mean(r.max(axis=1) < .05)),
        "Q10_percent": float(100 * np.mean(r.max(axis=1) < .10)),
        "models_all_below_5_percent": int(np.count_nonzero(r.max(axis=1) < .05)),
        "models_all_below_10_percent": int(np.count_nonzero(r.max(axis=1) < .10)),
        "heights_m": heights.tolist(), "gate_opening_times_ms": times.tolist(),
    }
    r3 = r.reshape(2700, 7, 16)
    for threshold in [.05, .10]:
        exceed = r3 >= threshold
        count = int(exceed.sum())
        report[f"late_four_gates_share_of_{int(threshold*100)}pct_exceedances_percent"] = (
            float(100 * exceed[:, :, -4:].sum() / count) if count else None)
    per_model = np.column_stack([r.mean(axis=1), (r < .05).mean(axis=1), (r < .10).mean(axis=1)])
    rng = np.random.default_rng(20260906)
    draws = np.empty((bootstrap_repeats, 3), dtype=np.float64)
    for i in range(bootstrap_repeats):
        draws[i] = 100 * per_model[rng.integers(0, len(ids), size=len(ids))].mean(axis=0)
    report["bootstrap"] = {
        "seed": 20260906, "generator": "numpy.random.default_rng", "replicates": bootstrap_repeats,
        "unit": "Whole 112-response model",
        "mean_relative_error_95ci_percent": np.quantile(draws[:, 0], [.025, .975]).tolist(),
        "P5_95ci_percent": np.quantile(draws[:, 1], [.025, .975]).tolist(),
        "P10_95ci_percent": np.quantile(draws[:, 2], [.025, .975]).tolist(),
        "scope": "Fixed checkpoint and current test distribution; not training-run uncertainty.",
    }
    for name, positions, dimension in [("by_height", heights, 1), ("by_time", times, 2)]:
        rows = []
        for i, position in enumerate(positions):
            v = np.take(r3, i, axis=dimension)
            rows.append({"coordinate": float(position), "mean_relative_error_percent": float(100*v.mean()),
                         "P5_percent": float(100*np.mean(v < .05)),
                         "P10_percent": float(100*np.mean(v < .10)),
                         "p95_relative_error_percent": float(100*np.quantile(v, .95))})
        with (out / (name + ".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot_precision(percent, out)
    if training_log is not None:
        report["training_log"] = plot_convergence(training_log, out)
    json_write(out / "recomputed_metrics.json", report)
    return report

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description='Recompute synthetic-test statistics and figures without training or inference.')
    p.add_argument("--archive", type=Path, default=EVIDENCE / "v6_test_predictions_and_labels.npz")
    p.add_argument("--training-log", type=Path, default=EVIDENCE / "training_epoch94.log")
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args()
    out = output_path(args.output_dir, "offline_statistics")
    result = generate_report(args.archive, out, args.training_log)
    print(f"Recomputed {result['response_points']} points; MRE={result['pointwise_relative_error']['mean_percent']:.8f}%; "
          f"P5={result['P5_percent']:.8f}%; P10={result['P10_percent']:.8f}%")
    print(f"Saved statistics and figures: {out}")

if __name__ == "__main__":
    main()
