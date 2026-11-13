import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "03_results"


def draw(output, sample, log_sigma, reference, prediction, edges, depth, heights, times):
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10})
    rho = np.power(10.0, -log_sigma.astype(float))
    cmap = plt.get_cmap("RdBu_r")
    norm = LogNorm(1, 1000)
    z_index = np.searchsorted(depth, 117, side="right") - 1
    axis_index = np.searchsorted(edges, 15, side="right") - 1

    figure = plt.figure(figsize=(8.6, 7.4), layout="constrained")
    grid = figure.add_gridspec(3, 3, height_ratios=(1.45, 1.15, 1.15))
    volume = figure.add_subplot(grid[0, :], projection="3d")
    xy_x, xy_y = np.meshgrid(edges, edges, indexing="ij")
    xy_const = np.ones_like(xy_x)
    xz_x, xz_z = np.meshgrid(edges, depth, indexing="ij")
    xz_const = np.ones_like(xz_x)
    volume.plot_surface(xy_x, xy_y, depth[0] * xy_const, facecolors=cmap(norm(rho[:, :, 0])),
                        rstride=1, cstride=1, shade=False, linewidth=0)
    volume.plot_surface(xz_x, edges[0] * xz_const, xz_z, facecolors=cmap(norm(rho[:, 0, :])),
                        rstride=1, cstride=1, shade=False, linewidth=0)
    volume.plot_surface(edges[-1] * xz_const, xz_x, xz_z, facecolors=cmap(norm(rho[-1, :, :])),
                        rstride=1, cstride=1, shade=False, linewidth=0)
    volume.view_init(elev=27, azim=-54)
    volume.set_box_aspect((np.ptp(edges), np.ptp(edges), np.ptp(depth)))
    volume.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Depth (m)", zlim=(depth[-1], 0),
               title=f"(a) Sample {sample}")

    sections = [figure.add_subplot(grid[1, i]) for i in range(3)]
    slices = [
        (rho[:, :, z_index], edges, "Y (m)", "X", "Depth = 117 m"),
        (rho[:, axis_index, :], depth, "Depth (m)", "X", "Y = 15 m"),
        (rho[axis_index, :, :], depth, "Depth (m)", "Y", "X = 15 m"),
    ]
    for axis, (values, vertical, ylabel, horizontal, title) in zip(sections, slices):
        image = axis.pcolormesh(edges, vertical, values.T, cmap=cmap, norm=norm, shading="flat")
        axis.set(xlabel=f"{horizontal} (m)", ylabel=ylabel, title=title)
        if ylabel == "Depth (m)":
            axis.invert_yaxis()
    figure.colorbar(image, ax=sections, fraction=.035, pad=.035, label="Resistivity (Ω·m)")

    bottom = grid[2, :].subgridspec(1, 2)
    curves = figure.add_subplot(bottom[0, 0])
    errors = figure.add_subplot(bottom[0, 1])
    relative = 100.0 * np.abs(prediction / reference - 1.0)
    for i, (height, color) in enumerate(zip(heights, plt.get_cmap("viridis")(np.linspace(.09, .85, len(heights))))):
        curves.plot(times, reference[i], color=color, lw=1.1, label=f"{height:g} m")
        curves.scatter(times, prediction[i], s=11, facecolors="none", edgecolors=color, linewidths=.7)
        errors.plot(times, relative[i], color=color, lw=1)
    curves.set(title="(c) FD (lines) and FWNO (circles)", xlabel="Time (ms)",
               ylabel="Amplitude (nT/s)", yscale="log")
    curves.legend(frameon=False, ncol=2, fontsize=7)
    errors.axhline(5, color="#B36C2D", linestyle="--", linewidth=1)
    errors.axhline(10, color="#378C71", linestyle=":", linewidth=1)
    errors.set(title="(d) Relative error", xlabel="Time (ms)", ylabel="Relative error (%)",
               ylim=(0, 10.8))
    for axis in (curves, errors):
        axis.grid(alpha=.22)
    figure.savefig(output / f"synthetic_sample_{sample}.png", dpi=240)
    plt.close(figure)
    return relative


def main():
    parser = argparse.ArgumentParser(description='Redraw the four synthetic examples from their saved models and amplitudes.')
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(RESULTS / "synthetic_examples.npz", allow_pickle=False) as data:
        for i, model_id in enumerate(data["model_ids"]):
            error = draw(args.out, i + 1, data["log10_conductivity"][i], data["fd_amplitudes"][i],
                         data["fwno_amplitudes"][i], data["horizontal_edges_m"], data["depth_edges_m"],
                         data["heights_m"], data["times_ms"])
            print(f"Sample {i + 1}, model {model_id}: mean {error.mean():.4f}%, maximum {error.max():.4f}%")


if __name__ == "__main__":
    main()
