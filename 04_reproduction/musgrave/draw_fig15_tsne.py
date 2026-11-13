from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np


from paths import DATA, FIGURES
OUT = FIGURES
SOURCE = DATA / "tsne_coordinates.csv"
STATS = DATA / "comparison_statistics.json"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with SOURCE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = np.asarray([row["dataset"] for row in rows])
    coordinates = np.asarray([[float(row["tsne_1"]), float(row["tsne_2"])]
                              for row in rows], dtype=np.float64)
    assert np.isfinite(coordinates).all()
    assert set(names) == {"synthetic_train", "south_australia"}
    train = coordinates[names == "synthetic_train"]
    field = coordinates[names == "south_australia"]
    assert len(train) == 21600 and len(field) == 396

    font_path = font_manager.findfont("Times New Roman", fallback_to_default=True)
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 10,
        "axes.labelsize": 11, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "axes.linewidth": .65, "axes.edgecolor": "#51565B",
        "text.color": "#24282C", "axes.labelcolor": "#24282C",
        "xtick.color": "#454B50", "ytick.color": "#454B50",
        "xtick.major.width": .65, "ytick.major.width": .65,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(5.72, 4.40))
    fig.subplots_adjust(left=.115, right=.975, bottom=.115, top=.925)
    train_color = "#537C9B"
    field_color = "#B4592C"
    a = ax.scatter(train[:, 0], train[:, 1], s=3.2, c=train_color,
                   alpha=.27, edgecolors="none", rasterized=True, zorder=1)
    b = ax.scatter(field[:, 0], field[:, 1], s=10, c=field_color,
                   marker="D", alpha=.95, edgecolors="none", zorder=3)

    assert np.array_equal(np.asarray(a.get_offsets()), train)
    assert np.array_equal(np.asarray(b.get_offsets()), field)

    low, high = coordinates.min(axis=0), coordinates.max(axis=0)
    margin = (high - low) * .045
    ax.set_xlim(low[0]-margin[0], high[0]+margin[0])
    ax.set_ylim(low[1]-margin[1], high[1]+margin[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("t-SNE 1", labelpad=3)
    ax.set_ylabel("t-SNE 2", labelpad=3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)

    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=train_color,
               markeredgewidth=0, markersize=4.6,
               label="Training Set"),
        Line2D([], [], marker="D", color="none", markerfacecolor=field_color,
               markeredgewidth=0, markersize=4.6,
               label="Musgrave"),
    ]
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .995),
                        ncol=2, frameon=False, fontsize=10, handlelength=.8,
                        handletextpad=.5, columnspacing=1.5, borderaxespad=0)
    fig.canvas.draw()
    extent = legend.get_window_extent(fig.canvas.get_renderer())
    assert extent.x0 >= 0 and extent.x1 <= fig.bbox.width, "Legend clipped"

    files = {}
    for suffix in ("png", "pdf", "svg"):
        path = OUT / f"fig15_tsne_distribution.{suffix}"
        fig.savefig(path, dpi=450)
        files[suffix] = str(path.resolve())
    plt.close(fig)
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    record = {
        "files": files,
        "source_coordinates": str(SOURCE.resolve()),
        "source_coordinates_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_statistics": str(STATS.resolve()),
        "source_statistics_sha256": hashlib.sha256(STATS.read_bytes()).hexdigest(),
        "counts": {"synthetic_training": len(train), "musgrave_windows": len(field)},
        "legend_labels": ["Training Set", "Musgrave"],
        "coordinates_preserved_exactly": True,
        "embedding_refitted": False,
        "font": "Times New Roman", "font_file": font_path,
        "figure_size_in": [5.72, 4.40],
        "minimum_text_size_pt": 9.5,
        "preprocessing": "Frozen epoch-82 per-voxel normalization of log10 conductivity, followed by nonoverlapping 4 x 4 horizontal averaging. All 20 depth layers retained. The 12 x 12 x 20 array is flattened to 2880 features.",
        "pca": {"components": 50, "fit_group": "synthetic_training_only", "solver": "randomized", "seed": 1234,
                "training_explained_variance": stats["tsne"]["pca_training_variance"]},
        "tsne": {**stats["tsne"], "input": "Joint projection of all 21600 training models and 396 Musgrave windows in the same PCA space", "initialization": "pca", "learning_rate": "auto", "metric": "euclidean", "method": "barnes_hut (sklearn default)"},
        "caption_zh": "合成训练模型与Musgrave地区模型窗口的输入分布。图中展示全部21,600个训练模型和396个Musgrave窗口在共同二维t-SNE嵌入中的位置。输入采用冻结FWNO的逐体素归一化，并经水平4×4平均与50维PCA处理。PCA仅由训练模型拟合。该图反映所选输入表征下的局部邻域结构，投影中的重叠不等同于三维模型分布一致，也不能替代FD响应误差验证。",
        "caption_en": "Input distributions of the synthetic training models and Musgrave model windows. All 21,600 training models and 396 Musgrave windows are shown in a joint two-dimensional t-SNE embedding. Inputs use the frozen FWNO per-voxel normalization, followed by horizontal 4 x 4 averaging and 50-component PCA fitted to training models only. The plot describes local neighborhood structure in this input representation. Projected overlap neither establishes agreement between the model distributions nor substitutes for validation against FD responses.",
        "interpretation": "The Musgrave windows form narrow local groups within part of the displayed training cloud. These overlapping windows are correlated. The embedding has no physical coordinate units, and its global distances, occupied areas, and apparent overlap must not be treated as quantitative evidence of accuracy or in-distribution generalization.",
    }
    (OUT / "fig15_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"files": files, "counts": record["counts"], "font": font_path,
                      "coordinate_assertions": "passed", "legend_inside_canvas": True}, ensure_ascii=True))


if __name__ == "__main__":
    main()
