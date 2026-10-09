"""Fig4.5: prepare the recorded hop counts and draw the four paper panels."""

from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parents[1] / "output" / "figure4_preview"
plt.rcParams["font.family"] = "Times New Roman"


def load_hop_counts():
    """原脚本中保存的五组计数，不重新执行网络遍历。"""
    return {
        "shell-1": [4645, 28609, 300503, 824261, 1813983, 3337167, 5746967, 8810441, 12293801, 15380145, 17616383,
                    19208857, 20300695, 20993259, 21360835, 21509735, 21557673, 21570647, 21574543, 21575759, 21576017,
                    21576025],
        "shell-2": [6406, 83232, 1425044, 3160812, 8141090, 14037390, 21824104, 28322256, 33361330, 36356496, 38293556,
                    39499906, 40208036, 40631662, 40865744, 40974636, 41020874, 41033470, 41036336, 41036802, 41036836],
        "shell-3": [3467, 16107, 484987, 1094801, 2540965, 4518067, 6900363, 8939385, 10239011, 11030735, 11474895,
                    11713617, 11853079, 11941853, 11994037, 12015577, 12019557, 12020023, 12020089],
        "shell-1&2": [7990, 108780, 1544590, 3790208, 9803420, 17986600, 29117914, 39390238, 47526724, 53433308,
                      57648192, 60398832, 62086878, 63040674, 63518172, 63725522, 63804046, 63829944, 63837754,
                      63839756, 63840092, 63840100],
        "shell-1&2&3": [8415, 121845, 1648173, 4374431, 11530091, 22158969, 35381747, 46186027, 54528239, 60688311,
                        65026795, 67793761, 69400253, 70228361, 70605243, 70748455, 70795449, 70808763, 70811715,
                        70812201, 70812225],
    }


def analyze_hop_counts(counts):
    """保留旧图的归一化：每一项除以该序列总和，不除以最后一项。"""
    results = {}
    for label, values in counts.items():
        total = sum(values)
        results[label] = {
            "h": list(range(1, len(values) + 1)),
            "counts": list(values),
            "normalized": [value / total for value in values],
        }
    return results


def plot_hop_counts(results, figsize=(18, 4.5)):
    """前三张为各单层原始计数，第四张叠加五组归一化结果。"""
    label_size = 18
    tick_size = 15
    fig, axes = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)
    panels = axes.ravel()
    for ax, label in zip(panels[:3], ("shell-1", "shell-2", "shell-3")):
        data = results[label]
        ax.plot(data["h"], data["counts"])
        ax.plot(data["h"], data["counts"], ".")

    labels = ("shell-1", "shell-2", "shell-3", "shell-1&2", "shell-1&2&3")
    for label in labels:
        data = results[label]
        panels[3].plot(data["h"], data["normalized"], label=label)
    for label in labels:
        data = results[label]
        panels[3].plot(data["h"], data["normalized"], ".")
    panels[3].legend(fontsize="x-large")

    for i, ax in enumerate(panels):
        ax.set_xlabel("Hop-Count, h", fontsize=label_size)
        ax.set_ylabel("Node-pair count, P(h)", fontsize=label_size)
        ax.tick_params(labelsize=tick_size)
        ax.set_title(f"({chr(97 + i)})", fontsize=label_size)
        ax.grid(True, alpha=0.3)
    panels[3].set_ylabel("Normalized count, P(h)", fontsize=label_size)
    fig.tight_layout()
    return fig, axes


if __name__ == "__main__":
    results = analyze_hop_counts(load_hop_counts())
    fig, axes = plot_hop_counts(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"Fig4.5.{suffix}", dpi=200, bbox_inches="tight")
    plt.show()
