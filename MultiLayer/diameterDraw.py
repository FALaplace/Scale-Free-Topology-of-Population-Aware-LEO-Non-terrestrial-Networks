"""Fig4.4: load recorded network diameters and draw the four time series."""

from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "satData"
OUTPUT_DIR = SCRIPT_DIR.parents[1] / "output" / "figure4_preview"
SHELLS = {
    "Shell1": "72_22_550_55_5000_60ms",
    "Shell2": "32_50_1110_55_5000_60ms",
    "Shell3": "8_50_1130_55_5000_60ms",
    "Combined": "StarlinkShell123",
}


def load_diameter_results(data_dir=DATA_DIR, end_point=400, sample_minutes=1):
    """读取已有 Diameter.txt；沿用旧图的前 400 个采样点。"""
    results = {}
    for label, folder in SHELLS.items():
        path = Path(data_dir) / folder / "Diameter" / "Diameter.txt"
        with path.open(encoding="utf-8-sig") as stream:
            diameter = [float(line.strip().split(",")[0]) for line in stream if line.strip()]
        diameter = diameter[:end_point]
        results[label] = {
            "time_minutes": [i * sample_minutes for i in range(len(diameter))],
            "diameter": diameter,
        }
    return results


def plot_diameter(res):
    """绘图不计算网络直径；返回横排四图的 Figure 和 Axes。"""
    label_size = 18
    tick_size = 15
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for i, (ax, label) in enumerate(zip(axes.ravel(), SHELLS)):
        data = res[label]
        ax.plot(data["time_minutes"], data["diameter"])
        ax.plot(data["time_minutes"], data["diameter"], ".")
        ax.set_xlabel("Time (mins)", fontsize=label_size)
        ax.set_ylabel("Network diameter", fontsize=label_size)
        ax.tick_params(labelsize=tick_size)
        ax.set_title(f"({chr(97 + i)}) {label}", fontsize=label_size)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, axes


if __name__ == "__main__":
    results = load_diameter_results()
    fig, axes = plot_diameter(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"Fig4.4.{suffix}", dpi=200, bbox_inches="tight")
    # plt.show()
