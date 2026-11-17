"""在 IDE 中直接运行：对比平面与球面覆盖下的多星连接结果。

读取已有快照统计，不重跑实验。第一行绘制卫星总度 CV，第二行绘制
最高度 1% 卫星的总度占比；同一行共用纵轴范围，输出 300 dpi PNG。
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_link_protocol_degree import DPI, NETWORKS, STYLE


# ---------- 常用调整参数 ----------
HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "results" / "snapshot_statistics.csv"
OUTPUT_FILE = HERE / "figures" / "Figure_R1_coverage_geometry_comparison.png"
FIGSIZE = (18, 8.8)
LINE_WIDTH = 1.6
SHOW_FIGURE = False
CURVES = {
    "flat_multi": {"label": "Flat coverage", "color": "#125E9D", "linestyle": "-"},
    "sphere_multi": {"label": "Spherical coverage", "color": "#D4232A", "linestyle": "--"},
}
Y_LABELS = (r"Total-degree CV", "Top-1% degree share (%)")
Y_LIMITS = ((0, 2.0), (5.0, 15.0))


class CoverageGeometryPlot:
    def __init__(self, data_file=DATA_FILE):
        self.series = {(network, scenario): []
                       for network, _ in NETWORKS for scenario in CURVES}
        with Path(data_file).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row["network"], row["scenario"])
                if key in self.series:
                    self.series[key].append((float(row["minute"]), float(row["sat_cv"]), 100 * float(row["sat_top1_share"])))
        expected_minutes = np.arange(0, 1440, 15)
        for key, rows in self.series.items():
            values = np.asarray(sorted(rows), dtype=float)
            if values.shape != (96, 3) or not np.isfinite(values).all():
                raise ValueError(f"快照数量或指标值异常：{key}")
            if not np.array_equal(values[:, 0], expected_minutes):
                raise ValueError(f"快照时刻应为 0 到 1425 min，间隔 15 min：{key}")
            if np.any(values[:, 1:] < 0) or np.any(values[:, 2] > 100):
                raise ValueError(f"CV 或度占比超出有效范围：{key}")
            self.series[key] = values

    def plot(self):
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(2, 4, figsize=FIGSIZE, sharex=True, sharey="row", layout="constrained")
            for row in range(2):
                axes[row, 0].set_ylabel(Y_LABELS[row])
                for column, (network, label) in enumerate(NETWORKS):
                    ax = axes[row, column]
                    for scenario, style in CURVES.items():
                        values = self.series[(network, scenario)]
                        ax.plot(values[:, 0], values[:, row + 1], linewidth=LINE_WIDTH, **style)
                    if not row:
                      ax.set_title(f"{label}", pad=6)
                    if row:
                      ax.set(xlabel="Time (min)", xlim=(0, 1440))
                      ax.set_xticks([0, 360, 720, 1080, 1440])
                   
                    ax.set_ylim(Y_LIMITS[row])
                    ax.set_yticks(np.linspace(*Y_LIMITS[row], 6))
                    ax.tick_params(direction="out", labelleft=True, labelbottom=True, top=False, right=False)
                    ax.grid(True, alpha=0.3)
            fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False)
            return fig

    def save(self, figure):
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(OUTPUT_FILE, dpi=DPI, facecolor="white", bbox_inches="tight")
        print(f"Saved: {OUTPUT_FILE}")


def self_check(plotter):
    # 与独立保存的汇总表核对，避免误取 terrestrial CV 或漏乘百分比。
    with (HERE / "results" / "summary_statistics.csv").open(
            encoding="utf-8-sig", newline="") as stream:
        summary = {(row["network"], row["scenario"]): row for row in csv.DictReader(stream)}
    for key, values in plotter.series.items():
        expected = [float(summary[key]["sat_cv_median"]),
                    100 * float(summary[key]["sat_top1_share_median"])]
        np.testing.assert_allclose(np.median(values[:, 1:], axis=0), expected,
                                   rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    plotter = CoverageGeometryPlot()
    self_check(plotter)
    figure = plotter.plot()
    assert len(figure.axes) == 8 and all(len(ax.lines) == 2 for ax in figure.axes)
    plotter.save(figure)
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(figure)
