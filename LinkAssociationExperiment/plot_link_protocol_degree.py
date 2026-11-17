r"""在 IDE 中直接运行：绘制球面覆盖下 multi / single 的卫星总度分布。

读取 results/node_degrees.npz，不重新运行实验。
默认取 t=0 min，保留全部卫星；纵轴是经验 CCDF，不进行分箱或幂律拟合。
修改下方参数即可调整图片，输出同名 PNG 和矢量 PDF，并显示交互窗口。
版式参照论文 Fig. 11 / Fig. 14：横排四面板、独立轴标签、Times New Roman、浅灰网格。

上传 PDF 后可用：
\includegraphics[width=\linewidth]{satellite_ground_link_protocols_t0.pdf}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------- 常用调整参数 ----------
HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "results" / "node_degrees.npz"
OUTPUT_DIR = HERE / "figures"
OUTPUT_NAME = "satellite_ground_link_protocols"

SNAPSHOT_MINUTE = 0          # 可选 0、15、30、…、1425；四个面板使用同一时刻
LAYOUT = (1, 4)
FIGSIZE = (18, 4.5)         # 与 Fig. 11 / Fig. 14 的原绘图画幅一致，单位：英寸
DPI = 300
SHOW_FIGURE = True
LINE_WIDTH = 1.5
MARKER_SIZE = 5
MARKER_EVERY = 8            # 只控制标记间隔，曲线仍使用全部观测度值
LEGEND_LOCATION = "lower left"  # CCDF 的右上区域有曲线，图例统一放在左下方空白处
X_LIMITS = None             # None 表示按数据统一设置，也可写成 (3.5, 300)
Y_LIMITS = None             # None 表示按样本量统一设置，也可写成 (0.0002, 1.2)

NETWORKS = (
    ("Shell1", "Shell1"),
    ("Shell2", "Shell2"),
    ("Shell3", "Shell3"),
    ("Combined", "Combined"),
)
CURVES = {
    "sphere_multi": {"label": "Multi-link", "color": "#125E9D", "linestyle": "-", "marker": "o"},
    "sphere_single": {"label": "Single-link", "color": "#D4232A", "linestyle": "--", "marker": "^"},
}
STYLE = {
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:style=italic",
    "mathtext.bf": "Times New Roman:weight=bold",
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "axes.linewidth": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


class LinkProtocolDegreePlot:
    def __init__(self, data_file=DATA_FILE, minute=SNAPSHOT_MINUTE):
        self.minute = minute
        self.degrees = {}
        with np.load(data_file, allow_pickle=False) as data:
            minutes = data["minutes"]
            matches = np.flatnonzero(minutes == minute)
            if len(matches) != 1:
                raise ValueError(f"找不到唯一的 {minute} min 快照；可选时刻为 {minutes.tolist()}")
            for network, _ in NETWORKS:
                for scenario in CURVES:
                    key = f"{network}__{scenario}__satellite"
                    array = data[key]
                    if array.ndim != 2 or array.shape[0] != len(minutes) or array.shape[1] == 0:
                        raise ValueError(f"度数组尺寸异常：{key}")
                    degree = array[matches[0]]
                    if not np.issubdtype(degree.dtype, np.integer) or np.any(degree < 4):
                        raise ValueError(f"卫星总度应为不小于 4 的整数：{key}")
                    self.degrees[(network, scenario)] = degree
                sizes = {len(self.degrees[(network, scenario)]) for scenario in CURVES}
                if len(sizes) != 1:
                    raise ValueError(f"两组必须保留同一完整卫星集合：{network}")

    @staticmethod
    def empirical_ccdf(degree):
        values, counts = np.unique(degree, return_counts=True)
        probability = np.cumsum(counts[::-1])[::-1] / counts.sum()
        return values, probability

    def plot(self):
        if LAYOUT[0] * LAYOUT[1] != len(NETWORKS):
            raise ValueError("LAYOUT 的面板数量需与 NETWORKS 一致")
        maximum = max(degree.max() for degree in self.degrees.values())
        largest_sample = max(len(degree) for degree in self.degrees.values())
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(*LAYOUT, figsize=FIGSIZE, sharex=True, sharey=True, squeeze=False)
            for index, ((network, label), ax) in enumerate(zip(NETWORKS, axes.flat)):
                for scenario, style in CURVES.items():
                    k, ccdf = self.empirical_ccdf(self.degrees[(network, scenario)])
                    # P(K >= k) 在相邻观测度之间取右端点的值，保留离散阶梯。
                    ax.step(k, ccdf, where="pre", linewidth=LINE_WIDTH,
                            markersize=MARKER_SIZE, markevery=MARKER_EVERY, **style)
                ax.set(xscale="log", yscale="log",
                       xlim=X_LIMITS if X_LIMITS is not None else (3.7, maximum * 1.15),
                       ylim=Y_LIMITS if Y_LIMITS is not None else (0.8 / largest_sample, 1.15))
                ax.set_title(f"({chr(97 + index)}) {label}", pad=6)
                ax.set_xlabel(r"Degree, $k$")
                ax.set_ylabel(r"CCDF, $\Pr(K\geq k)$")
                ax.tick_params(which="both", direction="out", top=False, right=False,
                               labelleft=True, labelbottom=True)
                ax.grid(True, alpha=0.3)
            axes.flat[-1].legend(loc=LEGEND_LOCATION, frameon=True)
            fig.tight_layout()
            return fig

    def save(self, fig):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with plt.rc_context(STYLE):
            for suffix in ("png", "pdf"):
                target = OUTPUT_DIR / f"{OUTPUT_NAME}_t{self.minute}.{suffix}"
                fig.savefig(target, dpi=DPI, facecolor="white", bbox_inches="tight")
                print(f"Saved: {target}")


def self_check():
    k, p = LinkProtocolDegreePlot.empirical_ccdf(np.array([4, 4, 5, 8]))
    assert np.array_equal(k, [4, 5, 8])
    assert np.allclose(p, [1.0, 0.5, 0.25])
    k, p = LinkProtocolDegreePlot.empirical_ccdf(np.array([4, 4]))
    assert np.array_equal(k, [4]) and np.array_equal(p, [1.0])


if __name__ == "__main__":
    self_check()
    plotter = LinkProtocolDegreePlot()
    figure = plotter.plot()
    plotter.save(figure)
    print(f"Spherical coverage; t = {SNAPSHOT_MINUTE} min; all satellite nodes retained.")
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(figure)
