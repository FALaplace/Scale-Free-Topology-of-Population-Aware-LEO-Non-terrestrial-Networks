"""Fig4.2: analyze stored degree histograms and draw the nine periodicity panels."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

plt.rcParams["font.family"] = "Times New Roman"


SCRIPT_DIR = Path(__file__).resolve().parent
# 论文旧图对应 1000 终端数据，不能直接沿用 SatConfigure 中的 5000 终端默认值。
DATA_DIR = (
    Path(r"F:\BaiduNetdiskDownload\毕业代码\NetworkSurvivability\satData") / "72_22_550_55_1000_60ms" / "CurveFit"
)
OUTPUT_DIR = SCRIPT_DIR.parents[1] / "output" / "figure4_preview"


def PowerLaw(x, a, b):
    return a * np.asarray(x) ** (-b)


def LogBinning(degreefreq):
    """保留原来的倍增分箱，去掉对全局 degrees 的依赖。"""
    degreefreq = [freq / sum(degreefreq) for freq in degreefreq]
    bins_end = int(np.log2(len(degreefreq))) + 1
    degreefreq += [0] * (2 ** bins_end - len(degreefreq))
    xdata, ydata = [], []
    for i in range(bins_end):
        start, end = 2 ** i, 2 ** (i + 1) - 1
        xdata.append((start + end) / 2)
        ydata.append(sum(degreefreq[start - 1:end]) / (end - start + 1))
    return xdata, ydata


def load_degree_data(data_dir=DATA_DIR):
    """读取已保存的度直方图，包含 t=0 的初始快照。"""
    path = Path(data_dir) / "degreeFreqData.txt"
    with path.open(encoding="utf-8-sig") as stream:
        return [
            [float(value) for value in line.strip().rstrip(",").split(",")]
            for line in stream if line.strip()
        ]


def analyze_periodicity(degree_data, long_period=191, short_period=13, short_cycles=15):
    """计算频率范围、均值、拟合系数和折叠序列，返回可保存为 JSON 的结果。

    延续论文旧图的口径：截取真实度 k>=4，绘图横坐标重编号为 1,2,...；
    191 和 13 是原脚本的折叠窗口长度，采样间隔为 1 min。
    """
    binned_degrees, binned_frequencies, parameters = [], [], []
    for histogram in degree_data:
        degrees, frequencies = LogBinning(histogram[4:])
        params, _ = curve_fit(PowerLaw, degrees, frequencies, maxfev=50000)
        binned_degrees.append(degrees)
        binned_frequencies.append(frequencies)
        parameters.append(params)

    x = np.asarray(binned_degrees)
    y = np.asarray(binned_frequencies)
    parameters = np.asarray(parameters)
    if not np.all(x == x[0]):
        raise ValueError("All snapshots must have the same logarithmic bins.")
    if len(parameters) < max(long_period, short_period * short_cycles):
        raise ValueError("Not enough snapshots for the requested folding windows.")

    long_cycles = len(parameters) // long_period
    count = long_cycles * long_period
    a, b = parameters[:, 0], parameters[:, 1]
    return {
        "source": "72_22_550_55_1000_60ms/CurveFit/degreeFreqData.txt",
        "snapshot_count": len(degree_data),
        "includes_initial_snapshot": True,
        "degree_offset": 3,
        "bin_degrees": x[0].tolist(),
        "frequency_min": y.min(axis=0).tolist(),
        "frequency_max": y.max(axis=0).tolist(),
        "frequency_mean": y.mean(axis=0).tolist(),
        "a": a.tolist(),
        "b": b.tolist(),
        "long_period": long_period,
        "short_period": short_period,
        "long_a": a[:count].reshape(long_cycles, long_period).tolist(),
        "long_b": b[:count].reshape(long_cycles, long_period).tolist(),
        "short_a": a[:short_period * short_cycles].reshape(short_cycles, short_period).tolist(),
        "short_b": b[:short_period * short_cycles].reshape(short_cycles, short_period).tolist(),
    }


def plot_periodicity(results, figsize=(14, 11)):
    """绘图只依赖分析结果；返回完整的 3×3 Figure 和 Axes。"""
    label_size = 18
    tick_size = 15
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    panels = axes.ravel()
    warm = ["#845EC2", "#D65DB1", "#FF6998", "#FF6F91", "#FF9671", "#FFC75F", "#F9F871"]
    cold = ["#79BB7B", "#1FAF8B", "#00A1A1", "#0090B7", "#0090B7", "#007BC3", "#0062BD"]

    panels[0].fill_between(results["bin_degrees"], results["frequency_max"],
                           results["frequency_min"], alpha=0.3)
    panels[0].loglog(results["bin_degrees"], results["frequency_mean"],
                     color="g", linewidth=1.5, alpha=0.7)

    long_period = results["long_period"]
    for step, (a, b) in enumerate(zip(results["long_a"], results["long_b"])):
        time = np.arange(1 + step * long_period, 1 + (step + 1) * long_period)
        panels[1].plot(time, b, linewidth=2, alpha=0.6,
                       color=("#0081CF", "#008F7A")[step % 2])
        panels[2].plot(time, a, linewidth=2, alpha=0.4,
                       color=("orange", "r")[step % 2])
        local_time = np.arange(1, long_period + 1)
        panels[3].plot(local_time, b, linewidth=2, alpha=0.5, linestyle="dashdot",
                       color=cold[-1 - step % len(cold)], label=f"T{step + 1}")
        panels[4].plot(local_time, a, linewidth=2, alpha=0.5, linestyle="dashdot",
                       color=warm[step % len(warm)], label=f"T{step + 1}")
    for ax in panels[3:5]:
        ax.legend(loc="upper right", fontsize=12, labelspacing=0.2)

    short_time = np.arange(1, results["short_period"] + 1)
    for a, b in zip(results["short_a"], results["short_b"]):
        panels[5].plot(short_time, b, alpha=0.6)
        panels[6].plot(short_time, a, alpha=0.6)
    panels[7].boxplot(np.asarray(results["short_b"]), medianprops={"color": "g"})
    panels[8].boxplot(np.asarray(results["short_a"]), medianprops={"color": "r"})
    panels[7].set_ylim(2.2, 3.3)
    panels[8].set_ylim(0.65, 0.9)

    ylabels = ["Probability, p(k)"] + [
        f"Fitting parameter, {symbol}" for symbol in ("b", "a", "b", "a", "b", "a", "b", "a")
    ]
    for i, (ax, ylabel) in enumerate(zip(panels, ylabels)):
        ax.set_xlabel("Degree, k" if i == 0 else "Time, t (min)",
                      fontsize=label_size, labelpad=2)
        ax.set_ylabel(ylabel, fontsize=label_size, labelpad=2)
        ax.tick_params(labelsize=tick_size)
        ax.set_title(f"({chr(97 + i)})", fontsize=label_size, pad=5)
        ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.7, w_pad=0.8, h_pad=1.0)
    return fig, axes


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = SCRIPT_DIR.parent / "Fig4.2.results.json"

    # 首次运行生成结果；之后只改绘图样式时直接读取。
    # 更换数据或分析参数后，将 rebuild_results 改为 True。
    rebuild_results = False
    if rebuild_results or not results_path.exists():
        results = analyze_periodicity(load_degree_data())
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    else:
        results = json.loads(results_path.read_text(encoding="utf-8"))

    fig, axes = plot_periodicity(results)
    # for suffix in ("png", "pdf"):
    #     fig.savefig(OUTPUT_DIR / f"Fig4.2.{suffix}", dpi=200, bbox_inches="tight")
    plt.show()


