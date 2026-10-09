"""Fig4.3: analyze the stored t=180 s graphs, then plot four degree distributions."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(r"F:\BaiduNetdiskDownload\毕业代码\NetworkSurvivability\satData")
OUTPUT_DIR = SCRIPT_DIR.parents[1] / "output" / "figure4_preview"
SHELLS = {
    "Shell1": "72_22_550_55_5000_60ms",
    "Shell2": "32_50_1110_55_5000_60ms",
    "Shell3": "8_50_1130_55_5000_60ms",
}


def load_graphs(data_dir=DATA_DIR, snapshot_seconds=180):
    """只读取原脚本使用的第 3 个一分钟快照，不重新生成网络。"""
    return {
        label: nx.read_gml(
            Path(data_dir) / folder / "Graph" / "satTerrainGraph"
            / f"{snapshot_seconds}_ms_satTerrainGraphgml"
        )
        for label, folder in SHELLS.items()
    }


def LogBinning(degreefreq):
    """沿用 RemoveNodes.LogBinning 的归一化和倍增分箱公式。"""
    degreefreq = [freq / sum(degreefreq) for freq in degreefreq]
    bins_end = int(np.log2(len(degreefreq))) + 1
    degreefreq += [0] * (2 ** bins_end - len(degreefreq))
    xdata, ydata = [], []
    for i in range(bins_end):
        start, end = 2 ** i, 2 ** (i + 1) - 1
        xdata.append((start + end) / 2)
        ydata.append(sum(degreefreq[start - 1:end]) / (end - start + 1))
    return xdata, ydata


def analyze_multilayer(graphs):
    """返回四组可直接绘图、可保存为 JSON 的分布结果。

    保留旧图口径：分别取各层最大连通分量，再组合并取最大连通分量；
    截取真实度 k>=4 后重新归一化，横坐标按原代码重编号为 1,2,...。
    """
    connected = {
        label: graph.subgraph(max(nx.connected_components(graph), key=len))
        for label, graph in graphs.items()
    }
    combined = nx.compose_all(list(connected.values()))
    connected["Combined"] = combined.subgraph(
        max(nx.connected_components(combined), key=len)
    )

    results = {}
    for label, graph in connected.items():
        histogram = nx.degree_histogram(graph)
        frequencies = [count / sum(histogram) for count in histogram][4:]
        frequencies = [value / sum(frequencies) for value in frequencies]
        degrees = list(range(1, len(frequencies) + 1))
        binned_degrees, binned_frequencies = LogBinning(frequencies)
        nonzero = [(degree, freq) for degree, freq in zip(degrees, frequencies) if freq > 0]
        results[label] = {
            "node_count": graph.number_of_nodes(),
            "degree_offset": 3,
            "degrees": [point[0] for point in nonzero],
            "frequencies": [point[1] for point in nonzero],
            "bin_degrees": binned_degrees,
            "bin_frequencies": binned_frequencies,
        }
    return results


def plot_multilayer(results, figsize=(18, 4.5)):
    """绘制 Shell1、Shell2、Shell3 和组合网络的横排四图。"""
    label_size = 18
    tick_size = 15
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    labels = ("Shell1", "Shell2", "Shell3", "Combined")
    for i, (ax, label) in enumerate(zip(axes.ravel(), labels)):
        data = results[label]
        ax.loglog(data["bin_degrees"], data["bin_frequencies"], ".",
                  label="log binning")
        # 延续旧图 endDegree=-1：线性分箱不显示最后一个非零点。
        ax.loglog(data["degrees"][:-1], data["frequencies"][:-1], ".",
                  color="#B2B2B2", alpha=0.3, label="linear binning")
        ax.set_xlabel("Degree, k", fontsize=label_size)
        ax.set_ylabel("Probability, p(k)", fontsize=label_size)
        ax.tick_params(labelsize=tick_size)
        ax.legend(fontsize="x-large")
        ax.set_title(f"({chr(97 + i)}) {label}", fontsize=label_size)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, axes


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = SCRIPT_DIR.parent / "Fig4.3.results.json"

    # 更换快照或分析口径后设为 True；仅调整绘图时读取已有结果即可。
    rebuild_results = False
    if rebuild_results or not results_path.exists():
        results = analyze_multilayer(load_graphs())
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    else:
        results = json.loads(results_path.read_text(encoding="utf-8"))

    fig, axes = plot_multilayer(results)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"Fig4.3.{suffix}", dpi=200, bbox_inches="tight")
    plt.show()


