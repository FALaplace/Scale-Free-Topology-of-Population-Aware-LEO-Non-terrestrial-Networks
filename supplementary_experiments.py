"""Code-based supplementary experiments for the ASR revision.

Run this file directly in an IDE and select experiments in
``EXPERIMENTS_TO_RUN`` at the bottom. Each ``run_*`` function is independent.
All inputs are read-only; CSV outputs go to ``supplementary_results`` next to
this script, while figures are shown interactively and are not saved. These
analyses describe the archived code/data, not a verified reproduction of the
submitted figures.
"""

import ast
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import optimize, signal, special, stats

HERE = Path(__file__).resolve().parent
ARCHIVE = Path(r"F:\BaiduNetdiskDownload\毕业代码\NetworkSurvivability\satData")
OUT = HERE / "supplementary_results"
TABLE1_DATA = HERE / "heatmap (LLR)" / "scaleFreeProveData(whole)"
TABLE1_CASES = (
    (169, 4000),
    (400, 8000),
    (900, 10000),
    (16900, 80000),
    (14400, 100000),
    (16900, 200000),
)
SHELLS = (
    "72_22_550_55_5000_60ms",
    "32_50_1110_55_5000_60ms",
    "8_50_1130_55_5000_60ms",
)
SNAPSHOT_MINUTES = (0, 360, 720, 1080)
SAT_NAME = re.compile(r"Sat\d+_\d+(?:_\d+)?\Z")
SEED = 12345678910
MIN_TAIL = 50
CI_REPS = 199
GOF_REPS = 99
BLUE, TEAL, GREEN, GREY = "#125E9D", "#008E9B", "#238B45", "#B2B2B2"
RED = "#D4232A"
PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
    "axes.linewidth": 1.2,
}


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _show_figure(fig):
    fig.tight_layout()
    plt.show()


def _alpha_mle(tail, xmin):
    logs = np.log(tail).sum()
    n = len(tail)
    objective = lambda alpha: n * np.log(special.zeta(alpha, xmin)) + alpha * logs
    result = optimize.minimize_scalar(objective, bounds=(1.01, 30.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Power-law fit failed at xmin={xmin}")
    return float(result.x)


def _powerlaw_ks(tail, xmin, alpha):
    values, counts = np.unique(tail, return_counts=True)
    empirical_after = np.cumsum(counts) / len(tail)
    empirical_before = empirical_after - counts / len(tail)
    normalizer = special.zeta(alpha, xmin)
    model_after = 1 - special.zeta(alpha, values + 1) / normalizer
    model_before = 1 - special.zeta(alpha, values) / normalizer
    return float(max(np.max(np.abs(empirical_after - model_after)),
                     np.max(np.abs(empirical_before - model_before))))


def fit_powerlaw(degrees, min_tail=MIN_TAIL):
    degrees = np.asarray(degrees, dtype=int)
    if len(degrees) < min_tail or np.any(degrees < 1):
        raise ValueError("Need positive, unbinned node degrees and enough observations")
    values, counts = np.unique(degrees, return_counts=True)
    tail_sizes = np.cumsum(counts[::-1])[::-1]
    candidates = values[tail_sizes >= min_tail]
    best = None
    for xmin in candidates:
        tail = degrees[degrees >= xmin]
        alpha = _alpha_mle(tail, int(xmin))
        ks = _powerlaw_ks(tail, int(xmin), alpha)
        if best is None or ks < best["ks"]:
            best = {"xmin": int(xmin), "alpha": alpha, "ks": ks, "n_tail": len(tail)}
    return best


def fit_log_binned_powerlaw(degrees, xmin):
    """Descriptive L-M fit on factor-two log bins over the fitted tail."""
    tail = np.asarray(degrees, dtype=int)
    tail = tail[tail >= xmin]
    centers, densities = [], []
    start = int(xmin)
    while start <= tail.max():
        stop = 2 * start
        count = np.count_nonzero((tail >= start) & (tail < stop))
        if count:
            centers.append((start + stop - 1) / 2)
            densities.append(count / (len(tail) * (stop - start)))
        start = stop
    if len(centers) < 2:
        return {
            "binned_lm_prefactor": float("nan"),
            "binned_lm_alpha": float("nan"),
            "binned_lm_r2_log": float("nan"),
            "binned_lm_n_bins": len(centers),
        }

    log_x, log_y = np.log(centers), np.log(densities)
    line = lambda values, log_prefactor, alpha: log_prefactor - alpha * values
    result = optimize.least_squares(
        lambda parameters: log_y - line(log_x, *parameters),
        x0=(float(log_y.max()), 2.0),
        method="lm",
    )
    if not result.success:
        raise RuntimeError("Log-binned power-law L-M fit failed")
    log_prefactor, alpha = result.x
    fitted = line(log_x, log_prefactor, alpha)
    total = np.sum((log_y - log_y.mean()) ** 2)
    r_squared = 1 - np.sum((log_y - fitted) ** 2) / total if len(log_x) > 2 and total else float("nan")
    return {
        "binned_lm_prefactor": float(np.exp(log_prefactor)),
        "binned_lm_alpha": float(alpha),
        "binned_lm_r2_log": float(r_squared),
        "binned_lm_n_bins": len(centers),
    }


def _powerlaw_logpmf(degrees, xmin, alpha):
    return -alpha * np.log(degrees) - np.log(special.zeta(alpha, xmin))


def _lognormal_logpmf(degrees, xmin, mu, sigma):
    high = (np.log(degrees + 0.5) - mu) / sigma
    low = (np.log(degrees - 0.5) - mu) / sigma
    log_mass = np.empty(len(degrees))
    left = high < 0
    for mask, first, second in (
            (left, special.log_ndtr(high), special.log_ndtr(low)),
            (~left, special.log_ndtr(-low), special.log_ndtr(-high)),
    ):
        delta = np.minimum(second[mask] - first[mask], -1e-15)
        log_mass[mask] = first[mask] + np.log(-np.expm1(delta))
    log_normalizer = special.log_ndtr(-(np.log(xmin - 0.5) - mu) / sigma)
    return log_mass - log_normalizer


def fit_lognormal(tail, xmin):
    logs = np.log(tail)
    center = float(np.mean(logs))
    spread = max(float(np.std(logs)), 0.2)
    bounds = ((np.log(xmin) - 30, np.log(max(tail)) + 10), (0.05, 20.0))
    starts = ((center, spread), (np.log(xmin), 1.0), (center - 1, 2.0),
              (np.log(xmin) - 5, 4.0), (np.log(xmin) - 15, 8.0))
    fits = [
        optimize.minimize(
            lambda params: -np.sum(_lognormal_logpmf(tail, xmin, *params)),
            start,
            bounds=bounds,
            method="L-BFGS-B",
        )
        for start in starts
    ]
    valid = [result for result in fits if result.success and np.isfinite(result.fun)]
    if not valid:
        raise RuntimeError("All lognormal optimizations failed")
    result = min(valid, key=lambda item: item.fun)
    at_boundary = any(np.isclose(value, limit, rtol=0, atol=1e-5)
                      for value, limits in zip(result.x, bounds) for limit in limits)
    return float(result.x[0]), float(result.x[1]), at_boundary


def _sample_powerlaw(rng, n, xmin, alpha):
    """Invert the exact Hurwitz-zeta CDF; no finite-support truncation."""
    u = rng.random(n)
    denominator = special.zeta(alpha, xmin)

    def cdf(k):
        return 1 - special.zeta(alpha, k + 1) / denominator

    lower = np.full(n, xmin, dtype=np.int64)
    upper = np.full(n, max(2 * xmin, xmin + 1), dtype=np.int64)
    missing = cdf(upper) < u
    while np.any(missing):
        upper[missing] *= 2
        missing = cdf(upper) < u
    while np.any(lower < upper):
        middle = lower + (upper - lower) // 2
        below = cdf(middle) < u
        lower = np.where(below, middle + 1, lower)
        upper = np.where(below, upper, middle)
    return lower


def evaluate_degrees(degrees, seed=SEED):
    degrees = np.sort(np.asarray(degrees, dtype=int))
    fit = fit_powerlaw(degrees)
    xmin, alpha = fit["xmin"], fit["alpha"]
    tail, body = degrees[degrees >= xmin], degrees[degrees < xmin]
    fit.update(fit_log_binned_powerlaw(tail, xmin))
    rng = np.random.default_rng(seed)
    ci = [_alpha_mle(rng.choice(tail, len(tail), replace=True), xmin) for _ in range(CI_REPS)]
    fit["ci_low"], fit["ci_high"] = np.quantile(ci, (0.025, 0.975))

    # Parametric bootstrap refits both alpha and xmin, as the observed fit did.
    simulated_ks = []
    for _ in range(GOF_REPS):
        synthetic = _sample_powerlaw(rng, len(tail), xmin, alpha)
        if len(body):
            synthetic = np.concatenate((synthetic, rng.choice(body, len(body), replace=True)))
        simulated_ks.append(fit_powerlaw(synthetic)["ks"])
    fit["p_gof"] = (1 + sum(value >= fit["ks"] for value in simulated_ks)) / (GOF_REPS + 1)

    mu, sigma, at_boundary = fit_lognormal(tail, xmin)
    log_ratio = _powerlaw_logpmf(tail, xmin, alpha) - _lognormal_logpmf(tail, xmin, mu, sigma)
    sd = float(np.std(log_ratio, ddof=1))
    fit["lognormal_mu"], fit["lognormal_sigma"] = mu, sigma
    fit["lognormal_at_boundary"] = at_boundary
    fit["llr_pl_vs_lognormal"] = float(log_ratio.sum())
    fit["p_vuong"] = float(2 * stats.norm.sf(abs(np.sqrt(len(tail)) * np.mean(log_ratio) / sd))) if sd else float("nan")
    fit["n_observations"] = len(degrees)
    fit["tail_fraction"] = len(tail) / len(degrees)
    return fit


def _read_saved_degree_distribution(path):
    with path.open(encoding="utf-8") as file:
        degree_row, probability_row = list(csv.reader(file))[:2]
    degrees = np.array([int(value) for value in degree_row if value], dtype=int)
    probabilities = np.array([float(value) for value in probability_row if value])
    if len(degrees) != len(probabilities) or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError(f"Invalid saved degree distribution: {path}")
    positive = probabilities[probabilities > 0]
    n_observations = round(1 / positive.min())
    counts = np.rint(probabilities * n_observations).astype(int)
    if counts.sum() != n_observations or not np.allclose(counts / n_observations, probabilities):
        raise ValueError(f"Could not recover integer degree counts: {path}")
    return np.repeat(degrees, counts)


def run_table1_refit():
    """R1-3/R1-4: robustly refit the six saved degree sets used in Table 1."""
    rows = []
    for n_satellite, n_terminal in TABLE1_CASES:
        path = TABLE1_DATA / f"GS{n_satellite}_CITY{n_terminal}.txt"
        row = {
            "n_satellite": n_satellite,
            "n_terminal": n_terminal,
            "source": path.name,
        }
        row.update(evaluate_degrees(_read_saved_degree_distribution(path),
                                    seed=SEED + n_satellite + n_terminal))
        rows.append(row)
        print("Table 1 refit", n_satellite, n_terminal, "alpha", round(row["alpha"], 3), flush=True)
    _write_csv(OUT / "table1_robust_refit.csv", rows)
    return rows


def _city_population_marginals():
    cities = []
    with (HERE / "worldcities.txt").open(encoding="utf-8") as file:
        for line in file:
            name, lat, lon, country, population = line.strip().split(",")
            cities.append((float(lat), float(lon), float(population)))
            if len(cities) == 1000:
                break
    cities = np.asarray(cities)
    lat_edges = np.linspace(-90, 90, 37)
    lon_edges = np.linspace(-180, 180, 37)
    lat_weights = np.histogram(cities[:, 0], bins=lat_edges, weights=cities[:, 2])[0]
    lon_weights = np.histogram(cities[:, 1], bins=lon_edges, weights=cities[:, 2])[0]
    return lat_edges, np.cumsum(lat_weights / lat_weights.sum()), lon_edges, np.cumsum(lon_weights / lon_weights.sum())


def _weighted_coordinates(n, seed):
    lat_edges, lat_cdf, lon_edges, lon_cdf = _city_population_marginals()
    rng = random.Random(seed)
    lat = np.array([lat_edges[min(np.searchsorted(lat_cdf, rng.random()), 35)] + 5 * rng.random() for _ in range(n)])
    lon = np.array([lon_edges[min(np.searchsorted(lon_cdf, rng.random()), 35)] + 10 * rng.random() for _ in range(n)])
    return lat, lon


def _uniform_coordinates(n, seed):
    rng = np.random.default_rng(seed)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    lon = rng.uniform(-180, 180, n)
    return lat, lon


def _static_grid_degrees(lat, lon):
    # Matches the 33x33 toroidal satellite grid and circular footprints in scaleFreeProve.py.
    n_grid = 33
    x = np.clip((lon / 360 + 0.5) * n_grid, 0, np.nextafter(n_grid, 0))
    y = np.clip((lat / 180 + 0.5) * n_grid, 0, np.nextafter(n_grid, 0))
    xi, yi = x.astype(int), y.astype(int)
    covered = (x - xi - 0.5) ** 2 + (y - yi - 0.5) ** 2 <= 0.25
    satellite = xi[covered] * n_grid + yi[covered]
    degrees = 4 + np.bincount(satellite, minlength=n_grid * n_grid)
    return degrees, satellite, covered


def run_attachment_proxy():
    """R1-1: test whether later attachment opportunities scale with prior degree."""
    lat, lon = _weighted_coordinates(30000, SEED)
    degrees, _, _ = _static_grid_degrees(lat, lon)
    half = len(lat) // 2
    before = 4 + np.bincount(_static_grid_degrees(lat[:half], lon[:half])[1], minlength=len(degrees))
    after = np.bincount(_static_grid_degrees(lat[half:], lon[half:])[1], minlength=len(degrees))
    kernel = []
    for degree in np.unique(before):
        group = before == degree
        kernel.append({"degree_before": int(degree), "satellites": int(group.sum()),
                       "mean_new_links": float(after[group].mean())})
    _write_csv(OUT / "attachment_proxy.csv", kernel)

    selected = [item for item in kernel if
                item["degree_before"] > 4 and item["satellites"] >= 5 and item["mean_new_links"] > 0]
    gain = np.log([item["mean_new_links"] for item in selected])
    total = stats.linregress(np.log([item["degree_before"] for item in selected]), gain)
    access = stats.linregress(np.log([item["degree_before"] - 4 for item in selected]), gain)
    fits = [
        {"predictor": "total_degree", "slope": total.slope, "slope_se": total.stderr,
         "r_squared": total.rvalue ** 2, "n_bins": len(selected)},
        {"predictor": "terrestrial_degree", "slope": access.slope, "slope_se": access.stderr,
         "r_squared": access.rvalue ** 2, "n_bins": len(selected)},
    ]
    _write_csv(OUT / "attachment_proxy_fit.csv", fits)

    x = np.array([item["degree_before"] - 4 for item in selected], dtype=float)
    y = np.array([item["mean_new_links"] for item in selected], dtype=float)
    fitted_x = np.geomspace(x.min(), x.max(), 200)
    fitted_y = np.exp(access.intercept) * fitted_x ** access.slope
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        ax.scatter(x, y, s=25, facecolors="white", edgecolors=BLUE,
                   linewidths=1.2, label="Simulation bins")
        ax.plot(fitted_x, fitted_y, color=RED, linewidth=1.8,
                label="Power-law fit")
        ax.set(xscale="log", yscale="log",
               xlabel=r"Prior terrestrial degree, $k_{\mathrm{pre}}$",
               ylabel=r"Mean new links per satellite, $\langle\Delta k\rangle$")
        ax.text(
            0.05, 0.95,
            rf"$\langle\Delta k\rangle={np.exp(access.intercept):.2f}"
            rf"k_{{\mathrm{{pre}}}}^{{{access.slope:.3f}}}$" "\n"
            rf"$\mathrm{{SE}}(\beta)={access.stderr:.3f},\ R^2={access.rvalue ** 2:.3f}$" "\n"
            rf"$n={len(selected)}$ bins",
            transform=ax.transAxes, va="top", ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.0},
        )
        ax.legend(frameon=True, loc="lower right")
        ax.grid(False)
        ax.tick_params(direction="in", width=0.8, length=3.5)
        ax.margins(x=0.06, y=0.08)
        _show_figure(fig)
    print("attachment proxy", "slope", round(access.slope, 3),
          "R^2", round(access.rvalue ** 2, 3), flush=True)
    return fits


def run_static_ablation():
    """R1-5: show Figure 5 beside a linear-scale uniform-population ablation."""
    rows, series = [], {}
    for label, coordinate_fn in (("city_population_marginals", _weighted_coordinates),
                                 ("surface_area_uniform", _uniform_coordinates)):
        lat, lon = coordinate_fn(30000, SEED)
        satellite_degrees, _, covered = _static_grid_degrees(lat, lon)
        # Figure 5 includes every covered terminal (degree one), not only satellites.
        all_degrees = np.concatenate((satellite_degrees,
                                      np.ones(int(covered.sum()), dtype=int)))
        counts = np.bincount(all_degrees)
        values = np.arange(1, len(counts))
        frequencies = counts[1:] / len(all_degrees)
        row = {
            "scenario": label,
            "n_generated": len(lat),
            "n_covered": int(covered.sum()),
            "n_satellites": len(satellite_degrees),
            "n_nodes": len(all_degrees),
            "mean_satellite_degree": float(satellite_degrees.mean()),
            "max_satellite_degree": int(satellite_degrees.max()),
        }
        rows.append(row)
        series[label] = (values, frequencies)
        print("static", label, "covered", row["n_covered"],
              "max degree", row["max_satellite_degree"], flush=True)
    _write_csv(OUT / "static_ablation.csv", rows)

    weighted_values, weighted_frequencies = series["city_population_marginals"]
    bins_end = int(np.log2(weighted_values.max())) + 1
    padded = np.pad(weighted_frequencies,
                    (0, 2 ** bins_end - weighted_values.max()))
    binned_values, binned_frequencies = [], []
    for index in range(bins_end):
        start = 2 ** index
        stop = 2 * start
        binned_values.append((start + stop - 1) / 2)
        binned_frequencies.append(padded[start - 1:stop - 1].mean())

    with plt.rc_context(PAPER_STYLE):
        fig, (population_ax, uniform_ax) = plt.subplots(1, 2, figsize=(13.2, 4.8))

        # Left: reproduce Figure 5, including its linear-scale inset.
        population_ax.loglog(weighted_values, weighted_frequencies, ".", color=GREY, alpha=0.5, label="linear")
        population_ax.loglog(binned_values, binned_frequencies, ".", color=BLUE, label="log binning")
        population_ax.set(xlabel="k'", ylabel="p(k')")
        population_ax.legend()
        inset = population_ax.inset_axes([0.08, 0.06, 0.34, 0.34])
        inset.plot(weighted_values, weighted_frequencies, color=GREY)
        inset.plot(binned_values, binned_frequencies, color=BLUE)

        # Right: uniform population, raw degree distribution on linear axes only.
        uniform_values, uniform_frequencies = series["surface_area_uniform"]
        uniform_ax.plot(uniform_values, uniform_frequencies, ".", color=GREY, alpha=0.5, label="linear")
        uniform_ax.set(xlabel="k'", ylabel="p(k')")
        uniform_ax.legend()
        detail = uniform_values > 1
        uniform_inset = uniform_ax.inset_axes([0.48, 0.18, 0.50, 0.42])
        uniform_inset.plot(uniform_values[detail], uniform_frequencies[detail], ".",
                           color=GREY, markersize=5)
        uniform_inset.set_xlim(2, uniform_values[detail].max() + 1)
        uniform_inset.set_ylim(0, uniform_frequencies[detail].max() * 1.08)
        uniform_inset.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
        _show_figure(fig)
    return rows


def _read_terrain(configuration, minute):
    path = ARCHIVE / configuration / "Graph" / "satTerrainGraph" / f"{minute * 60}_ms_satTerrainGraphgml"
    return nx.read_gml(path)


def _read_largest_terrain(configuration, minute):
    graph = _read_terrain(configuration, minute)
    return graph.subgraph(max(nx.connected_components(graph), key=len))


def _graph_row(graph, configuration, minute, source, population="original_all_nodes_degree_ge4"):
    satellites = {node for node in graph if SAT_NAME.fullmatch(str(node))}
    ground = [node for node in graph if node not in satellites]
    if population == "satellites_only":
        degrees = np.array([graph.degree(node) for node in satellites], dtype=int)
    else:
        degrees = np.array([degree for _, degree in graph.degree() if degree >= 4], dtype=int)
    access_edges = sum(1 for u, v in graph.edges if (u in satellites) != (v in satellites))
    row = {
        "configuration": configuration,
        "minute": minute,
        "source": source,
        "population": population,
        "n_satellite": len(satellites),
        "n_ground": len(ground),
        "access_edges": access_edges,
        "multi_ground_nodes": sum(graph.degree(node) > 1 for node in ground),
        "ground_nodes_degree_ge4": sum(graph.degree(node) >= 4 for node in ground),
    }
    row.update(evaluate_degrees(degrees, seed=SEED + minute + len(satellites)))
    return row, degrees


def _read_link_rows(path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            yield ast.literal_eval(line)


def _alternative_constellation():
    configuration = "48_24_550_55_5000_60ms"
    root = ARCHIVE / configuration
    graph = nx.Graph()
    graph.add_nodes_from(f"Sat{orbit}_{sat}" for orbit in range(48) for sat in range(24))
    for first, second, _ in _read_link_rows(root / "ISLLink" / "0_ms_GSLLink.txt"):
        graph.add_edge(first, second)
    for _, ground, satellite, *_ in _read_link_rows(root / "GSLLink" / "0_ms_GSLLink.txt"):
        graph.add_edge(ground, satellite)
    largest = graph.subgraph(max(nx.connected_components(graph), key=len))
    return _graph_row(largest, configuration, 0, "reconstructed_saved_links")


def _association_sensitivity_for_graph(graph):
    satellites = {node for node in graph if SAT_NAME.fullmatch(str(node))}
    isl_degree = Counter()
    for u, v in graph.edges:
        if u in satellites and v in satellites:
            isl_degree[u] += 1
            isl_degree[v] += 1
    candidates = {}
    raw_edges = set()
    path = ARCHIVE / SHELLS[0] / "GSLlink" / "0_ms_GSLLink.txt"
    for distance, ground, satellite, *_ in _read_link_rows(path):
        distance = float(distance)
        raw_edges.add((ground, satellite))
        if ground not in candidates or distance < candidates[ground][0]:
            candidates[ground] = (distance, satellite)
    graph_edges = {(v, u) if u in satellites else (u, v) for u, v in graph.edges if
                   (u in satellites) != (v in satellites)}
    if raw_edges != graph_edges:
        raise ValueError("Saved GSL candidates do not match saved terrain graph")
    selected = Counter(satellite for _, satellite in candidates.values())
    nearest_degrees = np.array([isl_degree[node] + selected[node] for node in satellites], dtype=int)
    original_degrees = np.array([graph.degree(node) for node in satellites], dtype=int)
    rows = []
    for label, degrees in (("stored_multi_association", original_degrees), ("nearest_saved_distance", nearest_degrees)):
        row = {"rule": label, "ground_nodes_with_candidate": len(candidates),
               "access_edges": len(raw_edges) if label.startswith("stored") else len(candidates)}
        row.update(evaluate_degrees(degrees, seed=SEED + len(rows)))
        rows.append(row)
    _write_csv(OUT / "association_sensitivity.csv", rows)
    return rows


def _hub_failure_sensitivity_for_graph(graph):
    """Count lost access, not routing performance, after t=0 satellite failures."""
    satellites = sorted(node for node in graph if SAT_NAME.fullmatch(str(node)))
    satellite_set = set(satellites)
    ground_neighbors = [set(graph.neighbors(node)) & satellite_set for node in graph if node not in satellite_set]
    ground_neighbors = [neighbors for neighbors in ground_neighbors if neighbors]
    access_degree = Counter(satellite for neighbors in ground_neighbors for satellite in neighbors)
    ordered = sorted(satellites, key=lambda node: (-graph.degree(node), node))
    rng = random.Random(SEED)
    rows = []

    def loss(removed):
        removed = set(removed)
        ground_lost = sum(neighbors <= removed for neighbors in ground_neighbors) / len(ground_neighbors)
        access_lost = sum(access_degree[node] for node in removed) / sum(access_degree.values())
        return ground_lost, access_lost

    for percent in (1, 5, 10):
        n_removed = max(1, round(len(satellites) * percent / 100))
        target_ground, target_access = loss(ordered[:n_removed])
        random_loss = np.array([loss(rng.sample(satellites, n_removed)) for _ in range(CI_REPS)])
        rows.append({
            "removed_percent": percent, "n_removed": n_removed,
            "target_ground_lost": target_ground, "random_ground_mean": random_loss[:, 0].mean(),
            "random_ground_q025": np.quantile(random_loss[:, 0], 0.025),
            "random_ground_q975": np.quantile(random_loss[:, 0], 0.975),
            "target_access_edges_lost": target_access, "random_access_edges_mean": random_loss[:, 1].mean(),
            "random_access_edges_q025": np.quantile(random_loss[:, 1], 0.025),
            "random_access_edges_q975": np.quantile(random_loss[:, 1], 0.975),
        })
    _write_csv(OUT / "hub_failure_sensitivity.csv", rows)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    x = np.array([row["removed_percent"] for row in rows])
    targeted = np.array([row["target_ground_lost"] for row in rows])
    random_mean = np.array([row["random_ground_mean"] for row in rows])
    ax.plot(x, targeted, "o-", color=BLUE, label="highest-degree satellites")
    ax.plot(x, random_mean, "s-", color=TEAL, label="random satellites")
    ax.fill_between(x, [row["random_ground_q025"] for row in rows],
                    [row["random_ground_q975"] for row in rows], color=TEAL, alpha=0.18)
    ax.set(xlabel="satellites removed (%)", ylabel="ground nodes losing all access links", ylim=(0, 1))
    ax.legend(frameon=False, fontsize=8)
    _show_figure(fig)
    return rows


def run_degree_population_check():
    """Definition check: compare the paper's degree population with satellites only."""
    graph = _read_largest_terrain(SHELLS[0], 0)
    original = _graph_row(graph, SHELLS[0], 0, "saved_gml")[0]
    satellites = _graph_row(graph, SHELLS[0], 0, "saved_gml", "satellites_only")[0]
    rows = [original, satellites]
    _write_csv(OUT / "degree_population_check.csv", rows)
    print("degree population check saved", flush=True)
    return rows


def run_association_sensitivity():
    """R1-6/R2-1: compare stored multi-association with nearest association."""
    rows = _association_sensitivity_for_graph(_read_largest_terrain(SHELLS[0], 0))
    print("association sensitivity saved", flush=True)
    return rows


def run_hub_failure_sensitivity():
    """R1-9: compare highest-degree and random satellite removals."""
    rows = _hub_failure_sensitivity_for_graph(_read_largest_terrain(SHELLS[0], 0))
    print("hub failure sensitivity saved", flush=True)
    return rows


def run_dynamic_snapshots():
    """R1-3/R1-10: fit saved shell and combined-network snapshots."""
    rows = []
    for minute in SNAPSHOT_MINUTES:
        graphs = []
        for configuration in SHELLS:
            graph = _read_largest_terrain(configuration, minute)
            graphs.append(graph)
            row, _ = _graph_row(graph, configuration, minute, "saved_gml")
            rows.append(row)
            print("dynamic", configuration, minute, "alpha", round(row["alpha"], 3), flush=True)
        combined = nx.compose_all(graphs)
        combined = combined.subgraph(max(nx.connected_components(combined), key=len))
        row, _ = _graph_row(combined, "three_shell_combined", minute, "composed_saved_gml")
        rows.append(row)
        print("dynamic", "three_shell_combined", minute, "alpha", round(row["alpha"], 3), flush=True)

    _write_csv(OUT / "dynamic_snapshot_statistics.csv", rows)
    return rows


def run_configuration_sensitivity():
    """R1-8: compare the saved t=0 topology configurations."""
    comparison = {}
    base_graph = _read_largest_terrain(SHELLS[0], 0)
    base_row, comparison[SHELLS[0]] = _graph_row(base_graph, SHELLS[0], 0, "saved_gml")

    alternative_row, alternative_degrees = _alternative_constellation()
    comparison[alternative_row["configuration"]] = alternative_degrees

    terminal_graph = _read_largest_terrain("72_22_550_55_1000_60ms", 0)
    terminal_row, terminal_degrees = _graph_row(
        terminal_graph, "72_22_550_55_1000_60ms", 0, "saved_gml"
    )
    comparison[terminal_row["configuration"]] = terminal_degrees
    rows = [base_row, alternative_row, terminal_row]
    _write_csv(OUT / "configuration_sensitivity.csv", rows)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for configuration, color, marker in (
            (SHELLS[0], BLUE, "o"),
            ("48_24_550_55_5000_60ms", TEAL, "s"),
            ("72_22_550_55_1000_60ms", GREEN, "^"),
    ):
        values, counts = np.unique(comparison[configuration], return_counts=True)
        ax.loglog(values, counts / len(comparison[configuration]), marker=marker, linestyle="none",
                  markersize=4, color=color, alpha=0.7, label=configuration.split("_60ms")[0])
    ax.set(xlabel="node degree (>=4)", ylabel="frequency")
    ax.legend(frameon=False, fontsize=8)
    _show_figure(fig)
    print("configuration sensitivity saved", flush=True)
    return rows


def run_periodicity():
    """R1-7: quantify periodicity using autocorrelation and spectral analysis."""
    rows = []
    fig = plt.figure(figsize=(8.5, 6.0))
    layout = fig.add_gridspec(2, 2, height_ratios=(1, 1.1))
    axes = (fig.add_subplot(layout[0, :]), fig.add_subplot(layout[1, 0]), fig.add_subplot(layout[1, 1]))
    for configuration, color in ((SHELLS[0], BLUE), (SHELLS[2], TEAL)):
        path = ARCHIVE / configuration / "CurveFit" / "poptData(power-law+binning).txt"
        data = np.loadtxt(path, delimiter=",", usecols=(0, 1))
        exponent = data[:, 1]
        if len(exponent) != 1440 or not np.isfinite(exponent).all():
            raise ValueError(f"Expected 1440 finite one-minute fitted exponents: {path}")
        centered = signal.detrend(exponent)
        autocorrelation = signal.correlate(centered, centered, mode="full", method="fft")[len(centered) - 1:]
        autocorrelation /= autocorrelation[0]
        frequency, power = signal.periodogram(centered, fs=1.0, window="hann", detrend=False, scaling="spectrum")
        usable = (frequency > 0) & (1 / np.maximum(frequency, 1e-12) >= 5) & (1 / np.maximum(frequency, 1e-12) <= 720)
        peaks, _ = signal.find_peaks(power)
        peaks = [index for index in peaks if usable[index]]
        peaks = sorted(peaks, key=lambda index: power[index], reverse=True)[:5]
        for rank, index in enumerate(peaks, 1):
            rows.append({"configuration": configuration, "rank": rank, "period_min": 1 / frequency[index],
                         "frequency_per_min": frequency[index], "spectral_power": power[index],
                         "acf_at_nearest_lag": autocorrelation[round(1 / frequency[index])]})
        axes[0].plot(np.arange(len(exponent)), exponent, color=color, linewidth=0.5, alpha=0.55,
                     label=configuration.split("_")[0] + " planes")
        axes[1].plot(np.arange(1, 241), autocorrelation[1:241], color=color, linewidth=0.9)
        axes[2].plot(1 / frequency[usable], power[usable], color=color, linewidth=1.1)
    axes[0].set(xlabel="time (min)", ylabel="L-M exponent b")
    axes[1].set(xlabel="lag (min)", ylabel="autocorrelation")
    axes[2].set(xlabel="period (min)", ylabel="spectral power", xlim=(5, 240), yscale="log")
    axes[0].legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    _write_csv(OUT / "periodicity_peaks.csv", rows)
    _show_figure(fig)
    return rows


def _self_check():
    rng = np.random.default_rng(17)
    sample = _sample_powerlaw(rng, 2000, 4, 2.5)
    assert sample.min() >= 4
    assert abs(_alpha_mle(sample, 4) - 2.5) < 0.3
    assert abs(fit_log_binned_powerlaw(sample, 4)["binned_lm_alpha"] - 2.5) < 0.3
    assert _static_grid_degrees(np.array([0.0]), np.array([0.0]))[0].sum() == 4 * 33 * 33 + 1
    assert len(_read_saved_degree_distribution(TABLE1_DATA / "GS169_CITY4000.txt")) == 3364


# Select one or more experiments here, then run this file directly in the IDE.
EXPERIMENTS_TO_RUN = [
    # run_attachment_proxy,        # R1-1
    # run_table1_refit,            # R1-3, R1-4
    # run_dynamic_snapshots,       # R1-3, R1-10
    run_static_ablation,           # R1-5
    # run_association_sensitivity, # R1-6, R2-1
    # run_periodicity,             # R1-7
    # run_configuration_sensitivity,  # R1-8
    # run_hub_failure_sensitivity, # R1-9
    # run_degree_population_check, # definition check
]


if __name__ == "__main__":
    # _self_check()
    for experiment in EXPERIMENTS_TO_RUN:
        print(f"\n=== Running {experiment.__name__} ===", flush=True)
        experiment()
    print(f"Saved supplementary outputs to {OUT}", flush=True)
