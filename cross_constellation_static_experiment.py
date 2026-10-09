"""R1-8: static cross-constellation sensitivity experiment.

Run this file directly in an IDE. It reuses the paper's saved three-shell
Starlink topology at t=0, rebuilds the archived Kuiper, Telesat, and OneWeb
topologies with the same ground terminals and coverage rule, writes the fitted
statistics to CSV, saves the comparison figure, and shows it interactively.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from supplementary_experiments import (
    BLUE,
    GREEN,
    OUT,
    PAPER_STYLE,
    RED,
    SAT_NAME,
    SEED,
    SHELLS,
    TEAL,
    _read_largest_terrain,
    _write_csv,
    evaluate_degrees,
)


HERE = Path(__file__).resolve().parent
CONSTELLATION_ROOT = Path(r"H:\PaperSubmit\IEEE2023\IEEE-program\constellation")
GROUND_FILE = HERE / "rawdata" / "ground_stations_cities_5000.txt"

CONSTELLATIONS_TO_RUN = ("Starlink", "Kuiper", "TeleSat", "OneWeb")
SNAPSHOT_INDEX = 0
GROUND_NODE_COUNT = 5000
MIN_ELEVATION_DEG = 55.0
EARTH_RADIUS_KM = 6371.0
SATELLITE_CHUNK_SIZE = 128
PAPER_STARLINK_CONFIGURATION = {
    "layers": ((72, 22), (32, 50), (8, 50)),
    "altitudes_km": (550.0, 1110.0, 1130.0),
    "inclinations_deg": (53.0, 53.8, 74.0),
    "isl_policy": "four_per_satellite",
    "source": "paper_saved_three_shell_gml",
}
HAS_ISL = {"Kuiper": True, "TeleSat": True, "OneWeb": False}
DISPLAY_NAMES = {
    "Starlink": "Starlink",
    "Kuiper": "Kuiper",
    "TeleSat": "Telesat",
    "OneWeb": "OneWeb",
}
COLORS = (BLUE, TEAL, RED, GREEN)
MARKERS = ("o", "s", "^", "D")
LINESTYLES = ("-", "--", "-.", ":")
FIGURE_STEM = "cross_constellation_degree_distribution"


def _read_configuration(root):
    lines = [line.strip() for line in (root / "con_info.txt").read_text().splitlines() if line.strip()]
    if len(lines) != 6:
        raise ValueError(f"Expected six lines in {root / 'con_info.txt'}")
    return {
        "layers": tuple(tuple(map(int, line.split(","))) for line in lines[:2]),
        "altitudes_km": tuple(map(float, lines[2].split(","))),
        "inclinations_deg": tuple(map(float, lines[3].split(","))),
        "phasing": tuple(map(float, lines[4].split(","))),
        "half_cone_deg": float(lines[5]),
    }


def _load_constellation(name):
    root = CONSTELLATION_ROOT / name
    configuration = _read_configuration(root)
    data = np.load(root / "LLA&LLR.npy", mmap_mode="r")
    with (root / "satdict.pkl").open("rb") as file:
        satellite_index = pickle.load(file)
    satellite_names = [
        node for node, _ in sorted(satellite_index.items(), key=lambda item: item[1])
    ]
    expected_names = [
        f"Sat_{layer}_{plane}_{satellite}"
        for layer, (planes, satellites_per_plane) in enumerate(configuration["layers"], start=1)
        for plane in range(planes)
        for satellite in range(satellites_per_plane)
    ]
    if satellite_names != expected_names or data.shape[0] != len(expected_names):
        raise ValueError(f"{name}: constellation metadata and trajectory data do not match")
    if data.ndim != 3 or data.shape[2] != 13:
        raise ValueError(f"{name}: expected trajectory shape (satellite, time, 13), got {data.shape}")
    return configuration, data, satellite_names


def _load_ground_nodes(limit=GROUND_NODE_COUNT):
    rows = [line.rstrip("\n").split(",") for line in GROUND_FILE.open(encoding="utf-8")]
    if len(rows) < limit or any(len(row) != 5 for row in rows[:limit]):
        raise ValueError(f"Invalid ground-terminal data in {GROUND_FILE}")
    rows = rows[:limit]
    node_ids = [f"UT_{index:05d}" for index in range(limit)]
    latitudes = np.array([float(row[2]) for row in rows])
    longitudes = np.array([float(row[3]) for row in rows])
    attributes = [
        (node, {"kind": "ground", "city": row[1], "lat": float(row[2]), "lon": float(row[3])})
        for node, row in zip(node_ids, rows)
    ]
    return node_ids, latitudes, longitudes, attributes


def _add_four_isl_edges(graph, configuration):
    for layer, (planes, satellites_per_plane) in enumerate(configuration["layers"], start=1):
        for plane in range(planes):
            for satellite in range(satellites_per_plane):
                node = f"Sat_{layer}_{plane}_{satellite}"
                graph.add_edge(
                    node,
                    f"Sat_{layer}_{plane}_{(satellite + 1) % satellites_per_plane}",
                    kind="ISL",
                )
                graph.add_edge(
                    node,
                    f"Sat_{layer}_{(plane + 1) % planes}_{satellite}",
                    kind="ISL",
                )


def _add_ground_links(graph, satellite_names, data, snapshot_index, ground_ids, ground_lat, ground_lon):
    satellite_lat = np.radians(np.asarray(data[:, snapshot_index, 7], dtype=float))
    satellite_lon = np.radians(np.asarray(data[:, snapshot_index, 8], dtype=float))
    satellite_altitude = np.asarray(data[:, snapshot_index, 9], dtype=float)
    ground_lat = np.radians(ground_lat)
    ground_lon = np.radians(ground_lon)
    footprint_radius = satellite_altitude / np.tan(np.radians(MIN_ELEVATION_DEG))
    access_edges = 0

    for start in range(0, len(satellite_names), SATELLITE_CHUNK_SIZE):
        stop = min(start + SATELLITE_CHUNK_SIZE, len(satellite_names))
        sat_lat = satellite_lat[start:stop, None]
        sat_lon = satellite_lon[start:stop, None]
        delta_lat = ground_lat[None, :] - sat_lat
        delta_lon = ground_lon[None, :] - sat_lon
        haversine = (
            np.sin(delta_lat / 2) ** 2
            + np.cos(sat_lat) * np.cos(ground_lat)[None, :] * np.sin(delta_lon / 2) ** 2
        )
        surface_distance = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
        satellite_indices, ground_indices = np.where(
            surface_distance <= footprint_radius[start:stop, None]
        )
        edges = [
            (satellite_names[start + int(satellite)], ground_ids[int(ground)], {"kind": "GSL"})
            for satellite, ground in zip(satellite_indices, ground_indices)
        ]
        graph.add_edges_from(edges)
        access_edges += len(edges)
    return access_edges


def build_static_graph(name, snapshot_index=SNAPSHOT_INDEX, include_isl=True):
    configuration, data, satellite_names = _load_constellation(name)
    if not 0 <= snapshot_index < data.shape[1]:
        raise ValueError(f"{name}: snapshot {snapshot_index} is outside 0..{data.shape[1] - 1}")

    graph = nx.Graph()
    graph.add_nodes_from(
        (
            node,
            {
                "kind": "satellite",
                "lat": float(data[index, snapshot_index, 7]),
                "lon": float(data[index, snapshot_index, 8]),
                "alt_km": float(data[index, snapshot_index, 9]),
            },
        )
        for index, node in enumerate(satellite_names)
    )
    if include_isl:
        _add_four_isl_edges(graph, configuration)
        if graph.number_of_edges() != 2 * len(satellite_names):
            raise AssertionError(f"{name}: unexpected number of four-ISL edges")
        if any(graph.degree(node) != 4 for node in satellite_names):
            raise AssertionError(f"{name}: the four-ISL construction is not degree four")

    ground_ids, ground_lat, ground_lon, ground_attributes = _load_ground_nodes()
    graph.add_nodes_from(ground_attributes)
    access_edges = _add_ground_links(
        graph,
        satellite_names,
        data,
        snapshot_index,
        ground_ids,
        ground_lat,
        ground_lon,
    )
    if access_edges == 0:
        raise AssertionError(f"{name}: no satellite-terminal links were generated")
    configuration = dict(configuration)
    configuration["isl_policy"] = "four_per_satellite" if include_isl else "none"
    configuration["source"] = "archived_constellation_trajectory"
    return graph, set(satellite_names), configuration, access_edges


def build_paper_starlink_graph(snapshot_minute=SNAPSHOT_INDEX):
    graphs = [_read_largest_terrain(shell, snapshot_minute) for shell in SHELLS]
    graph = nx.compose_all(graphs)
    graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
    satellites = {node for node in graph if SAT_NAME.fullmatch(str(node))}
    expected_satellites = sum(planes * satellites_per_plane for planes, satellites_per_plane in PAPER_STARLINK_CONFIGURATION["layers"])
    if len(satellites) != expected_satellites:
        raise AssertionError(
            f"Starlink: expected {expected_satellites} satellites, found {len(satellites)}"
        )
    access_edges = sum(
        1 for u, v in graph.edges if (u in satellites) != (v in satellites)
    )
    if access_edges == 0:
        raise AssertionError("Starlink: no satellite-terminal links in saved topology")
    return graph, satellites, PAPER_STARLINK_CONFIGURATION, access_edges


def _analyse_graph(name, graph, satellites, configuration, access_edges, snapshot_index, seed):
    largest_nodes = max(nx.connected_components(graph), key=len)
    largest = graph.subgraph(largest_nodes)
    ground_nodes = [node for node in largest if node not in satellites]
    degree_threshold = 1 if configuration["isl_policy"] == "none" else 4
    degrees = np.array(
        [degree for _, degree in largest.degree() if degree >= degree_threshold], dtype=int
    )
    if not len(degrees):
        raise AssertionError(
            f"{name}: no nodes with degree >= {degree_threshold} in the largest component"
        )
    top_count = max(1, int(np.ceil(0.01 * len(degrees))))
    mean_degree = float(degrees.mean())
    layers = [f"{planes}x{satellites_per_plane}" for planes, satellites_per_plane in configuration["layers"]]
    isl_edges = sum(u in satellites and v in satellites for u, v in graph.edges)
    row = {
        "constellation": name,
        "snapshot_minute": snapshot_index,
        "source": configuration["source"],
        "layer_1": layers[0],
        "layer_2": layers[1],
        "layer_3": layers[2] if len(layers) > 2 else "",
        "altitudes_km": "/".join(f"{value:g}" for value in configuration["altitudes_km"]),
        "inclinations_deg": "/".join(f"{value:g}" for value in configuration["inclinations_deg"]),
        "isl_policy": configuration["isl_policy"],
        "isl_edges": isl_edges,
        "satellites": len(satellites),
        "ground_terminals": GROUND_NODE_COUNT,
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "components": nx.number_connected_components(graph),
        "largest_component_nodes": largest.number_of_nodes(),
        "largest_component_ground": len(ground_nodes),
        "access_edges": access_edges,
        "multi_access_ground": sum(largest.degree(node) > 1 for node in ground_nodes),
        "minimum_elevation_deg": MIN_ELEVATION_DEG,
        "degree_population": f"largest_component_all_nodes_degree_ge{degree_threshold}",
        "mean_degree": mean_degree,
        "degree_cv": float(degrees.std() / mean_degree),
        "max_degree": int(degrees.max()),
        "max_to_mean_degree": float(degrees.max() / mean_degree),
        "top_1_percent_degree_share": float(np.sort(degrees)[-top_count:].sum() / degrees.sum()),
    }
    row.update(evaluate_degrees(degrees, seed=seed))
    return row, degrees


def _log_binned_distribution(degrees):
    centers, densities = [], []
    start = int(degrees.min())
    while start <= degrees.max():
        stop = 2 * start
        count = np.count_nonzero((degrees >= start) & (degrees < stop))
        if count:
            centers.append((start + stop - 1) / 2)
            densities.append(count / (len(degrees) * (stop - start)))
        start = stop
    return np.asarray(centers), np.asarray(densities)


def _plot_distributions(results, degrees_by_name, show):
    label_size = 18
    tick_size = 15
    style = {**PAPER_STYLE, "font.family": "Times New Roman", "mathtext.fontset": "stix"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)
        axes = axes.ravel()
        for panel, ax, row, color, marker, linestyle in zip(
            "abcd", axes, results, COLORS, MARKERS, LINESTYLES
        ):
            name = row["constellation"]
            degrees = degrees_by_name[name]
            values, counts = np.unique(degrees, return_counts=True)
            ax.loglog(values, counts / len(degrees), ".", color=color, alpha=0.18, markersize=3)
            centers, densities = _log_binned_distribution(degrees)
            ax.loglog(
                centers,
                densities,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=5,
                color=color,
            )
            ax.set_title(f"({panel}) {DISPLAY_NAMES[name]}", fontsize=label_size, pad=6)
            ax.set_xlabel(r"Node degree, $k$", fontsize=label_size)
            ax.set_ylabel(r"Probability density, $p(k)$", fontsize=label_size)
            ax.tick_params(which="both", direction="out", top=False, right=False,
                           labelsize=tick_size, labelleft=True, labelbottom=True)
            ax.grid(True, alpha=0.3)

        for ax in axes[len(results):]:
            ax.set_visible(False)
        fig.tight_layout()
        # OUT.mkdir(exist_ok=True)
        # fig.savefig(OUT / f"{FIGURE_STEM}.png", dpi=300, bbox_inches="tight")
        # fig.savefig(OUT / f"{FIGURE_STEM}.pdf", bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)


def run_cross_constellation_static_experiment(
    constellations=CONSTELLATIONS_TO_RUN,
    snapshot_index=SNAPSHOT_INDEX,
    show=True,
):
    rows = []
    degrees_by_name = {}
    for offset, name in enumerate(constellations):
        print(f"Building {name} snapshot {snapshot_index}...", flush=True)
        if name == "Starlink":
            graph, satellites, configuration, access_edges = build_paper_starlink_graph(snapshot_index)
        else:
            graph, satellites, configuration, access_edges = build_static_graph(
                name,
                snapshot_index,
                include_isl=HAS_ISL[name],
            )
        row, degrees = _analyse_graph(
            name,
            graph,
            satellites,
            configuration,
            access_edges,
            snapshot_index,
            SEED + offset,
        )
        rows.append(row)
        degrees_by_name[name] = degrees
        print(
            f"{name}: nodes={row['largest_component_nodes']}, "
            f"CV={row['degree_cv']:.3f}, max/mean={row['max_to_mean_degree']:.3f}, "
            f"top1%={row['top_1_percent_degree_share']:.3f}",
            flush=True,
        )

    _write_csv(OUT / "cross_constellation_static_statistics.csv", rows)
    _plot_distributions(rows, degrees_by_name, show)
    # print(f"Statistics saved to {OUT / 'cross_constellation_static_statistics.csv'}", flush=True)
    # print(f"Figure saved to {OUT / (FIGURE_STEM + '.pdf')}", flush=True)
    return rows


if __name__ == "__main__":
    run_cross_constellation_static_experiment()
