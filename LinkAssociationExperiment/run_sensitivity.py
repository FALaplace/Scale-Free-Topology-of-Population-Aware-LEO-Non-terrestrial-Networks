"""Run in an IDE. Read archived inputs; write only inside MinorRevision."""
import os
import sys
sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import ast
import csv
import hashlib
import json
import platform
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent
CODE = ROOT.parent / "code"
sys.path.insert(0, str(CODE))
import supplementary_experiments as original

EARTH_KM = 6371.0
ELEVATION = np.radians(55.0)
ELEVATION_TIE_TOL = 1e-12
MINUTES = tuple(range(0, 1440, 15))
CHECKPOINTS = (0, 360, 720, 1080)
SHELL_NAMES = ("Shell1", "Shell2", "Shell3")
NETWORKS = (*SHELL_NAMES, "Combined")
SIZES = (1584, 1600, 400)
OFFSETS = (0, 1584, 3184, 3584)
SCENARIOS = ("flat_multi", "flat_single", "sphere_multi", "sphere_single")
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
LABELS = ("Flat / multiple", "Flat / single", "Spherical / multiple", "Spherical / single")
SEED = 20261111


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def spherical_radius(altitude):
    return EARTH_KM * (np.arccos(EARTH_KM / (EARTH_KM + altitude) * np.cos(ELEVATION)) - ELEVATION)


def geometry_links(positions, ground):
    """Same great-circle metric in both arms; elevation from spherical geometry."""
    glat, glon = np.radians(ground).T
    links = {"flat": [], "sphere": []}
    for start in range(0, len(positions), 128):
        lat, lon = np.radians(positions[start:start + 128, :2]).T
        altitude = positions[start:start + 128, 2] - EARTH_KM
        hav = (np.sin((lat[:, None] - glat) / 2) ** 2
               + np.cos(lat[:, None]) * np.cos(glat) * np.sin((lon[:, None] - glon) / 2) ** 2)
        angle = 2 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
        for model, radius in (("flat", altitude / np.tan(ELEVATION)),
                              ("sphere", spherical_radius(altitude))):
            si, gi = np.nonzero(EARTH_KM * angle <= radius[:, None])
            psi = angle[si, gi]
            orbit_radius = EARTH_KM + altitude[si]
            elev = np.arctan2(orbit_radius * np.cos(psi) - EARTH_KM, orbit_radius * np.sin(psi))
            if model == "sphere":
                assert np.all(elev >= ELEVATION - 1e-12)
            links[model].append((si + start, gi, elev))
    return {key: tuple(np.concatenate([chunk[i] for chunk in chunks]) for i in range(3))
            for key, chunks in links.items()}


def select_single(satellite, ground, elevation):
    """Maximum elevation, then lowest fixed satellite ID, within each terminal."""
    if not len(ground):
        return satellite, ground
    maximum = np.full(int(ground.max()) + 1, -np.inf)
    np.maximum.at(maximum, ground, elevation)
    # Coincident archived positions must not be distinguished by floating-point noise.
    tied = elevation >= maximum[ground] - ELEVATION_TIE_TOL
    satellite, ground = satellite[tied], ground[tied]
    order = np.lexsort((satellite, ground))
    unique = np.r_[True, np.diff(ground[order]) != 0]
    take = order[unique]
    return satellite[take], ground[take]


def degree_metrics(values):
    values = np.asarray(values, dtype=float)
    total = float(values.sum())
    ordered = np.sort(values)
    mean = float(values.mean())
    n = len(values)
    return {
        "mean": mean, "cv": float(values.std() / mean) if mean else 0.0,
        "gini": float(2 * np.dot(np.arange(1, n + 1), ordered) / (n * total) - (n + 1) / n) if total else 0.0,
        "max": int(values.max()), "max_mean": float(values.max() / mean) if mean else 0.0,
        "top1_share": float(ordered[-max(1, int(np.ceil(0.01 * n))):].sum() / total) if total else 0.0,
        "zero_count": int(np.sum(values == 0)),
    }


def tail_job(job):
    network, scenario, minute, degrees, seed = job
    row = {"network": network, "scenario": scenario, "minute": minute,
           "population": "all_satellites_total_degree", "seed": seed}
    row.update(original.evaluate_degrees(degrees, seed=seed))
    return row


def self_check():
    for h in (550.0, 1110.0, 1130.0):
        psi = spherical_radius(h) / EARTH_KM
        el = np.arctan2((EARTH_KM + h) * np.cos(psi) - EARTH_KM, (EARTH_KM + h) * np.sin(psi))
        assert abs(el - ELEVATION) < 1e-12
        assert 0 < spherical_radius(h) < h / np.tan(ELEVATION)
    s, g = select_single(np.array([0, 1, 2, 0]), np.array([0, 0, 0, 1]), np.array([0.8, 1.0, 1.0, 0.9]))
    assert list(zip(s, g)) == [(1, 0), (0, 1)]
    assert select_single(np.array([0, 1]), np.array([0, 0]), np.array([1.0, 1.0 + 1e-14]))[0][0] == 0
    assert select_single(np.array([0, 1]), np.array([0, 0]), np.array([1.0, 1.0 + 1e-10]))[0][0] == 1
    assert len(select_single(np.array([], int), np.array([], int), np.array([]))[0]) == 0
    assert degree_metrics([4, 4, 4])["gini"] == 0
    assert np.isclose(degree_metrics([0, 0, 4])["gini"], 2 / 3)
    # A terminal cannot join disconnected shells after single association.
    graph = sparse.coo_matrix((np.ones(4), ([0, 2, 0, 2], [1, 3, 4, 5])), shape=(6, 6))
    assert connected_components(graph, directed=False)[0] == 2
    psi = spherical_radius(550) / EARTH_KM
    test_ground = np.array([[0, 0], [0, np.degrees(psi) * 0.999], [0, np.degrees(psi) * 1.001]])
    found = geometry_links(np.array([[0, 0, EARTH_KM + 550]]), test_ground)
    assert set(found["sphere"][1]) == {0, 1}
    assert set(found["flat"][1]) == {0, 1, 2}
    print("Self-checks passed", flush=True)


class RevisionExperiment:
    def __init__(self):
        self.input_dir = ROOT / "inputs"
        self.results_dir = ROOT / "results"
        self.fig_dir = ROOT / "figures"
        for folder in (self.input_dir, self.results_dir, self.fig_dir):
            folder.mkdir(exist_ok=True)

    def load_inputs(self):
        source = CODE / "rawdata" / "ground_stations_cities_5000.txt"
        raw = list(csv.reader(source.open(encoding="utf-8-sig")))
        assert len(raw) == 5001 and all(len(row) == 5 for row in raw)
        self.ground = np.array([[float(row[2]), float(row[3])] for row in raw])
        assert np.isfinite(self.ground).all()
        self.n_ground = len(raw)
        ground_rows = [dict(terminal_id=f"UT{i:05d}", source_id=row[0], city=row[1],
                            latitude_deg=float(row[2]), longitude_deg=float(row[3]), country=row[4])
                       for i, row in enumerate(raw)]
        write_csv(self.input_dir / "ground_terminals.csv", ground_rows)
        by_coordinate = {}
        for i, coord in enumerate(self.ground):
            by_coordinate.setdefault(tuple(coord), []).append(i)
        source_files = [{"path": str(source), "sha256": sha256(source), "role": "ground_records"}]
        self.positions = np.empty((len(MINUTES), sum(SIZES), 3))
        self.isl = []
        audit_rows, satellite_rows, label_mapping = [], [], set()
        for shell_index, (shell, folder) in enumerate(zip(SHELL_NAMES, original.SHELLS)):
            previous_isl, names = None, None
            offset = OFFSETS[shell_index]
            for ti, minute in enumerate(MINUTES):
                path = original.ARCHIVE / folder / "Graph" / "satTerrainGraph" / f"{minute * 60}_ms_satTerrainGraphgml"
                graph = nx.read_gml(path)
                current = sorted(n for n in graph if original.SAT_NAME.fullmatch(str(n)))
                if names is None:
                    names = current
                    assert len(names) == SIZES[shell_index]
                    satellite_rows.extend(dict(satellite_index=offset + i, shell=shell, archived_name=n) for i, n in enumerate(names))
                assert current == names
                indices = {n: offset + i for i, n in enumerate(names)}
                xyz = np.array([[graph.nodes[n]["Lat"], graph.nodes[n]["Long"], graph.nodes[n]["Alt"] / 1000] for n in names])
                assert np.isfinite(xyz).all() and np.all(xyz[:, 2] > EARTH_KM)
                self.positions[ti, offset:OFFSETS[shell_index + 1]] = xyz
                isl = sorted(tuple(sorted((indices[a], indices[b]))) for a, b in graph.edges if a in indices and b in indices)
                assert len(isl) == 2 * len(names)
                assert np.all(np.bincount(np.array(isl).ravel() - offset, minlength=len(names)) == 4)
                if previous_isl is None:
                    previous_isl = isl
                    self.isl.extend(isl)
                assert previous_isl == isl
                source_files.append({"path": str(path), "sha256": sha256(path), "role": "archived_graph"})
                if minute in CHECKPOINTS:
                    links_path = original.ARCHIVE / folder / "GSLlink" / f"{minute * 60}_ms_GSLLink.txt"
                    legacy, records, labels_coordinates = set(), set(), {}
                    repeated_rows, expected_multiplicity = {}, {}
                    max_distance_difference = 0.0
                    for line in links_path.open(encoding="utf-8"):
                        distance, label, sat, lat, lon, slat, slon = ast.literal_eval(line)
                        candidates = by_coordinate[(float(lat), float(lon))]
                        matches = candidates if len(candidates) == 1 else [i for i in candidates if raw[i][1] == label or re.fullmatch(re.escape(label) + r"\d+", raw[i][1])]
                        assert matches, (label, lat, lon, candidates)
                        key = (label, sat, lat, lon)
                        occurrence = repeated_rows.get(key, 0)
                        assert occurrence < len(matches), (key, occurrence, matches)
                        gi = matches[occurrence]
                        repeated_rows[key] = occurrence + 1
                        expected_multiplicity[key] = len(matches)
                        label_mapping.add((label, f"UT{gi:05d}", raw[gi][1], float(lat), float(lon)))
                        si = indices[sat] - offset
                        assert np.allclose(xyz[si, :2], [slat, slon], rtol=0, atol=1e-9)
                        legacy.add((label, sat))
                        records.add(si * self.n_ground + gi)
                        labels_coordinates.setdefault(label, set()).add((lat, lon))
                        la, lo, sa, so = np.radians([lat, lon, slat, slon])
                        hav = np.sin((la - sa) / 2) ** 2 + np.cos(la) * np.cos(sa) * np.sin((lo - so) / 2) ** 2
                        d = 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
                        max_distance_difference = max(max_distance_difference, abs(d - float(distance) / 1000))
                    assert repeated_rows == expected_multiplicity
                    expected = {(b, a) if a in indices else (a, b) for a, b in graph.edges if (a in indices) != (b in indices)}
                    assert legacy == expected, "Raw candidate links disagree with archived graph"
                    rebuilt = geometry_links(xyz, self.ground)["flat"]
                    rebuilt_ids = set((rebuilt[0] * self.n_ground + rebuilt[1]).tolist())
                    audit_rows.append({"shell": shell, "minute": minute, "archived_ground_nodes": len(graph) - len(names),
                                       "new_ground_records": self.n_ground, "archived_access_edges": len(legacy),
                                       "record_level_wgs84_access_edges": len(records), "flat_greatcircle_access_edges": len(rebuilt_ids),
                                       "metric_only_added_edges": len(rebuilt_ids - records), "metric_only_removed_edges": len(records - rebuilt_ids),
                                       "covered_legacy_labels_with_multiple_locations": sum(len(v) > 1 for v in labels_coordinates.values()),
                                       "max_wgs84_vs_sphere_distance_difference_km": max_distance_difference,
                                       "raw_links_match_archived_graph": True})
                    source_files.append({"path": str(links_path), "sha256": sha256(links_path), "role": "archived_candidate_links"})
                if ti % 16 == 0:
                    print(f"Read {shell}: minute {minute}", flush=True)
        self.isl = np.array(self.isl, dtype=int)
        np.savez_compressed(self.input_dir / "satellite_snapshots.npz", minutes=np.array(MINUTES),
                            latitude_longitude_radius_km=self.positions, isl_edges=self.isl)
        write_csv(self.input_dir / "satellite_ids.csv", satellite_rows)
        write_csv(self.input_dir / "terminal_label_mapping.csv", [dict(zip(
            ("archived_label", "terminal_id", "current_label", "latitude_deg", "longitude_deg"), item))
            for item in sorted(label_mapping)])
        write_csv(self.results_dir / "input_audit.csv", audit_rows)
        self.manifest = {"seed": SEED, "minutes": MINUTES, "checkpoints": CHECKPOINTS, "earth_radius_km": EARTH_KM,
                         "minimum_elevation_deg": 55, "terminal_count": self.n_ground, "satellite_counts": SIZES,
                         "elevation_tie_tolerance_rad": ELEVATION_TIE_TOL,
                         "bootstrap_seed_policy": "Identical paired degree arrays within a network/snapshot share the seed of the first scenario in SCENARIOS order.",
                         "node_identity": "one node per archived coordinate record; no city-name merging",
                         "distance_metric": "same spherical great-circle distance in both main arms",
                         "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                                      "networkx": nx.__version__, "matplotlib": matplotlib.__version__},
                         "runner_sha256": sha256(__file__), "statistics_source_sha256": sha256(CODE / "supplementary_experiments.py"),
                         "sources": source_files}
        (self.input_dir / "manifest.json").write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def run_networks(self):
        rows, self.degrees = [], {}
        for ni, network in enumerate(NETWORKS):
            ns = SIZES[ni] if ni < 3 else sum(SIZES)
            for scenario in SCENARIOS:
                self.degrees[f"{network}__{scenario}__satellite"] = np.empty((len(MINUTES), ns), dtype=np.int32)
                self.degrees[f"{network}__{scenario}__ground"] = np.empty((len(MINUTES), self.n_ground), dtype=np.int32)
        for ti, minute in enumerate(MINUTES):
            links = geometry_links(self.positions[ti], self.ground)
            for ni, network in enumerate(NETWORKS):
                lo, hi = (OFFSETS[ni], OFFSETS[ni + 1]) if ni < 3 else (0, sum(SIZES))
                ns = hi - lo
                isl = self.isl[(self.isl[:, 0] >= lo) & (self.isl[:, 0] < hi)] - lo
                previous = {}
                for scenario in SCENARIOS:
                    model, association = scenario.split("_")
                    si, gi, el = links[model]
                    keep = (si >= lo) & (si < hi)
                    si, gi, el = si[keep] - lo, gi[keep], el[keep]
                    candidate_covered = np.unique(gi).size
                    if association == "single":
                        si, gi = select_single(si, gi, el)
                        assert len(gi) == candidate_covered
                    edge_ids = set((si * self.n_ground + gi).tolist())
                    if model == "sphere":
                        assert edge_ids <= previous[f"flat_{association}"]
                    previous[scenario] = edge_ids
                    access = np.bincount(si, minlength=ns)
                    ground = np.bincount(gi, minlength=self.n_ground)
                    satellite = access + 4
                    assert access.sum() == ground.sum() == len(si)
                    if association == "single":
                        assert ground.max() <= 1
                    a, b = np.r_[isl[:, 0], si], np.r_[isl[:, 1], ns + gi]
                    adjacency = sparse.coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(ns + self.n_ground,) * 2)
                    count, labels = connected_components(adjacency, directed=False)
                    sizes = np.bincount(labels)
                    largest = labels == np.argmax(sizes)
                    all_degrees = np.r_[satellite, ground]
                    assert all_degrees.sum() == 2 * (len(isl) + len(si))
                    selected = all_degrees[largest & (all_degrees >= 4)]
                    row = {"network": network, "scenario": scenario, "minute": minute, "satellites": ns,
                           "ground_records": self.n_ground, "isl_edges": len(isl), "access_edges": len(si),
                           "covered_ground": int(np.sum(ground > 0)), "multi_access_ground": int(np.sum(ground > 1)),
                           "all_components_including_isolates": int(count), "satellite_components": int(np.unique(labels[:ns]).size),
                           "lcc_nodes": int(largest.sum()), "lcc_satellites": int(largest[:ns].sum()),
                           "lcc_ground": int(largest[ns:].sum()), "paper_population_n": len(selected),
                           "paper_population_ground_n": int(np.sum(largest[ns:] & (ground >= 4)))}
                    for prefix, values in (("sat", satellite), ("access", access), ("paper", selected)):
                        row.update({f"{prefix}_{k}": v for k, v in degree_metrics(values).items()})
                    rows.append(row)
                    self.degrees[f"{network}__{scenario}__satellite"][ti] = satellite
                    self.degrees[f"{network}__{scenario}__ground"][ti] = ground
            if ti % 8 == 0:
                print(f"Four-arm network analysis: minute {minute}", flush=True)
        self.rows = rows
        write_csv(self.results_dir / "snapshot_statistics.csv", rows)
        np.savez_compressed(self.results_dir / "node_degrees.npz", minutes=np.array(MINUTES), **self.degrees)
        self.summarize()

    def summarize(self):
        summary, comparisons = [], []
        keys = ("access_edges", "covered_ground", "sat_cv", "sat_gini", "sat_top1_share", "sat_max_mean",
                "access_cv", "access_gini", "access_top1_share", "satellite_components", "lcc_satellites", "paper_cv")
        for network in NETWORKS:
            for scenario in SCENARIOS:
                selected = [r for r in self.rows if r["network"] == network and r["scenario"] == scenario]
                item = {"network": network, "scenario": scenario, "snapshots": len(selected)}
                for key in keys:
                    values = np.array([r[key] for r in selected])
                    item.update({key + "_median": float(np.median(values)), key + "_min": float(values.min()), key + "_max": float(values.max())})
                summary.append(item)
            for base, changed, effect in (("flat_multi", "flat_single", "association_flat"),
                                          ("sphere_multi", "sphere_single", "association_sphere"),
                                          ("flat_multi", "sphere_multi", "geometry_multi"),
                                          ("flat_single", "sphere_single", "geometry_single")):
                for ti, minute in enumerate(MINUTES):
                    old = next(r for r in self.rows if r["network"] == network and r["scenario"] == base and r["minute"] == minute)
                    new = next(r for r in self.rows if r["network"] == network and r["scenario"] == changed and r["minute"] == minute)
                    x = self.degrees[f"{network}__{base}__satellite"][ti]
                    y = self.degrees[f"{network}__{changed}__satellite"][ti]
                    item = {"network": network, "effect": effect, "minute": minute,
                            "satellite_degree_spearman": float(stats.spearmanr(x, y).statistic),
                            "satellite_degree_wasserstein": float(stats.wasserstein_distance(x, y))}
                    for key in keys:
                        item[key + "_change"] = new[key] - old[key]
                        item[key + "_relative_change_pct"] = 100 * (new[key] / old[key] - 1) if old[key] else float("nan")
                    comparisons.append(item)
        write_csv(self.results_dir / "summary_statistics.csv", summary)
        write_csv(self.results_dir / "paired_comparisons.csv", comparisons)
        geometry = []
        for h in (550., 1110., 1130.):
            flat, sphere = h / np.tan(ELEVATION), spherical_radius(h)
            geometry.append({"height_km": h, "flat_radius_km": flat, "sphere_radius_km": sphere,
                             "flat_overestimate_pct": 100 * (flat / sphere - 1),
                             "footprint_area_reduction_pct": 100 * (1 - (1 - np.cos(sphere / EARTH_KM)) / (1 - np.cos(flat / EARTH_KM)))})
        write_csv(self.results_dir / "coverage_geometry.csv", geometry)

    def run_tail_tests(self):
        jobs, sample_seeds = [], {}
        for ni, network in enumerate(NETWORKS):
            for si, scenario in enumerate(SCENARIOS):
                for minute in CHECKPOINTS:
                    degrees = self.degrees[f"{network}__{scenario}__satellite"][MINUTES.index(minute)]
                    # Identical paired samples must not differ solely through Monte Carlo noise.
                    seed = sample_seeds.setdefault((network, minute, degrees.tobytes()), SEED + ni * 10000 + si * 2000 + minute)
                    jobs.append((network, scenario, minute, degrees, seed))
        rows = []
        with ProcessPoolExecutor(max_workers=4) as executor:
            pending = [executor.submit(tail_job, job) for job in jobs]
            for future in as_completed(pending):
                row = future.result()
                rows.append(row)
                write_csv(self.results_dir / "tail_tests.csv", sorted(rows, key=lambda r: (r["network"], r["scenario"], r["minute"])))
                print(f"Tail test {len(rows)}/{len(jobs)}: {row['network']} {row['scenario']} {row['minute']}, p={row['p_gof']:.2f}", flush=True)

    def plot(self):
        if not hasattr(self, "degrees"):
            loaded = np.load(self.results_dir / "node_degrees.npz")
            self.degrees = {k: loaded[k] for k in loaded.files if k != "minutes"}
        if not hasattr(self, "rows"):
            self.rows = list(csv.DictReader((self.results_dir / "snapshot_statistics.csv").open(encoding="utf-8")))
        style = {"font.family": "DejaVu Serif", "font.size": 10, "axes.labelsize": 10,
                 "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42}
        with plt.rc_context(style):
            fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.3), layout="constrained")
            for ni, network in enumerate(NETWORKS):
                for si, scenario in enumerate(SCENARIOS):
                    degree = self.degrees[f"{network}__{scenario}__satellite"][0]
                    values, counts = np.unique(degree, return_counts=True)
                    ccdf = np.cumsum(counts[::-1])[::-1] / len(degree)
                    axes[0, ni].step(values, ccdf, where="post", color=COLORS[si], linestyle=("-", "--", "-.", ":")[si], label=LABELS[si])
                    rows = [r for r in self.rows if r["network"] == network and r["scenario"] == scenario]
                    axes[1, ni].plot([float(r["minute"]) for r in rows], [float(r["sat_cv"]) for r in rows],
                                     color=COLORS[si], linestyle=("-", "--", "-.", ":")[si], linewidth=1.3)
                axes[0, ni].set(xscale="log", yscale="log", xlabel="Satellite total degree, k", title=f"({chr(97 + ni)}) {network}")
                axes[1, ni].set(xlabel="Time (min)", title=f"({chr(101 + ni)}) {network}", xlim=(0, 1440))
                axes[1, ni].set_xticks([0, 360, 720, 1080, 1440])
            axes[0, 0].set_ylabel("Pr(K ≥ k), t = 0")
            axes[1, 0].set_ylabel("Satellite-degree CV")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
            for suffix in ("png", "pdf"):
                fig.savefig(self.fig_dir / f"Figure_R1_four_arm_comparison.{suffix}", dpi=300)
            plt.close(fig)
            fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), layout="constrained")
            for ni, network in enumerate(NETWORKS):
                base = [r for r in self.rows if r["network"] == network and r["scenario"] == "flat_multi"]
                changed = [r for r in self.rows if r["network"] == network and r["scenario"] == "sphere_multi"]
                for ax, metric, ylabel in zip(axes, ("access_edges", "sat_cv", "sat_top1_share"),
                                             ("Access-edge change (%)", "Satellite-degree CV change (%)", "Top-1% degree-share change (%)")):
                    delta = np.array([100 * (float(b[metric]) / float(a[metric]) - 1) for a, b in zip(base, changed)])
                    ax.plot(MINUTES, delta, label=network, color=COLORS[ni], linewidth=1.3)
                    ax.axhline(0, color="0.65", linewidth=0.6)
                    ax.set(xlabel="Time (min)", ylabel=ylabel, xlim=(0, 1440))
                    ax.set_xticks([0, 360, 720, 1080, 1440])
            for i, ax in enumerate(axes):
                ax.set_title(f"({chr(97 + i)})")
            fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=4, frameon=False)
            for suffix in ("png", "pdf"):
                fig.savefig(self.fig_dir / f"Figure_R2_spherical_coverage_effect.{suffix}", dpi=300)
            plt.close(fig)


if __name__ == "__main__":
    self_check()
    experiment = RevisionExperiment()
    experiment.load_inputs()
    experiment.run_networks()
    experiment.plot()
    experiment.run_tail_tests()
    print("Completed all network comparisons and 64 tail tests.", flush=True)
