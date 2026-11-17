"""Verify the saved experiment, including an independent Cartesian geometry check."""
import sys
sys.dont_write_bytecode = True
import csv
import json
from pathlib import Path

import run_sensitivity as run
import numpy as np


def read_csv(name):
    return list(csv.DictReader((run.ROOT / "results" / name).open(encoding="utf-8")))


def unit_vectors(latlon):
    lat, lon = np.radians(latlon).T
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def verify():
    run.self_check()
    manifest = json.loads((run.ROOT / "inputs" / "manifest.json").read_text(encoding="utf-8"))
    for source in manifest["sources"]:
        assert run.sha256(source["path"]) == source["sha256"], source["path"]
    assert run.sha256(run.__file__) == manifest["runner_sha256"]
    assert run.sha256(run.CODE / "supplementary_experiments.py") == manifest["statistics_source_sha256"]
    saved = np.load(run.ROOT / "results" / "node_degrees.npz")
    inputs = np.load(run.ROOT / "inputs" / "satellite_snapshots.npz")
    rows = read_csv("snapshot_statistics.csv")
    assert len(rows) == 96 * 4 * 4
    assert len({(r["network"], r["scenario"], r["minute"]) for r in rows}) == len(rows)
    assert np.array_equal(saved["minutes"], run.MINUTES)
    for row in rows:
        prefix = f"{row['network']}__{row['scenario']}"
        ti = run.MINUTES.index(int(row["minute"]))
        satellite = saved[prefix + "__satellite"][ti]
        ground = saved[prefix + "__ground"][ti]
        assert len(satellite) == int(row["satellites"])
        assert len(ground) == 5001
        assert satellite.min() >= 4 and ground.min() >= 0
        assert (satellite - 4).sum() == ground.sum() == int(row["access_edges"])
        assert np.count_nonzero(ground) == int(row["covered_ground"])
        assert np.isclose(satellite.std() / satellite.mean(), float(row["sat_cv"]))
        if row["scenario"].endswith("single"):
            assert ground.max() <= 1
            assert int(row["satellite_components"]) == (3 if row["network"] == "Combined" else 1)
        else:
            assert int(row["satellite_components"]) == 1
    print("All 1536 graph records and input hashes verified", flush=True)

    raw_ground = list(csv.DictReader((run.ROOT / "inputs" / "ground_terminals.csv").open(encoding="utf-8")))
    mapping = list(csv.DictReader((run.ROOT / "inputs" / "terminal_label_mapping.csv").open(encoding="utf-8")))
    assert len(mapping) == len({r["terminal_id"] for r in mapping}) == 5001
    assert len({r["archived_label"] for r in mapping}) == 4831
    ground = np.array([[float(r["latitude_deg"]), float(r["longitude_deg"])] for r in raw_ground])
    positions = inputs["latitude_longitude_radius_km"][0]
    satellite_vectors = unit_vectors(positions[:, :2])
    ground_vectors = unit_vectors(ground)
    radius = positions[:, 2, None]
    links = {"flat": [], "sphere": []}
    for start in range(0, len(ground), 128):
        cosine = np.clip(satellite_vectors @ ground_vectors[start:start + 128].T, -1, 1)
        angle = np.arccos(cosine)
        elevation = np.arctan2(radius * cosine - run.EARTH_KM, radius * np.sqrt(np.maximum(0, 1 - cosine**2)))
        conditions = {"flat": run.EARTH_KM * angle <= (radius - run.EARTH_KM) / np.tan(run.ELEVATION),
                      "sphere": elevation >= run.ELEVATION}
        for model, condition in conditions.items():
            si, gi = np.nonzero(condition)
            links[model].append((si, gi + start, elevation[si, gi]))
    links = {model: tuple(np.concatenate([part[i] for part in parts]) for i in range(3))
             for model, parts in links.items()}
    for ni, network in enumerate(run.NETWORKS):
        lo, hi = (run.OFFSETS[ni], run.OFFSETS[ni + 1]) if ni < 3 else (0, sum(run.SIZES))
        for scenario in run.SCENARIOS:
            model, association = scenario.split("_")
            si, gi, elevation = links[model]
            keep = (si >= lo) & (si < hi)
            si, gi, elevation = si[keep] - lo, gi[keep], elevation[keep]
            if association == "single":
                candidates = {}
                for satellite, terminal, el in zip(si, gi, elevation):
                    candidates.setdefault(terminal, []).append((el, satellite))
                chosen = {terminal: min(s for el, s in options if el >= max(e for e, _ in options) - run.ELEVATION_TIE_TOL)
                          for terminal, options in candidates.items()}
                gi = np.array(list(chosen), dtype=int)
                si = np.array([chosen[g] for g in gi], dtype=int)
            prefix = f"{network}__{scenario}"
            assert np.array_equal(4 + np.bincount(si, minlength=hi - lo), saved[prefix + "__satellite"][0]), prefix
            assert np.array_equal(np.bincount(gi, minlength=5001), saved[prefix + "__ground"][0])
    print("Independent Cartesian geometry and association: all 16 t=0 graphs agree", flush=True)

    tails = read_csv("tail_tests.csv")
    assert len(tails) == 64
    assert len({(r["network"], r["scenario"], r["minute"]) for r in tails}) == 64
    for row in tails:
        assert int(row["n_tail"]) >= 50
        assert 0.01 <= float(row["p_gof"]) <= 1
        assert int(row["minute"]) in run.CHECKPOINTS
    target = next(r for r in tails if r["network"] == "Shell3" and r["scenario"] == "sphere_single" and r["minute"] == "0")
    repeated = run.original.evaluate_degrees(saved["Shell3__sphere_single__satellite"][0], seed=int(target["seed"]))
    for key, value in repeated.items():
        if isinstance(value, (bool, np.bool_)):
            assert str(value) == target[key]
        else:
            assert np.isclose(value, float(target[key]), equal_nan=True), key
    report = {"status": "passed", "graph_records": len(rows), "tail_tests": len(tails),
              "source_files_hash_verified": len(manifest["sources"]), "independent_geometry_graphs": 16,
              "deterministic_tail_repeat": "Shell3 / sphere_single / 0 min",
              "limitations": "Cartesian cross-check covers t=0; source hashes and graph-array checks cover all saved snapshots."}
    (run.ROOT / "results" / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    verify()
