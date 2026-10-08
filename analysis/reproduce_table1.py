"""Reproduce the robust power-law refits reported for Table 1.

The script uses only the processed degree distributions distributed with this
repository. Run it directly in an IDE or with ``python analysis/reproduce_table1.py``.
"""

import csv
from pathlib import Path

import numpy as np
from scipy import optimize, special, stats


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "table1_degree_distributions"
REFERENCE_CSV = ROOT / "results" / "core" / "table1_robust_refit.csv"
OUTPUT_CSV = ROOT / "results" / "table1_robust_refit_reproduced.csv"

TABLE1_CASES = (
    (169, 4000),
    (400, 8000),
    (900, 10000),
    (16900, 80000),
    (14400, 100000),
    (16900, 200000),
)
SEED = 12345678910
MIN_TAIL = 50
CI_REPS = 199
GOF_REPS = 99


def _alpha_mle(tail, xmin):
    logs = np.log(tail).sum()
    n = len(tail)
    objective = lambda alpha: n * np.log(special.zeta(alpha, xmin)) + alpha * logs
    result = optimize.minimize_scalar(
        objective,
        bounds=(1.01, 30.0),
        method="bounded",
    )
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
    return float(
        max(
            np.max(np.abs(empirical_after - model_after)),
            np.max(np.abs(empirical_before - model_before)),
        )
    )


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
            best = {
                "xmin": int(xmin),
                "alpha": alpha,
                "ks": ks,
                "n_tail": len(tail),
            }
    return best


def fit_log_binned_powerlaw(degrees, xmin):
    """Fit a descriptive power law on factor-two logarithmic bins."""
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
        raise RuntimeError("Log-binned power-law fit failed")

    log_prefactor, alpha = result.x
    fitted = line(log_x, log_prefactor, alpha)
    total = np.sum((log_y - log_y.mean()) ** 2)
    r_squared = (
        1 - np.sum((log_y - fitted) ** 2) / total
        if len(log_x) > 2 and total
        else float("nan")
    )
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
    log_normalizer = special.log_ndtr(
        -(np.log(xmin - 0.5) - mu) / sigma
    )
    return log_mass - log_normalizer


def fit_lognormal(tail, xmin):
    logs = np.log(tail)
    center = float(np.mean(logs))
    spread = max(float(np.std(logs)), 0.2)
    bounds = (
        (np.log(xmin) - 30, np.log(max(tail)) + 10),
        (0.05, 20.0),
    )
    starts = (
        (center, spread),
        (np.log(xmin), 1.0),
        (center - 1, 2.0),
        (np.log(xmin) - 5, 4.0),
        (np.log(xmin) - 15, 8.0),
    )
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
    at_boundary = any(
        np.isclose(value, limit, rtol=0, atol=1e-5)
        for value, limits in zip(result.x, bounds)
        for limit in limits
    )
    return float(result.x[0]), float(result.x[1]), at_boundary


def _sample_powerlaw(rng, n, xmin, alpha):
    """Invert the exact Hurwitz-zeta CDF without finite-support truncation."""
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
    tail = degrees[degrees >= xmin]
    body = degrees[degrees < xmin]
    fit.update(fit_log_binned_powerlaw(tail, xmin))

    rng = np.random.default_rng(seed)
    ci = [
        _alpha_mle(rng.choice(tail, len(tail), replace=True), xmin)
        for _ in range(CI_REPS)
    ]
    fit["ci_low"], fit["ci_high"] = np.quantile(ci, (0.025, 0.975))

    simulated_ks = []
    for _ in range(GOF_REPS):
        synthetic = _sample_powerlaw(rng, len(tail), xmin, alpha)
        if len(body):
            synthetic = np.concatenate(
                (synthetic, rng.choice(body, len(body), replace=True))
            )
        simulated_ks.append(fit_powerlaw(synthetic)["ks"])
    fit["p_gof"] = (
        1 + sum(value >= fit["ks"] for value in simulated_ks)
    ) / (GOF_REPS + 1)

    mu, sigma, at_boundary = fit_lognormal(tail, xmin)
    log_ratio = _powerlaw_logpmf(tail, xmin, alpha) - _lognormal_logpmf(
        tail,
        xmin,
        mu,
        sigma,
    )
    sd = float(np.std(log_ratio, ddof=1))
    fit["lognormal_mu"] = mu
    fit["lognormal_sigma"] = sigma
    fit["lognormal_at_boundary"] = at_boundary
    fit["llr_pl_vs_lognormal"] = float(log_ratio.sum())
    fit["p_vuong"] = (
        float(2 * stats.norm.sf(abs(np.sqrt(len(tail)) * np.mean(log_ratio) / sd)))
        if sd
        else float("nan")
    )
    fit["n_observations"] = len(degrees)
    fit["tail_fraction"] = len(tail) / len(degrees)
    return fit


def read_saved_degree_distribution(path):
    """Recover integer observations from a saved two-row degree PMF."""
    with path.open(encoding="utf-8") as file:
        degree_row, probability_row = list(csv.reader(file))[:2]
    degrees = np.array([int(value) for value in degree_row if value], dtype=int)
    probabilities = np.array(
        [float(value) for value in probability_row if value]
    )
    if len(degrees) != len(probabilities) or not np.isclose(
        probabilities.sum(),
        1.0,
    ):
        raise ValueError(f"Invalid saved degree distribution: {path}")

    positive = probabilities[probabilities > 0]
    n_observations = round(1 / positive.min())
    counts = np.rint(probabilities * n_observations).astype(int)
    if counts.sum() != n_observations or not np.allclose(
        counts / n_observations,
        probabilities,
    ):
        raise ValueError(f"Could not recover integer degree counts: {path}")
    return np.repeat(degrees, counts)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def reproduce_table1():
    rows = []
    for n_satellite, n_terminal in TABLE1_CASES:
        path = DATA_DIR / f"GS{n_satellite}_CITY{n_terminal}.txt"
        row = {
            "n_satellite": n_satellite,
            "n_terminal": n_terminal,
            "source": path.name,
        }
        row.update(
            evaluate_degrees(
                read_saved_degree_distribution(path),
                seed=SEED + n_satellite + n_terminal,
            )
        )
        rows.append(row)
        print(
            f"{path.name}: xmin={row['xmin']}, alpha={row['alpha']:.3f}, "
            f"p_gof={row['p_gof']:.3f}"
        )
    return rows


def verify_against_reference(rows):
    with REFERENCE_CSV.open(newline="", encoding="utf-8") as file:
        reference = list(csv.DictReader(file))
    if len(rows) != len(reference):
        raise AssertionError("The reproduced and reference row counts differ")

    numeric_fields = ("xmin", "alpha", "ks", "n_tail", "p_gof")
    for actual, expected in zip(rows, reference):
        if actual["source"] != expected["source"]:
            raise AssertionError("The reproduced and reference cases differ")
        for field in numeric_fields:
            if not np.isclose(float(actual[field]), float(expected[field]), rtol=1e-8):
                raise AssertionError(
                    f"Mismatch for {actual['source']} field {field}: "
                    f"{actual[field]} != {expected[field]}"
                )


def main():
    rows = reproduce_table1()
    verify_against_reference(rows)
    write_csv(OUTPUT_CSV, rows)
    print(f"Verified against {REFERENCE_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
