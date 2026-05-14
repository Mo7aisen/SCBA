from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scba.metrics.statistical_tests import compare_all_methods, friedman_test


@dataclass
class AuditIssue:
    severity: str  # "critical" | "major" | "minor"
    message: str
    path: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_close(a: float, b: float, *, name: str, path: Path, rtol: float = 1e-10, atol: float = 1e-12) -> None:
    if not np.isfinite(a) or not np.isfinite(b):
        raise AssertionError(f"{path}: {name} is not finite (a={a}, b={b})")
    if not np.isclose(a, b, rtol=rtol, atol=atol):
        raise AssertionError(f"{path}: {name} mismatch (a={a}, b={b})")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _audit_publication_dir(output_dir: Path) -> None:
    json_candidates = sorted(output_dir.glob("scba_publication*results.json"))
    if not json_candidates:
        return
    results_path = json_candidates[0]

    with open(results_path, "r") as f:
        payload = json.load(f)

    config = payload.get("config", {})
    seed = config.get("seed", 42)
    xai_method_keys: list[str] = list(config.get("xai_methods", []))
    xai_method_labels: dict[str, str] = dict(config.get("xai_method_labels", {}))
    cf_configs = config.get("cf_configs", [])
    detailed: dict[str, Any] = dict(payload.get("detailed_results", {}))
    summary: dict[str, Any] = dict(payload.get("summary", {}))
    stats: dict[str, Any] = dict(payload.get("statistics", {}))
    n_patients_any = None
    try:
        if isinstance(payload.get("summary"), dict) and payload["summary"]:
            n_patients_any = int(next(iter(payload["summary"].values())).get("n_patients", -1))
    except Exception:
        n_patients_any = None

    if not xai_method_keys or not xai_method_labels:
        raise AssertionError(f"{results_path}: missing xai_methods/xai_method_labels in config")
    if set(xai_method_keys) != set(xai_method_labels.keys()):
        raise AssertionError(f"{results_path}: xai_methods and xai_method_labels disagree")

    expected_cf_keys = [f"{c['operation']}_r{c['radius_px']}" for c in cf_configs]
    if not expected_cf_keys:
        raise AssertionError(f"{results_path}: missing cf_configs")

    per_method_patient = {m: {} for m in xai_method_keys}
    complete = {m: set() for m in xai_method_keys}

    for patient_id, patient_results in detailed.items():
        if not isinstance(patient_results, dict):
            continue
        for method_key in xai_method_keys:
            method_results = patient_results.get(method_key)
            if not isinstance(method_results, dict):
                continue
            if not all(k in method_results for k in expected_cf_keys):
                continue
            delta_vals: list[float] = []
            shift_vals: list[float] = []
            dc_vals: list[float] = []
            perceptual_vals: list[float] = []
            for cf_key in expected_cf_keys:
                cf = method_results[cf_key]
                metrics = cf.get("metrics", {})
                delta_vals.append(float(metrics["delta_am_roi"]))
                shift_vals.append(float(metrics["shift_distance"]))
                dc_vals.append(float(metrics["directional_consistency"]))
                pd = cf.get("perceptual_distance")
                if pd is not None:
                    perceptual_vals.append(float(pd))
            per_method_patient[method_key][patient_id] = {
                "delta_am_roi": float(np.mean(delta_vals)),
                "shift_distance": float(np.mean(shift_vals)),
                "directional_consistency": float(np.mean(dc_vals)),
                "perceptual_distance": float(np.mean(perceptual_vals)) if perceptual_vals else float("nan"),
            }
            complete[method_key].add(patient_id)

    common_patients = sorted(set.intersection(*(complete[m] for m in xai_method_keys)))
    if not common_patients:
        raise AssertionError(f"{results_path}: no complete-case patients across methods")
    if len(common_patients) < 2:
        # Scientific justification: smoke-test outputs may intentionally use
        # `--limit-samples 1`, which cannot support variance estimates or
        # hypothesis tests. In that case, we only validate that JSON structure
        # and patient-level aggregation are internally consistent.
        return

    for method_key in xai_method_keys:
        if method_key not in summary:
            raise AssertionError(f"{results_path}: summary missing method {method_key}")
        s = summary[method_key]
        if int(s.get("n_patients", -1)) != len(common_patients):
            raise AssertionError(
                f"{results_path}: n_patients mismatch for {method_key} "
                f"(json={s.get('n_patients')}, recomputed={len(common_patients)})"
            )

        values = np.array([per_method_patient[method_key][pid]["delta_am_roi"] for pid in common_patients], dtype=float)
        _assert_close(float(np.mean(values)), float(s["mean_delta_am_roi"]), name=f"{method_key}.mean_delta_am_roi", path=results_path)
        _assert_close(float(np.median(values)), float(s["median_delta_am_roi"]), name=f"{method_key}.median_delta_am_roi", path=results_path)

        dc = np.array([per_method_patient[method_key][pid]["directional_consistency"] for pid in common_patients], dtype=float)
        if np.any((dc < -1e-8) | (dc > 1 + 1e-8)):
            raise AssertionError(f"{results_path}: DC out of [0,1] bounds for method {method_key}")
        _assert_close(float(np.mean(dc)), float(s["mean_dc"]), name=f"{method_key}.mean_dc", path=results_path)

    # Recompute stored statistical objects from patient-level arrays.
    def _metric_arrays(metric_key: str) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for method_key in xai_method_keys:
            label = xai_method_labels[method_key]
            out[label] = np.array([per_method_patient[method_key][pid][metric_key] for pid in common_patients], dtype=float)
        return out

    recomputed = {
        "delta_am_roi": compare_all_methods(
            _metric_arrays("delta_am_roi"), metric_name="ΔAM-ROI", seed=seed
        ),
        "shift_distance": compare_all_methods(
            _metric_arrays("shift_distance"), metric_name="CoA Shift (pixels)", seed=seed
        ),
        "directional_consistency": compare_all_methods(
            _metric_arrays("directional_consistency"), metric_name="Directional Consistency", seed=seed
        ),
    }
    recomputed_friedman = {
        "delta_am_roi": friedman_test(_metric_arrays("delta_am_roi")),
        "shift_distance": friedman_test(_metric_arrays("shift_distance")),
        "directional_consistency": friedman_test(_metric_arrays("directional_consistency")),
    }

    for metric_name, recomputed_obj in recomputed.items():
        stored_obj = stats.get(metric_name)
        if not isinstance(stored_obj, dict):
            raise AssertionError(f"{results_path}: missing statistics.{metric_name}")

        for label, s in recomputed_obj["summary"].items():
            st = stored_obj["summary"][label]
            _assert_close(float(s["mean"]), float(st["mean"]), name=f"{metric_name}.{label}.mean", path=results_path)
            _assert_close(float(s["median"]), float(st["median"]), name=f"{metric_name}.{label}.median", path=results_path)
            _assert_close(float(s["ci_lower"]), float(st["ci_lower"]), name=f"{metric_name}.{label}.ci_lower", path=results_path)
            _assert_close(float(s["ci_upper"]), float(st["ci_upper"]), name=f"{metric_name}.{label}.ci_upper", path=results_path)

        # Pairwise comparisons: compare keyed values, not list order.
        def _pair_key(p: dict[str, Any]) -> tuple[str, str]:
            return (p["method_1"], p["method_2"])

        rec_pairs = {_pair_key(p): p for p in recomputed_obj["pairwise_comparisons"]}
        st_pairs = {_pair_key(p): p for p in stored_obj["pairwise_comparisons"]}
        if rec_pairs.keys() != st_pairs.keys():
            raise AssertionError(f"{results_path}: pairwise comparison keys differ for {metric_name}")
        for key, rp in rec_pairs.items():
            sp = st_pairs[key]
            _assert_close(float(rp["t_p_value"]), float(sp["t_p_value"]), name=f"{metric_name}.{key}.t_p_value", path=results_path)
            _assert_close(float(rp["wilcoxon_p_value"]), float(sp["wilcoxon_p_value"]), name=f"{metric_name}.{key}.wilcoxon_p_value", path=results_path)
            _assert_close(float(rp["cohens_d"]), float(sp["cohens_d"]), name=f"{metric_name}.{key}.cohens_d", path=results_path)

        fr = recomputed_friedman[metric_name]
        fs = stats.get("friedman_tests", {}).get(metric_name)
        if not isinstance(fs, dict):
            raise AssertionError(f"{results_path}: missing friedman_tests.{metric_name}")
        _assert_close(float(fr["p_value"]), float(fs["p_value"]), name=f"friedman.{metric_name}.p_value", path=results_path)
        _assert_close(float(fr["statistic"]), float(fs["statistic"]), name=f"friedman.{metric_name}.statistic", path=results_path)

    # Table CSV cross-check (if present)
    tables_dir = output_dir / "tables"
    summary_csv = tables_dir / "delta_am_roi_summary.csv"
    if summary_csv.exists():
        rows = _read_csv_rows(summary_csv)
        by_method = {r["method"]: r for r in rows}
        for label, s in stats["delta_am_roi"]["summary"].items():
            r = by_method.get(label)
            if not r:
                raise AssertionError(f"{summary_csv}: missing method row: {label}")
            _assert_close(float(r["mean"]), float(s["mean"]), name=f"csv.delta_am.{label}.mean", path=summary_csv)
            _assert_close(float(r["ci_lower"]), float(s["ci_lower"]), name=f"csv.delta_am.{label}.ci_lower", path=summary_csv)
            _assert_close(float(r["ci_upper"]), float(s["ci_upper"]), name=f"csv.delta_am.{label}.ci_upper", path=summary_csv)

    # Figure existence + decodability (basic visual sanity)
    fig_candidates = sorted(output_dir.glob("scba_publication*figure.png"))
    if fig_candidates:
        with Image.open(fig_candidates[0]) as im:
            im.verify()


def _audit_radius_dir(output_dir: Path) -> None:
    json_files = sorted(output_dir.glob("*_radius_sensitivity.json"))
    if not json_files:
        return
    for json_path in json_files:
        with open(json_path, "r") as f:
            payload = json.load(f)
        detailed = payload.get("detailed_results", {})
        aggregates = payload.get("method_aggregates", {})
        if not isinstance(detailed, dict) or not isinstance(aggregates, dict):
            raise AssertionError(f"{json_path}: malformed radius sensitivity json")

        # Recompute aggregates directly from detailed_results.
        for method, method_data in aggregates.items():
            for cf_key, stats in method_data.items():
                delta_vals = []
                shift_vals = []
                dc_vals = []
                ssim_vals = []
                perc_vals = []
                intensity_vals = []
                for _pid, pdata in detailed.items():
                    if method in pdata and cf_key in pdata[method]:
                        cf = pdata[method][cf_key]
                        metrics = cf["metrics"]
                        delta_vals.append(metrics["am_roi_perturbed"] - metrics["am_roi_original"])
                        shift_vals.append(metrics["shift_distance"])
                        dc_vals.append(metrics["directional_consistency"])
                        ssim_vals.append(cf.get("ssim", np.nan))
                        if cf.get("perceptual_distance") is not None:
                            perc_vals.append(cf["perceptual_distance"])
                        intensity_vals.append(cf.get("intensity_delta", 0.0))
                if not delta_vals:
                    raise AssertionError(f"{json_path}: no records for {method}/{cf_key}")
                _assert_close(float(np.mean(delta_vals)), float(stats["mean_delta_am_roi"]), name=f"{method}.{cf_key}.mean_delta_am_roi", path=json_path)
                _assert_close(float(np.mean(dc_vals)), float(stats["mean_dc"]), name=f"{method}.{cf_key}.mean_dc", path=json_path)
                _assert_close(float(np.nanmean(ssim_vals)), float(stats["mean_ssim"]), name=f"{method}.{cf_key}.mean_ssim", path=json_path)


def _audit_ablation_dir(output_dir: Path) -> None:
    json_files = sorted(output_dir.glob("scba_ablation_*.json"))
    if not json_files:
        return
    for json_path in json_files:
        with open(json_path, "r") as f:
            payload = json.load(f)
        ablations = payload.get("ablation_results", {})
        if not isinstance(ablations, dict):
            raise AssertionError(f"{json_path}: missing ablation_results")
        for cfg_name, cfg_payload in ablations.items():
            records = cfg_payload.get("records", [])
            summary = cfg_payload.get("summary", {})
            if not records:
                raise AssertionError(f"{json_path}: missing records for {cfg_name}")
            delta = [r["delta_am_roi"] for r in records]
            dc = [r["directional_consistency"] for r in records]
            _assert_close(float(np.nanmean(delta)), float(summary["mean_delta_am_roi"]), name=f"{cfg_name}.mean_delta_am_roi", path=json_path)
            _assert_close(float(np.nanmean(dc)), float(summary["mean_dc"]), name=f"{cfg_name}.mean_dc", path=json_path)

        # Cross-check CSV summary if present.
        csv_path = output_dir / f"scba_ablation_{payload.get('dataset', '').lower()}.csv"
        if csv_path.exists():
            rows = _read_csv_rows(csv_path)
            by_name = {r["config"]: r for r in rows}
            for cfg_name, cfg_payload in ablations.items():
                row = by_name.get(cfg_name)
                if not row:
                    raise AssertionError(f"{csv_path}: missing config row {cfg_name}")


def _audit_sanity_dir(output_dir: Path) -> None:
    json_files = sorted(output_dir.glob("scba_randomization_*.json"))
    if not json_files:
        return
    for json_path in json_files:
        with open(json_path, "r") as f:
            payload = json.load(f)
        stages = payload.get("stages", {})
        order = payload.get("randomization_order", [])
        if not stages or not order:
            raise AssertionError(f"{json_path}: missing stages/randomization_order")
        csv_path = output_dir / f"scba_randomization_{payload.get('dataset', '').lower()}.csv"
        if csv_path.exists():
            rows = _read_csv_rows(csv_path)
            by_stage = {r["stage"]: r for r in rows}
            for stage in order:
                s = stages[stage]
                r = by_stage.get(stage)
                if not r:
                    raise AssertionError(f"{csv_path}: missing stage row {stage}")
                _assert_close(
                    float(r["mean_pearson_corr"]),
                    float(s["mean_pearson_corr"]),
                    name=f"{stage}.mean_pearson_corr",
                    path=csv_path,
                    rtol=1e-6,
                    atol=1e-6,
                )
                _assert_close(
                    float(r["mean_ssim"]),
                    float(s["mean_ssim"]),
                    name=f"{stage}.mean_ssim",
                    path=csv_path,
                    rtol=1e-6,
                    atol=1e-6,
                )


def _write_manifest(output_dir: Path, manifest_path: Path) -> None:
    entries = []
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
        entries.append({"path": str(path.relative_to(output_dir)), "sha256": _sha256(path), "bytes": path.stat().st_size})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({"root": str(output_dir), "files": entries}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SCBA output directories for internal consistency.")
    parser.add_argument("--input", type=str, required=True, help="Output directory to audit.")
    parser.add_argument("--write-manifest", action="store_true", help="Write sha256 manifest for all files under --input.")
    args = parser.parse_args()

    output_dir = Path(args.input)
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)

    # Scientific justification: automated audits catch silent mismatches between
    # manuscript tables/figures and the underlying result JSONs.
    _audit_publication_dir(output_dir)
    _audit_radius_dir(output_dir)
    _audit_ablation_dir(output_dir)
    _audit_sanity_dir(output_dir)

    if args.write_manifest:
        _write_manifest(output_dir, output_dir / "audit_manifest_sha256.json")

    print(f"✓ Audit OK: {output_dir}")


if __name__ == "__main__":
    main()
