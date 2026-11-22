"""Export MLflow experiment runs and registered model versions to Markdown/JSON for submission."""

import argparse
import json
from pathlib import Path

from baselines.difference_detector import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI
from preprocessing.dataset import PROJECT_ROOT
from training.train import DEFAULT_REGISTERED_MODEL


def export_report(tracking_uri=DEFAULT_TRACKING_URI, experiment_name=DEFAULT_EXPERIMENT,
                  model_name=DEFAULT_REGISTERED_MODEL, output_dir=PROJECT_ROOT / "outputs/mlflow_report"):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment = client.get_experiment_by_name(experiment_name)
    runs_summary = []
    if experiment is not None:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
        )
        for run in runs:
            runs_summary.append({
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "status": run.info.status,
                "model_type": run.data.tags.get("model_type", ""),
                "params": dict(sorted(run.data.params.items())),
                "metrics": dict(sorted(run.data.metrics.items())),
                "tags": {
                    k: v for k, v in sorted(run.data.tags.items())
                    if not k.startswith("mlflow.")
                },
            })

    registered_versions = []
    try:
        for mv in client.search_model_versions(f"name='{model_name}'"):
            registered_versions.append({
                "name": mv.name,
                "version": mv.version,
                "run_id": mv.run_id,
                "status": mv.status,
                "source": mv.source,
                "current_stage": getattr(mv, "current_stage", ""),
            })
    except Exception:
        pass

    payload = {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "registered_model_name": model_name,
        "runs": runs_summary,
        "registered_model_versions": registered_versions,
    }
    json_path = output_dir / "mlflow_runs_and_registry.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# MLflow Experiment Tracking & Model Registry Export",
        "",
        f"- **Tracking URI**: `{tracking_uri}`",
        f"- **Experiment Name**: `{experiment_name}`",
        f"- **Total Runs Logged**: `{len(runs_summary)}`",
        "",
        "## 1. Logged Runs (Baseline & Primary Model)",
        "",
        "| Run Name | Run ID | Type | Status | Key Metrics | Dataset SHA256 / DVC |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for item in runs_summary:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in item["metrics"].items())
        dvc_str = f"dvc={item['params'].get('dvc_md5', 'n/a')[:8]} / split={item['params'].get('split_sha256', 'n/a')[:8]}"
        lines.append(
            f"| `{item['run_name']}` | `{item['run_id'][:8]}` | `{item['model_type']}` | "
            f"`{item['status']}` | {metrics_str} | `{dvc_str}` |"
        )

    lines.extend([
        "",
        "## 2. Registered Models in MLflow Model Registry",
        "",
        "| Model Name | Version | Run ID | Status | Source URI |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])
    for mv in registered_versions:
        lines.append(
            f"| `{mv['name']}` | `v{mv['version']}` | `{mv['run_id'][:8]}` | "
            f"`{mv['status']}` | `{mv['source']}` |"
        )

    md_path = output_dir / "mlflow_runs_and_registry.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Exported MLflow JSON report: {json_path}")
    print(f"Exported MLflow Markdown report: {md_path}")
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--model-name", default=DEFAULT_REGISTERED_MODEL)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/mlflow_report")
    args = parser.parse_args()
    export_report(args.tracking_uri, args.experiment_name, args.model_name, args.output_dir)


if __name__ == "__main__":
    main()
