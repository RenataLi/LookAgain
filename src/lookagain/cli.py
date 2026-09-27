from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="LookAgain research pilot")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch-model", help="Download a pinned public model snapshot")
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    fetch.add_argument("--revision", default="main")
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--limit", type=int)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--run", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    visualize = sub.add_parser("visualize", help="Render measured figures and a local image gallery")
    visualize.add_argument("--run", type=Path, required=True)
    visualize.add_argument("--manifest", type=Path, required=True)
    visualize.add_argument("--summary", type=Path, required=True)
    visualize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fetch-model":
        from huggingface_hub import HfApi, snapshot_download
        info = HfApi().model_info(args.model_id, revision=args.revision)
        target = args.output.resolve()
        marker = target / "lookagain-model.json"
        if marker.exists():
            old = json.loads(marker.read_text(encoding="utf-8"))
            if old != {"model_id": args.model_id, "revision": info.sha}:
                raise ValueError("model directory belongs to another revision")
        snapshot_download(args.model_id, revision=info.sha, local_dir=str(target), allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"], max_workers=3)
        marker.write_text(json.dumps({"model_id": args.model_id, "revision": info.sha}, indent=2), encoding="utf-8")
        print(marker, flush=True)
    elif args.command == "run":
        from .runner import run_pilot
        run_pilot(args.manifest.resolve(), args.model_dir.resolve(), args.config.resolve(), args.output.resolve(), args.limit)
    elif args.command == "visualize":
        from .visuals import render_figures, render_gallery
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        records = [json.loads(line) for line in (args.run / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        for output in render_figures(summary, args.output):
            print(output)
        print(render_gallery(records, args.manifest, args.output / "gallery.html"))
    else:
        from .analysis import analyze_records, write_report
        metadata = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in (args.run / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        summary = analyze_records(records, bootstrap_samples=10000, expected_actions=metadata["config"]["actions"])
        planned = set(metadata["example_ids"])
        seen = {row["example_id"] for row in records}
        if seen - planned:
            raise ValueError("records include examples outside the run manifest")
        absent = sorted(planned - seen)
        summary["coverage"]["planned_examples"] = len(planned)
        summary["coverage"]["wholly_absent_examples"] = absent
        summary["coverage"]["planned_completion_fraction"] = summary["coverage"]["complete_examples"] / len(planned)
        if absent:
            summary["status"] = "incomplete_coverage"
            summary["warnings"].append(f"{len(absent)} planned examples have no recorded actions; this run is incomplete.")
        summary["format_diagnostics"] = {}
        for action in metadata["config"]["actions"]:
            selected = [row for row in records if row["action"] == action]
            summary["format_diagnostics"][action] = {"records": len(selected), "invalid_answers": sum(not row["parse_valid"] for row in selected), "truncated_generations": sum(row["generation_truncated"] for row in selected)}
        summary["run_fingerprint"] = metadata["fingerprint"]
        summary["planned_examples"] = len(metadata["example_ids"])
        summary["development_only"] = True
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(summary, args.output / "report.md")
        with (args.output / "report.md").open("a", encoding="utf-8") as report:
            report.write(f"\nPlanned examples: {len(planned)}. Entirely absent: {len(absent)}. This is a development convenience sample, not a held-out benchmark result.\n\n")
            report.write("| Action | Recorded | Invalid answer format | Truncated generation |\n| --- | ---: | ---: | ---: |\n")
            for action, counts in summary["format_diagnostics"].items():
                report.write(f"| {action} | {counts['records']} | {counts['invalid_answers']} | {counts['truncated_generations']} |\n")
            report.write("\nGQA source images may be upsampled to the fixed visual-token budget. Upsampling introduces no new source detail. These interventions measure changes from visual re-encoding and attention allocation.\n")
        print(args.output / "report.md")


if __name__ == "__main__":
    main()
