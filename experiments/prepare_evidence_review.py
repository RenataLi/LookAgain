"""Make a condition-blind review packet for ALL changed primary-pair answers."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(run, manifest, summary_path, output, key_path):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and summary["integrity"]["passed"]
    assert sha(run / "records.jsonl") == summary["bindings"]["records_sha256"]
    assert sha(manifest) == summary["bindings"]["manifest_sha256"]
    completion = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    assert completion["records_sha256"] == sha(run / "records.jsonl")
    if output.exists() or key_path.exists():
        raise ValueError("Use new review and key paths")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    records = [json.loads(line) for line in (run / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    lookup = {(r["example_id"],r["action"]):r for r in records}
    output.mkdir(parents=True)
    packets, keys = [], []
    for row in rows:
        pair = [lookup[row["example_id"],f"{f}_256"] for f in ("native","degraded")]
        changed_parsed = pair[0]["predicted_answer"] != pair[1]["predicted_answer"]
        invalid_raw_change = (not all(r["parse_valid"] for r in pair) and pair[0]["response"] != pair[1]["response"])
        if not (changed_parsed or invalid_raw_change):
            continue
        case = f"case{len(packets)+1:03d}"
        # Swap order independently of score, report content or model condition.
        seed = int(hashlib.sha256(("v5-semantic-blind-20260926:"+row["example_id"]).encode()).hexdigest()[:16],16)
        random.Random(seed).shuffle(pair)
        with Image.open(manifest.parent / row["image_path"]) as raw:
            source = raw.convert("RGB")
        source.save(output / f"{case}-page.png")
        x0,y0,x1,y1 = row["roi_pixels"]
        context = [max(0,x0-300),max(0,y0-300),min(source.width,x1+300),min(source.height,y1+300)]
        source.crop(context).save(output / f"{case}-context.png")
        packets.append(dict(case_id=case, question=row["question"], reference=row["answer"],
            answer_A=pair[0]["predicted_answer"], answer_B=pair[1]["predicted_answer"],
            raw_response_A=pair[0]["response"], raw_response_B=pair[1]["response"],
            page=f"{case}-page.png",context=f"{case}-context.png",
            context_box_source_pixels=context, source_sha256=row["image_sha256"]))
        keys.append(dict(case_id=case,example_id=row["example_id"],source_id=row["source_id"],
                         A=pair[0]["action"],B=pair[1]["action"]))
    assert len(packets) >= summary["paired_native_degraded_response_agreement"]["256"]["parsed_answer_changed"]
    packet=dict(condition_labels_hidden=True, scope="All parsed-answer changes in the primary pair; includes equal-score pairs.",
        instruction="Inspect the source. Judge A and B as correct, incorrect or ambiguous. Distinguish content repair from formatting/span variation, both acceptable, both wrong, or ambiguity. Do not read the condition key before saving judgments. Do not change official scores.",
        records_sha256=sha(run / "records.jsonl"), cases=packets)
    (output / "packet.json").write_text(json.dumps(packet,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    key_path.parent.mkdir(parents=True,exist_ok=True)
    key_path.write_text(json.dumps(dict(packet_sha256=sha(output/"packet.json"),keys=keys),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(cases=len(packets),packet=str(output/"packet.json"))))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("run","manifest","summary","output","key"):
        parser.add_argument("--"+name,type=Path,required=True)
    a=parser.parse_args()
    prepare(a.run.resolve(),a.manifest.resolve(),a.summary.resolve(),a.output.resolve(),a.key.resolve())
