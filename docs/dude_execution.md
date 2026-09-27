# DUDE four-condition execution

This guide records the original local experiment workflow. Its commands require the historical source manifests, raw records and identity artifacts, which are not bundled in this numerical release. See the [study index](../reports/README.md) and [current CPU replay](reproduce.md) for the published materials.

The source/design lock and preparation configuration remain immutable. A new execution lock binds the dedicated DUDE runner, analyzer and blinded-review tools, their unchanged request/backend/scorer dependencies, this guide, the final manifests, actual local model/processor files and runtime. Separate engineering and main locks use separate run IDs and output directories. Creating a lock generates no answers.

Engineering verification preceded the planned 660-source main experiment. Scientific inputs, outcomes and advancement criteria stay as specified in `dude_replication_protocol.md`. Ten approved engineering sources produce 40 experimental calls, excluded from main. Six synthetic two-token warmup calls precede each fresh process and do not enter experimental scores. A main lock requires successful engineering completion with identical executable code; engineering bugs require fixes, fresh verification and a new engineering run before main.

## Integrity and measurement

The original v5 request builder and Qwen backend remain unchanged. A read-only wrapper around the processor observes the actual CPU BatchFeature immediately before CUDA transfer. It compares input ID, attention mask, pixel and grid tensor hashes, formatted prompt hash, token count and grids with the bound source-preparation CPU checks. It returns the same processor object outputs without changing their values. CPU input hashes are input evidence, not model activation measurements.

Before generation, source image and literal request identities are checked. Every saved record includes source/reference/audit identity, exact request geometry and hashes, raw continuation and full response, parser/metric results, actual grids/tokens, truncation, synchronized latency and peak allocated/reserved GPU memory. Scores are logged for later reconstruction but no interim quality analysis is performed. Invalid or truncated answers are retained. Execution errors abort without automatic retry or source replacement.

Measured standalone time includes image decode, view construction, RGB hashing, processor work, the added input observation/hash validation, device transfer, generation, decode and diagnostic log probability. Observer overhead is recorded separately and is not silently subtracted. PDF preparation, model loading, synthetic warmup and output record I/O are outside this timing. The machine is an active desktop; results are not isolated throughput or energy measurements. Decision-state cost adds direct256 latency to regional latency and uses the maximum sequential memory peak; localization cost remains unmeasured.

Runtime fixes seed20260927, four PyTorch CPU threads, CUBLAS workspace `:4096:8`, TF32 off and cuDNN benchmarking off; greedy BF16/SDPA parameters remain unchanged. Sources run in frozen manifest order. Within each source, the four actions are shuffled with Python `random.Random(int(SHA256(str(seed)+":"+example_id)[:16],16))`. Fresh requests do not reuse previous answers or KV state. Six synthetic warmups cover landscape, square and portrait direct/native shapes, not every production shape.

Each action is flushed and each source fsynced. Resume validates the complete immutable identity, every existing record and reconstructed pixels/grades. It never mixes code/runtime/model identities. An error log blocks reuse; source/scientific changes require preservation of the failed run and a separately documented new lock. A completion file binds all planned records and their hash. Completed-run analysis rejects partial, duplicate, extra or mismatched records, altered references and changed source/config/code/model identity. Recovered completion after interruption may lack total wall time; it must disclose that limit.

## Statistics and blinded source review

The sole primary contrast and all scientific thresholds are unchanged. Bootstrap implementation uses Python `random.Random(20260927)` to draw 10,000 samples of 660 complete source clusters with replacement, and NumPy linear-interpolated 2.5/97.5 percentiles. The same sampled indices are shared across action/metric contrasts. Exact McNemar uses the two-sided binomial test with p=.5. It is not a mid-p or asymptotic test. The descriptive content screen cannot override a failed quantitative screen or alter automatic scores.

After all main calls and integrity validation, `prepare_dude_review.py` creates a blinded packet for every native/degraded parsed-answer change, plus differing raw answers when either parse is invalid. Generic case/image names hide source IDs, conditions and scores. A private condition key is kept outside the packet. Each qualitative judgment records page/ROI inspection, A/B correctness or uncertainty, and a reasoned category. All judgments and image/packet hashes are frozen before unblinding. The judgments are descriptive and do not constitute independent expert validation.

Case order is ascending SHA256 of `dude-semantic-case-order:20260927:example_id`, with example ID as a tie-breaker. The A/B assignment shuffles `[native_256,degraded_256]` with Python Random seeded by the big-endian integer SHA256 of `dude-semantic-blind:20260927:example_id`. Both are independent of responses and scores. The packet commits to a separate key containing the source mapping and original run hashes. Review ledgers bind every pair and both inspected images; their complete decisions are frozen before the key is read.

## Local commands

Use the prepared local manifest directory, pinned local model and a verification JSON produced after tests and independent code review. The verification must bind every execution file's SHA256; it is checked before a lock is written.

```text
python experiments/dude_replication.py lock --role engineering --manifest DATA/engineering_manifest.jsonl --model-dir MODEL --verification VERIFICATION.json --run-id dude-engineering-01 --output ENGINEERING-LOCK.json
python experiments/dude_replication.py run --role engineering --manifest DATA/engineering_manifest.jsonl --model-dir MODEL --lock ENGINEERING-LOCK.json --output ENGINEERING-RUN
python experiments/analyze_dude_replication.py --engineering --manifest DATA/engineering_manifest.jsonl --lock ENGINEERING-LOCK.json --run ENGINEERING-RUN --output ENGINEERING-REPORT
python experiments/dude_replication.py lock --manifest DATA/main_manifest.jsonl --model-dir MODEL --verification VERIFICATION.json --run-id dude-main-01 --engineering-run ENGINEERING-RUN --engineering-lock ENGINEERING-LOCK.json --output MAIN-LOCK.json
python experiments/dude_replication.py run --manifest DATA/main_manifest.jsonl --model-dir MODEL --lock MAIN-LOCK.json --output MAIN-RUN
```

Run the main analyzer only on the completed validated run. Preserve all raw records, run/lock/completion identities, engineering evidence, source-only audits, frozen blinded judgments and code. Source media, raw OCR, raw answers, source-linked manifests and weights remain outside this numerical release. Earlier five experiments and the DUDE source-preparation release are immutable historical artifacts.
