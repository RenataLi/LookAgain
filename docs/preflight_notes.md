# Engineering preflight and frozen main-pilot contract

The development run was prepared on 25 September 2026. These notes distinguish debugging from the main paired experiment. None of the preflight runs are a held-out evaluation.

1. The first four-image check exposed copying of an angle-bracket answer placeholder and omission of an explicit final marker. Those outputs are excluded from the main experiment.
2. A system instruction without placeholders removed the placeholder problem. The parser was made explicit: accept a bare one-line answer in its entirety, or a unique final `ANSWER:` line; reject malformed multi-line responses. Normalized exact matching remains conservative and reference-independent.
3. An initial expanded preflight was stopped after observing verbose answers to short-answer questions. It remains an incomplete debugging run, not a result. Its outputs are not pooled with the final run.
4. The frozen short-answer instruction is `Answer the question using a single word or short phrase. Do not explain.` Short branches prefill the assistant prefix `ANSWER:` using the model's chat template. Only answer content is then generated. The reasoning branch generates brief reasoning and its own final marker. All branches share the same parser and normalization.
5. A 16-image format check produced 128 completed action records with zero invalid formats and zero truncated generations. This establishes an engineering check, not a guarantee of format compliance on all data or a performance claim.
6. The final run, `pilot400-final`, starts afresh over all 400 development images using the frozen configuration and source digest. It records its complete planned ID list, immutable model/dataset revisions, runtime versions and exact input sizes.

Prompt selection was based on response-contract compliance. No yes/no-prefix rescue, target-word searching, semantic judge, or per-example answer repair was added. A verbose answer can still fail exact matching; inspect the raw outputs and format diagnostics before interpreting differences.

The model snapshot is `Qwen/Qwen3-VL-4B-Instruct` at `ebb281ec70b05090aa6165b016eac8ec08e71b17`. The GQA mirror is pinned at `a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8`.

Source changes after engineering checks are recorded in the main run's source digest. This is a documented development protocol, not external preregistration or evidence of test-set independence.
