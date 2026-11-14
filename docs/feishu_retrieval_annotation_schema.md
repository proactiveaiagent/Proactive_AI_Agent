# Feishu retrieval annotation template

The generated JSONL (schema version 2) is an empty human-review template for the 35 videos in `test_data/feishu_ar_agent_memory_retrieval_20261013/manifest.json`. It contains no model captions, reference descriptions, or benchmark answers.

The generator freezes split using feishu-retrieval-split-v1: within each scenario category, the clip with the smallest SHA-256 rank is assigned to holdout (one clip per category); other clips are dev. The split uses only manifest scenario and clip IDs. It is a small pilot partition, not ground truth, and must remain fixed while retrieval is tuned.

Generate the template with `scripts/memory/prepare_feishu_retrieval_annotations.py`. Add `--verify-sha256` to verify every source video against the manifest before creating it. The script refuses to overwrite an existing output unless `--force` is supplied.

## Fields to fill

- `split` and `split_scheme`: generated and frozen by the deterministic split version above; do not change assignments after retrieval tuning begins.
- `history_clip_ids`: memory clips available at query time, in intended ingest order.
- `relevant_memory_clip_ids`: historical clips containing facts retrieval should return. Use an empty list only after review confirms no relevant memory; leave null while unreviewed.
- `hard_negative_clip_ids`: similar but irrelevant memories for false-association checks.
- `facts_to_retrieve`: concise facts that justify each positive, with fact type and source clip.
- `source_evidence`: source video and start/end seconds when the evidence can be located.
- `review_status`, `annotator`, `adjudicator`: use distinct people for the annotator and adjudicator; do not score until the selected split is fully adjudicated.

## Validation before scoring

Run `scripts/memory/validate_feishu_retrieval_annotations.py --split holdout` before calculating retrieval metrics. The validator checks that the annotation rows still match the frozen manifest split, query/history IDs are consistent, hard negatives are distinct, and fact/evidence sources are available historical clips. It reports readiness only when every row in the selected split is adjudicated; it never computes accuracy. Use `--require-ready` in a scoring job to fail closed while labels are incomplete.

Use review states `unreviewed`, `annotated`, and `adjudicated`. A fact entry contains `fact_type` (person/object/location/activity/environment/event), `entity`, `fact`, and `source_clip_id`. Each source evidence entry contains `clip_id`, `start_s`, `end_s`, and `modality` (video/audio). A relevant fact must cite an adjudicated relevant history clip and that clip must have source evidence.

## Leakage boundary

Run the query path with video input. Do not pass this annotation file, reference facts, or answers to Qwen/VLM/ASR, the planner, or the retrieval query builder. Keep annotations separate from caches and memory stores. The blank template is not ground truth and cannot support an accuracy claim.
