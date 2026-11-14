# Feishu video-only memory QA — 2026-10-18

## Scope and limits

This is a provisional visual QA note based on the video frames only. It is not an independently reviewed human gold label, is not used to score retrieval, and has not been written into the runtime cache or query annotations. No transcript or test-set text was supplied to the perception model during this review.

## Source identity check

The source video SHA-256 values match the videos referenced by the current Qwen cache records:

| Clip | Video SHA-256 | Cache prompt version | Cached at |
|---|---|---|---|
| FEISHU_3_1_3adf23f5bb | 3adf23f5bb0e2f1bc528e28c744389c3114ef4a3e2e45bf611a22d77a59bb1eb | memory-v7.4-scene-entity-timeline-sourcehash-2026-10-18 | 2026-10-18 00:44:12 |
| FEISHU_3_3_6f73e3eba3 | 6f73e3eba312ed5f2ba480e0c42a7cabbaeca052e0b8473ddca92d884e3eb6be | memory-v7.4-scene-entity-timeline-sourcehash-2026-10-18 | 2026-10-18 00:44:10 |

Both cache entries record six sampled frames.

## Observations to verify

- **3.1, forgetting-item scene:** sampled frames show a black, screen-like handheld object on the nightstand, then a hand placing an object into the top drawer. The cached Scene and Action call the moved object a “black wallet,” while also listing a smartphone as a separate object. This may be an object-identity error. A human reviewer should decide whether the visible evidence supports “phone/device,” “wallet,” or only “black rectangular object.”
- **3.3, wallet scene:** sampled frames show a wallet being taken from a drawer and opened, with cards/cash-like contents visible. The cached description as a wallet is consistent with the sampled visual evidence.
- These are different source clips. Do not infer that the same person or the same physical object appears in both.

## Follow-up

1. Add the 3.1 object-identity question to a human-reviewed development regression set after adjudication.
2. Preserve the source-backed state transition (nightstand surface → top drawer) only if a reviewer confirms it; keep identity uncertain when the frames do not resolve it.
3. Decide whether to refine the perception prompt after the active full cache pass finishes. If changed, use a new prompt version and re-run cache coverage checks; do not hand-edit this cached record.
4. Keep this provisional QA separate from retrieval accuracy metrics until independent labels and a holdout set exist.

## Preliminary structured-field coverage — 2026-10-18 01:55 CST

At this snapshot, 354 v7.4 cache records with source fingerprints contain 2,000 temporal facts; 1,999 have a frame index and all 2,000 carry an evidence string. The same records contain 681 person observations with appearance descriptions, 1,425 environment entries, and 4,011 object entries. Only 64 temporal facts explicitly include a `from_location` or `to_location` field. These are model-output coverage counts, not accuracy scores; person-age coverage and factual correctness still need separate review. This supports prioritizing evidence-backed object state/location changes and conservative uncertainty handling in the next quality audit.

## Partial v7.4 cache inventory — 2026-10-18 07:02 CST

The live prewarm had saved 1,292 v7.4 perception results (1,400/2,793 items visited; 108 rejected) when this read-only snapshot was taken. These coverage counts describe model output structure only; they are not factual-accuracy scores and do not represent the full profile inventory.

- Scene and activity were populated in 1,292/1,292; location in 1,290/1,292; time/occasion in 1,260/1,292; environment in 1,292/1,292. Person observations appeared in 840/1,292 clips (2,709 observations). Exact-age fields were not supported by visible text in this snapshot; apparent-age estimates remain separately marked as estimates.
- There were 7,480 temporal facts across 1,193 clips. All 7,480 had an evidence string; 7,479 had a sampled frame index and 6,954 had a time scope. The snapshot had 4,021 object facts, including 1,199 `located_at` facts.
- All 70 object facts explicitly labeled `moved` had a before/after location or state pair plus frame and evidence. The broader predicates `placed` and `picked_up` are not automatically treated as confirmed movements.
- A read-only audit found 21 interaction facts whose value explicitly names another observed neutral person ID; 12 did not repeat that ID in `affected_people`. This is a focused structural-quality gap to address in a subsequent prompt revision or controlled normalization review. It is not a claim that 12 interactions were factually wrong.
- `source_evidence` is intentionally empty in Qwen fields; clip/frame provenance is attached by the memory runner. Do not score this field as missing provenance.

After the active cache finishes, repeat this inventory on the full 4,556-clip scope and manually review a stratified sample of interaction target links before changing the prompt version or accepting a new affected-person rule.

### Downstream impact-person indexing check — 2026-10-18

A code-path review confirms that when Qwen supplies actor or affected_people, both are retained by the canonical temporal-fact sanitizer and added to the person entity index. Entity-detail records include the matching facts, and the existing regression test_affected_people_are_searchable asserts that a person named only as an affected target can be retrieved with that fact. Therefore the observed 12/21 omissions are upstream perception-field omissions, not loss in persistence or indexing. No target was filled by string heuristics or test annotations. Recheck the full 4,556-clip cache after the new parser is actually loaded; if the omission rate remains material, use a video-only targeted AI repair and preserve its raw output and source evidence.
