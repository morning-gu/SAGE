# Released verification artifacts

Pseudonymized, non-image-derived artifacts backing the
paper's headline numbers.  Regenerate with
`PYTHONPATH=. python scripts/export_review_artifacts.py` (requires the
withheld local records; the committed files are the frozen release).

## Contents

- `perframe_primary_s{42,43,44}.json` — per-frame primary correctness
  for the VLM-only and post-veto systems (fields: `vlm_primary`,
  `veto_primary`, `gt_primary`, `vlm_correct`, `veto_correct`,
  `primary_demoted`, `demotion_outcome`).  Recomputes: per-seed and
  pooled accuracy deltas, exact McNemar (pooled 7/1 and frame-level
  5/1), demotion outcomes (7 improved / 6 neutral / 1 degraded).
- `veto_decisions_s{42,43,44}.jsonl` — fired veto decisions only
  (23 rows pooled): `sample_id`, `class`, `verdict`, `raw_evidence`,
  `tilde_e`, `was_primary`, `correct`, `gate`.  Recomputes: veto
  precision 22/23, decision-level and frame-level views (10 distinct
  frames), TPK 1/639.
- `contradicted_decisions_s{42,43,44}.jsonl` — ALL contradicted claims
  (105 pooled rows), fired and withheld: `gate` is `frozen_threshold`
  (with `vetoed` true/false per the raw floor) or `quantile_tau`
  (authority never fired, `vetoed` false).
  Recomputes: the 74-claim quantile-path composition (64 true / 10
  false), the 8 gate-withheld slope claims (all false), the ungated
  counterfactual (31 vetoes / 30 correct / 1 false), and the aggregate
  wrong-claim clearance 22/40.
- `consistent_decisions_s{42,43,44}.jsonl` — ALL consistent verdicts on
  claimed classes (84 pooled rows: 29/26/29), the checkers'
  cleared-claim record.  Fields: `sample_id`, `class`, `verdict`,
  `raw_evidence`, `was_primary`, `claim_is_true` (consistent verdicts
  never reach the authority gate, so no `gate`/`vetoed` fields).
  Recomputes: the consistent-side precision 76/84 = 0.905 pooled
  (26/29, 24/26, 26/29 per seed), the complementary error side of the
  checkers that VP does not cover.  Filter discipline matches
  `contradicted_decisions`: only classes the VLM claimed.
- `e4_perframe_s42.json` — per-frame correctness of the four
  comparison variants (none / veto / fusion / removal-only fusion),
  mainline seed.  Recomputes: Table 4 accuracies, all four pairwise
  exact-McNemar tests, and the add/remove surgery counts.

`sample_id` values are stable frame identifiers; no image data, image
features, or annotation files are included.
