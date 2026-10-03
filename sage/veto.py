"""Claim-conditioned geometric veto (Algorithm 1) — single source of truth.

Implements the frozen design in docs/veto-algorithm.md (v2, 2026-09-22):

  §3  per-strategy validation ECDF normalization of raw evidence, with the
      n_min=30 calibration floor and the v4 degenerate-pool guard
      (uncalibrated or point-mass strategies never veto),
  §4  Algorithm 1: veto on 'contradicted', gated by one of two validation-
      frozen authorities: the uniform quantile gate ~e >= tau (v3), or —
      for strategies whose contradiction region is the majority region of
      the pool (slope/tilt) — a per-strategy frozen raw floor frozen on
      validation claims under the same FVR <= delta rule (v5); normal is
      outside the veto scope; fallback to the default state (normal) via a
      Stage-1 rule,
  §5  veto-decision quality metrics: VP / FVR / TPK / coverage, plus
      primary-demotion outcomes.

The veto layer only REMOVES labels (no-add invariant) and only runs on VLM
anomaly claims (claim-conditioned contract; see sage_recheck/enums.py).
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from pathlib import Path

from sage.taxonomy import (
    CODE_TO_NAME,
    GEOMETRIC_CUED,
    NORMAL_CODE,
    SEMANTIC_CUED,
)

# Frozen protocol constants (docs/veto-algorithm.md §7).
DELTA = 0.05                 # FVR safety floor (preregistered)
N_MIN = 30                   # per-strategy validation calibration floor
TAU_GRID = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50..0.95
DEGENERATE_MASS = 0.5        # single-value pool share that disables a strategy


# ---------------------------------------------------------------------------
# Evidence-strength normalization (§3)
# ---------------------------------------------------------------------------

class ECDFStore:
    """Frozen per-strategy empirical CDFs mapping raw evidence -> [0, 1].

    Fitted on validation frames; applied frozen to test frames.  A strategy
    is disabled — its raw evidence maps to None, i.e. no veto authority —
    when either floor is violated:

    * n_min: fewer than N_MIN validation invocations (insufficient
      calibration), or
    * degenerate pool (v4): a single value accounts for more than
      DEGENERATE_MASS of the samples.  A point-mass pool makes the
      quantile gate meaningless: the dominant value sits at tilde ~= 1.0
      regardless of tau, so any contradiction landing on it vetoes at
      every tau.  This is exactly how the (since removed) away strategy
      failed -- seated_frac was 1.0 on 283/284 val frames, so every
      contradiction landing on the modal value was a false kill.
    """

    def __init__(self, samples: dict[str, list[float]],
                 disabled: set[str] | None = None,
                 disabled_reasons: dict[str, str] | None = None):
        self._sorted = {k: sorted(v) for k, v in samples.items()}
        self.disabled = disabled or set()
        self.disabled_reasons = disabled_reasons or {}

    @classmethod
    def fit(cls, records: list[dict]) -> "ECDFStore":
        """Fit from evidence records: [{name, raw_evidence, ...}, ...].

        Only records with a numeric raw_evidence contribute.  Strategies
        failing the N_MIN or DEGENERATE_MASS floors are disabled (never
        veto at test time).
        """
        from collections import Counter

        samples: dict[str, list[float]] = {}
        for r in records:
            raw = r.get("raw_evidence")
            if raw is None:
                continue
            samples.setdefault(r["name"], []).append(float(raw))
        kept: dict[str, list[float]] = {}
        disabled: set[str] = set()
        reasons: dict[str, str] = {}
        for name, vals in samples.items():
            if len(vals) < N_MIN:
                disabled.add(name)
                reasons[name] = f"n_min ({len(vals)} < {N_MIN})"
                continue
            modal = Counter(round(v, 6) for v in vals).most_common(1)[0][1]
            if modal / len(vals) > DEGENERATE_MASS:
                disabled.add(name)
                reasons[name] = (f"degenerate pool ({modal}/{len(vals)} "
                                 f"= {modal / len(vals):.0%} at one value)")
                continue
            kept[name] = vals
        return cls(kept, disabled, reasons)

    def apply(self, name: str, raw: float | None) -> float | None:
        """Quantile of raw under the strategy's frozen ECDF, or None."""
        if raw is None or name in self.disabled:
            return None
        vals = self._sorted.get(name)
        if not vals:
            return None
        n = len(vals)
        if raw <= vals[0]:
            return 0.0 if raw < vals[0] else 1.0 / n
        if raw >= vals[-1]:
            return 1.0
        # Piecewise-linear interpolation between order statistics: order
        # statistic i (0-based) sits at quantile (i+1)/n.
        i = bisect.bisect_left(vals, raw)
        if vals[i] == raw:
            j = bisect.bisect_right(vals, raw) - 1
            return (j + 1) / n
        lo_q, hi_q = i / n, (i + 1) / n
        lo_v, hi_v = vals[i - 1], vals[i]
        # interpolate between the two surrounding order statistics
        frac = (raw - lo_v) / (hi_v - lo_v)
        return lo_q + frac * (hi_q - lo_q)

    def save(self, path: str | Path) -> None:
        payload = {
            "samples": self._sorted,
            "disabled": sorted(self.disabled),
            "disabled_reasons": self.disabled_reasons,
            "n_min": N_MIN,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ECDFStore":
        payload = json.loads(Path(path).read_text())
        return cls(payload["samples"], set(payload.get("disabled", [])),
                   payload.get("disabled_reasons", {}))


# ---------------------------------------------------------------------------
# Algorithm 1 (§4)
# ---------------------------------------------------------------------------

@dataclass
class VetoRecord:
    """One veto decision (a removed (frame, class) claim)."""
    name: str
    code: str
    verdict: str                 # 'contradicted'
    raw_evidence: float | None
    tilde_e: float | None        # normalized evidence strength
    was_primary: bool
    correct: bool | None = None  # None without ground truth
    gate: str = "quantile"       # 'quantile' | 'frozen_threshold' (v5)


@dataclass
class VetoOutcome:
    """Result of applying the veto layer to one frame."""
    claims: list[str] = field(default_factory=list)
    claim_verdicts: dict[str, str] = field(default_factory=dict)
    vetoed: list[VetoRecord] = field(default_factory=list)
    primary_demoted: bool = False
    final_code: str = ""
    final_name: str = ""
    corrected_codes: list[str] = field(default_factory=list)
    demotion_outcome: str | None = None  # improved|degraded|neutral (w/ gt)

    @property
    def decisive(self) -> int:
        """Claims that received a decisive evidence check."""
        return sum(1 for v in self.claim_verdicts.values()
                   if v in ("contradicted", "consistent"))


def apply_veto(probs: dict[str, float],
               primary_code: str,
               evidence: dict[str, "EvidenceResult"],
               tau: float,
               ecdf: ECDFStore | None = None,
               threshold: float = 0.5,
               raw_floors: dict[str, float] | None = None,
               gt_labels: dict[str, bool] | None = None,
               gt_primary: str | None = None) -> VetoOutcome:
    """Apply Algorithm 1 to one frame.

    Args:
        probs: code-keyed VLM probabilities (from the first-token sigmoid).
        primary_code: VLM primary prediction (argmax code).
        evidence: name-keyed claim-conditioned evidence results (only VLM
            claims were checked; see StrategyRunner.run).
        tau: frozen veto threshold (quantile scale); gates every veto that
            runs on the quantile authority.
        ecdf: frozen normalization store; None disables all quantile vetoes
            (no calibration, no veto authority — uniform gating v3).
        threshold: multi-label decision threshold for claim extraction.
        raw_floors: optional per-strategy frozen raw floors (v5) — for
            strategies whose raw evidence is directly interpretable and
            whose contradiction region is the MAJORITY region of the
            calibration pool (slope/tilt: a level head is the normal
            state), the quantile gate is structurally unable to
            discriminate (it degenerates to exact-0 point masses).  For
            these, the freeze stage froze a raw floor on validation claims
            under the same FVR <= delta rule; the floor REPLACES the
            quantile gate for that strategy.  Veto authority still
            requires a contradicted verdict and a numeric raw value.
        gt_labels / gt_primary: optional ground truth (name-keyed) enabling
            per-record correctness and demotion-outcome annotation.
    """
    from sage_recheck.enums import EvidenceVerdict

    out = VetoOutcome()
    # Claims: VLM-predicted anomaly positives (normal is out of scope).
    # Claims of classes without a strategy (no evidence entry — the registry
    # was trimmed to 7 in v4) are outside the veto scope and are not counted
    # in the coverage denominator.
    claimed = [c for c, p in probs.items()
               if p >= threshold and c != NORMAL_CODE]
    checked = [c for c in claimed if CODE_TO_NAME.get(c, c) in evidence]
    out.claims = [CODE_TO_NAME.get(c, c) for c in checked]
    out.claim_verdicts = {}
    vetoed_names: set[str] = set()

    for code in checked:
        name = CODE_TO_NAME.get(code, code)
        r = evidence.get(name)
        out.claim_verdicts[name] = r.verdict.value
        if r.verdict == EvidenceVerdict.INSUFFICIENT:
            continue
        if r.verdict != EvidenceVerdict.CONTRADICTED:
            continue
        is_primary = (code == primary_code)
        floor = raw_floors.get(name) if raw_floors else None
        if floor is not None:
            # Frozen-threshold authority (v5): raw >= floor, frozen on
            # validation claims with FVR <= delta.  Replaces the quantile
            # gate for this strategy; tilde is still computed for the
            # case record (observability) but not required.
            tilde = ecdf.apply(name, r.raw_evidence) if ecdf else None
            gate = "frozen_threshold"
            authority = r.raw_evidence is not None and r.raw_evidence >= floor
        else:
            # Uniform gating (v3): every veto requires a calibrated
            # strength ~e >= tau. An uncomputable strength (raw missing,
            # or the strategy disabled by the calibration floors) carries
            # no veto authority. This is what makes FVR controllable by
            # tau and the preregistered freeze rule satisfiable.
            tilde = ecdf.apply(name, r.raw_evidence) if ecdf else None
            gate = "quantile"
            authority = tilde is not None and tilde >= tau
        if not authority:
            continue
        correct = None if gt_labels is None else not gt_labels.get(name, False)
        out.vetoed.append(VetoRecord(
            name=name, code=code, verdict=r.verdict.value,
            raw_evidence=r.raw_evidence, tilde_e=tilde,
            was_primary=is_primary, correct=correct, gate=gate))
        vetoed_names.add(name)
        if is_primary:
            out.primary_demoted = True

    out.corrected_codes = [c for c in claimed
                           if CODE_TO_NAME.get(c, c) not in vetoed_names]

    # Primary fallback (Stage-1 rule, not a Stage-2 judgment).
    if out.primary_demoted:
        remaining = [c for c in out.corrected_codes]
        if remaining:
            final = max(remaining, key=lambda c: probs.get(c, 0.0))
        else:
            final = NORMAL_CODE
        if gt_primary is not None:
            if CODE_TO_NAME.get(final, final) == gt_primary:
                out.demotion_outcome = "improved"
            elif CODE_TO_NAME.get(primary_code, primary_code) == gt_primary:
                out.demotion_outcome = "degraded"
            else:
                out.demotion_outcome = "neutral"
    else:
        final = primary_code
    out.final_code = final
    out.final_name = CODE_TO_NAME.get(final, final)
    return out


# ---------------------------------------------------------------------------
# Veto-decision quality metrics (§5)
# ---------------------------------------------------------------------------

def aggregate_metrics(outcomes: list[VetoOutcome],
                      gt_lists: list[dict[str, bool]]) -> dict:
    """Compute VP / FVR / TPK / coverage + demotion stats over frames.

    Args:
        outcomes: per-frame veto outcomes (gt-annotated).
        gt_lists: per-frame ground-truth label sets (name-keyed).
    """
    vetoes = [r for o in outcomes for r in o.vetoed]
    correct = [r for r in vetoes if r.correct]
    false_vetoes = [r for r in vetoes if r.correct is False]

    total_true_pos = sum(1 for gt in gt_lists
                         for n, v in gt.items() if v and n != "normal")
    total_claims = sum(len(o.claims) for o in outcomes)
    decisive = sum(o.decisive for o in outcomes)
    demotions = [o for o in outcomes if o.primary_demoted]
    dem_out = {"improved": 0, "degraded": 0, "neutral": 0}
    for o in demotions:
        if o.demotion_outcome:
            dem_out[o.demotion_outcome] += 1

    vp = (len(correct) / len(vetoes)) if vetoes else None
    return {
        "n_frames": len(outcomes),
        "n_vetoes": len(vetoes),
        "n_vetoes_correct": len(correct),
        "n_vetoes_false": len(false_vetoes),
        "veto_precision": vp,
        "false_veto_rate": (1.0 - vp) if vp is not None else None,
        "true_positive_kill_rate": (len(false_vetoes) / total_true_pos)
        if total_true_pos else 0.0,
        "n_claims": total_claims,
        "n_claims_decisive": decisive,
        "coverage": (decisive / total_claims) if total_claims else None,
        "n_primary_demotions": len(demotions),
        "demotion_outcomes": dem_out,
    }


def metrics_by_group(outcomes: list[VetoOutcome],
                     gt_lists: list[dict[str, bool]]) -> dict:
    """Decompose veto metrics into semantic-/geometric-cued groups."""
    out = {}
    for group, classes in (("semantic", SEMANTIC_CUED),
                           ("geometric", GEOMETRIC_CUED)):
        keep = set(classes)
        sub_out, sub_gt = [], []
        for o, gt in zip(outcomes, gt_lists):
            o2 = VetoOutcome(
                claims=[c for c in o.claims if c in keep],
                claim_verdicts={k: v for k, v in o.claim_verdicts.items()
                                if k in keep},
                vetoed=[r for r in o.vetoed if r.name in keep],
                primary_demoted=o.primary_demoted,
                final_code=o.final_code, final_name=o.final_name,
                corrected_codes=o.corrected_codes,
                demotion_outcome=o.demotion_outcome)
            sub_out.append(o2)
            sub_gt.append({k: v for k, v in gt.items() if k in keep})
        out[group] = aggregate_metrics(sub_out, sub_gt)
    return out
