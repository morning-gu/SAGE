"""Strategy registry: 7 claim-conditioned anomaly strategies (v4, 2026-09-26).

History: 14 (v2) -> 8 (v3: item strategies/turn/blocked/chinrest removed on
empirical decisive rates) -> 7 (v4: away removed).

away was removed during development: its semantics contradict the
annotation convention.  The dataset's away means the seat is EMPTY
(student left), but the strategy checked "any person standing" -- when
the student leaves, everyone else visible is seated, so seated_frac =
1.0 fires a contradiction on exactly the frames where the claim is
true.  Its raw_evidence was also a point mass (283/284 val frames at
1.0), so the ECDF quantile gate could not discriminate.  The VLM
already solves this class at F1 0.98.

The kept strategies target the VLM's weak classes (slope, recline,
eyesclosed, tilt) plus the reliably decisive pose/face checks (lookup,
bowed, prone).  All emit a continuous raw_evidence (contradiction-
oriented: larger = stronger contradiction of the claim) for the ECDF
normalization layer (sage.veto §3), which additionally guards against
degenerate point-mass pools (v4).

Normal has no strategy by design (frozen decision 3): the veto layer only
checks VLM anomaly claims, so the default state is never verified and
never vetoed.
"""
from __future__ import annotations

from sage_recheck.strategies.bowed import BowedStrategy
from sage_recheck.strategies.eyesclosed import EyesclosedStrategy
from sage_recheck.strategies.lookup import LookupStrategy
from sage_recheck.strategies.prone import ProneStrategy
from sage_recheck.strategies.recline import ReclineStrategy
from sage_recheck.strategies.slope import SlopeStrategy
from sage_recheck.strategies.tilt import TiltStrategy

PRIMARY_STRATEGIES = [
    ProneStrategy(),
    BowedStrategy(),
    TiltStrategy(),
    SlopeStrategy(),
    LookupStrategy(),
    ReclineStrategy(),
    EyesclosedStrategy(),
]

# Classes the veto layer can act on (claims outside this set are ignored).
VETO_SUPPORTED = frozenset(s.behavior_name for s in PRIMARY_STRATEGIES)
