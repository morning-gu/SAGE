"""SAGE small-model evidence layer (claim-conditioned geometric checks).

Public entry points: ``StrategyRunner`` (sage_recheck.strategies.runner)
for claim-conditioned evidence checks, and ``sage.veto.apply_veto`` for the
veto decision. Normal has no strategy by design (frozen decision 3).
"""
from sage_recheck.strategies.runner import StrategyRunner

__all__ = ["StrategyRunner"]
