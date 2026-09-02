"""Margin estimation.

Indian F&O margin is SPAN + exposure, computed by the exchange and reported by
the broker. It is **not** something this module can know precisely, and
pretending otherwise would put a fabricated number into a sizing calculation —
so the default here is an explicit, conservative estimate, and a broker-backed
implementation is expected to replace it.

The estimate errs high on purpose. Under-estimating margin produces an order
the broker rejects at the worst possible moment; over-estimating produces a
smaller position than strictly necessary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from index_option_brain.contracts.strike import StrikeCandidate


class MarginModel(ABC):
    @abstractmethod
    def estimate(self, structure: StrikeCandidate, lots: int) -> Decimal:
        """Margin required to carry `lots` of this structure."""
        ...


class DefinedRiskMarginModel(MarginModel):
    """A conservative estimate for hedged, defined-risk option structures.

    * Long premium (a net debit) costs the debit and nothing more.
    * A hedged spread receives the exchange's spread benefit, so the
      requirement approximates its maximum loss. A buffer is added because
      the real figure moves with volatility and time to expiry.

    This is honest about being an approximation. It is not valid for naked
    short options, which is why `RiskLimits.allow_undefined_risk` defaults to
    False — an undefined-risk structure must not be sized off this model.
    """

    def __init__(self, buffer: Decimal = Decimal("1.15")) -> None:
        self._buffer = buffer

    def estimate(self, structure: StrikeCandidate, lots: int) -> Decimal:
        if lots <= 0:
            return Decimal(0)

        per_lot = (
            structure.net_premium
            if structure.net_premium > 0
            else structure.max_loss * self._buffer
        )
        return (per_lot * lots).quantize(Decimal("0.01"))
