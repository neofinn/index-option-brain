"""Contract specifications from Dhan's public instrument master.

Lot size, tick size and security id for every listed contract, from a CSV
Dhan publishes with **no authentication**. That last part matters: this is
usable before any subscription, and it removes the worst piece of guesswork
in the system.

Why this exists
---------------
Lot sizes are revised by exchange circular, and no NSE public endpoint
exposes them — the derivative-quote endpoint that carries `marketLot` returns
404 to an automated client. So they were hardcoded, with a comment saying to
verify them.

They were wrong. The hardcoded NIFTY lot size was 75; this file says **65**,
on every listed expiry from the near weekly to 2031. Every position size,
max loss, margin estimate and exposure figure computed from 75 was about 15%
overstated.

Worse, the Execution Gate's LOT_SIZE_VALID check could not catch it. That
check compares a leg's `contract.lot_size` against `IndexSpec.lot_size` — and
both came from the same wrong constant, so they agreed with each other. A
consistency check between two copies of one number is not a correctness
check, which is the general lesson: contract specifications have to come from
the exchange's own record, not from a constant with a warning comment
attached.

Refresh it daily. Contracts are added and revised, and a stale master is the
same class of bug in slower motion.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from itertools import pairwise

from index_option_brain.contracts.enums import OptionType
from index_option_brain.data.adapters.base import DataAdapterError
from index_option_brain.data.http import HttpError, HttpSession, HttpxSession

SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"

# The file is ~25 MB and ~198,000 rows, almost all of it single-stock and
# currency contracts. Only these two instrument types are kept, which is about
# 18,000 rows — small enough to hold in memory and search directly.
INDEX_INSTRUMENT = "INDEX"
INDEX_OPTION_INSTRUMENT = "OPTIDX"

# Dhan reports SEM_TICK_SIZE in paise: index options come through as 5.0000,
# which is 5 paise, or 0.05 rupees. Sending an order priced on the unscaled
# figure would be off by a factor of a hundred.
_PAISE_PER_RUPEE = Decimal(100)


@dataclass(frozen=True)
class InstrumentRecord:
    """One listed contract, as the exchange describes it."""

    security_id: str
    exchange: str
    segment: str
    instrument: str
    trading_symbol: str
    underlying: str
    lot_size: int
    tick_size: Decimal
    """In rupees, converted from the paise the file reports."""
    expiry: date | None = None
    strike: Decimal | None = None
    option_type: OptionType | None = None

    @property
    def is_index(self) -> bool:
        return self.instrument == INDEX_INSTRUMENT

    @property
    def is_index_option(self) -> bool:
        return self.instrument == INDEX_OPTION_INSTRUMENT


def _parse_decimal(raw: str) -> Decimal | None:
    text = raw.strip()
    if not text:
        return None
    try:
        return Decimal(text)
    except (ArithmeticError, ValueError):
        return None


def _parse_expiry(raw: str) -> date | None:
    """Expiry arrives as "2026-09-08 14:30:00", occasionally date-only."""
    text = raw.strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()  # noqa: DTZ007 - date only
        except ValueError:
            continue
    return None


def _underlying_from_symbol(trading_symbol: str, instrument: str) -> str:
    """The underlying an option is written on.

    Index option trading symbols are `UNDERLYING-MonYYYY-STRIKE-CE`, and no
    index underlying contains a hyphen, so the first segment is the name. For
    an index row the symbol is the underlying already.
    """
    if instrument == INDEX_OPTION_INSTRUMENT:
        return trading_symbol.split("-", 1)[0].strip().upper()
    return trading_symbol.strip().upper()


def parse_scrip_master(content: str) -> list[InstrumentRecord]:
    """Parse the master, keeping only indices and index options.

    A malformed row is skipped rather than aborting the load: one bad line in
    198,000 must not cost the whole file, and the rows this system needs are a
    small, well-formed subset.
    """
    records: list[InstrumentRecord] = []
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        instrument = (row.get("SEM_INSTRUMENT_NAME") or "").strip()
        if instrument not in (INDEX_INSTRUMENT, INDEX_OPTION_INSTRUMENT):
            continue

        security_id = (row.get("SEM_SMST_SECURITY_ID") or "").strip()
        trading_symbol = (row.get("SEM_TRADING_SYMBOL") or "").strip()
        if not security_id or not trading_symbol:
            continue

        lot_raw = _parse_decimal(row.get("SEM_LOT_UNITS") or "")
        tick_raw = _parse_decimal(row.get("SEM_TICK_SIZE") or "")
        if lot_raw is None or lot_raw <= 0:
            continue

        option_type: OptionType | None = None
        raw_type = (row.get("SEM_OPTION_TYPE") or "").strip().upper()
        if raw_type in ("CE", "PE"):
            option_type = OptionType(raw_type)

        records.append(
            InstrumentRecord(
                security_id=security_id,
                exchange=(row.get("SEM_EXM_EXCH_ID") or "").strip(),
                segment=(row.get("SEM_SEGMENT") or "").strip(),
                instrument=instrument,
                trading_symbol=trading_symbol,
                underlying=_underlying_from_symbol(trading_symbol, instrument),
                lot_size=int(lot_raw),
                tick_size=(
                    tick_raw / _PAISE_PER_RUPEE if tick_raw else Decimal("0.05")
                ),
                expiry=_parse_expiry(row.get("SEM_EXPIRY_DATE") or ""),
                strike=_parse_decimal(row.get("SEM_STRIKE_PRICE") or ""),
                option_type=option_type,
            )
        )
    return records


@dataclass
class DhanInstrumentMaster:
    """Loaded contract specifications, queryable by underlying and contract."""

    records: list[InstrumentRecord] = field(default_factory=list)
    loaded_at: datetime | None = None

    _by_underlying: dict[str, list[InstrumentRecord]] = field(default_factory=dict)
    _indices: dict[str, InstrumentRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._reindex()

    def _reindex(self) -> None:
        self._by_underlying = {}
        self._indices = {}
        for record in self.records:
            self._by_underlying.setdefault(record.underlying, []).append(record)
            if record.is_index and record.underlying not in self._indices:
                self._indices[record.underlying] = record

    @classmethod
    async def load(
        cls,
        session: HttpSession | None = None,
        *,
        url: str = SCRIP_MASTER_URL,
    ) -> DhanInstrumentMaster:
        """Fetch and parse the master. No credentials required."""
        owned = session is None
        client = session or HttpxSession(timeout=120.0)
        try:
            response = await client.get(url)
        except HttpError as exc:
            raise DataAdapterError(
                f"Could not fetch the Dhan instrument master: {exc}"
            ) from exc
        finally:
            if owned:
                await client.aclose()

        if not response.is_ok:
            raise DataAdapterError(
                f"Dhan instrument master returned HTTP {response.status_code}"
            )
        records = parse_scrip_master(response.text)
        if not records:
            raise DataAdapterError(
                "Dhan instrument master parsed to zero index contracts — the "
                "file format has probably changed, and using it would silently "
                "produce no lot sizes at all"
            )
        return cls(records=records, loaded_at=datetime.now(tz=None).astimezone())

    # ------------------------------------------------------------- queries

    @property
    def underlyings(self) -> list[str]:
        return sorted(self._by_underlying)

    def index(self, symbol: str) -> InstrumentRecord:
        try:
            return self._indices[symbol.upper()]
        except KeyError:
            raise DataAdapterError(
                f"No index instrument named {symbol!r} in the Dhan master"
            ) from None

    def index_security_id(self, symbol: str) -> str:
        """Dhan's id for the index — 13 for NIFTY, 25 for BANKNIFTY."""
        return self.index(symbol).security_id

    def options_for(self, underlying: str) -> list[InstrumentRecord]:
        return [
            record
            for record in self._by_underlying.get(underlying.upper(), [])
            if record.is_index_option
        ]

    def lot_size(self, underlying: str, expiry: date | None = None) -> int:
        """The lot size the exchange lists for this underlying.

        `expiry` narrows it, because a lot-size revision applies to newly
        introduced contracts while existing ones keep the old size until they
        expire. Ignoring that during a transition would mis-size the near
        expiry by exactly the revision.
        """
        options = self.options_for(underlying)
        if not options:
            raise DataAdapterError(
                f"No index options listed for {underlying!r} in the Dhan master"
            )
        if expiry is not None:
            for_expiry = [record for record in options if record.expiry == expiry]
            if for_expiry:
                return for_expiry[0].lot_size
        sizes = {record.lot_size for record in options}
        if len(sizes) > 1:
            # A revision is in flight. Refusing is right: whichever value is
            # picked will be wrong for some expiry, and quietly picking one
            # mis-sizes real orders.
            raise DataAdapterError(
                f"{underlying} lists several lot sizes {sorted(sizes)} across "
                "expiries, which means a revision is in progress. Pass the "
                "expiry to get the size that applies to it."
            )
        return sizes.pop()

    def tick_size(self, underlying: str) -> Decimal:
        options = self.options_for(underlying)
        if not options:
            raise DataAdapterError(
                f"No index options listed for {underlying!r} in the Dhan master"
            )
        return options[0].tick_size

    def expiries(self, underlying: str) -> list[date]:
        return sorted(
            {
                record.expiry
                for record in self.options_for(underlying)
                if record.expiry is not None
            }
        )

    def strike_step(self, underlying: str, expiry: date) -> Decimal | None:
        """The modal gap between adjacent listed strikes for one expiry."""
        strikes = sorted(
            {
                record.strike
                for record in self.options_for(underlying)
                if record.expiry == expiry and record.strike is not None
            }
        )
        if len(strikes) < 3:
            return None
        gaps: dict[Decimal, int] = {}
        for first, second in pairwise(strikes):
            gap = second - first
            if gap > 0:
                gaps[gap] = gaps.get(gap, 0) + 1
        if not gaps:
            return None
        return max(gaps, key=lambda gap: gaps[gap])

    def option(
        self,
        underlying: str,
        expiry: date,
        strike: Decimal,
        option_type: OptionType,
    ) -> InstrumentRecord | None:
        """One specific contract, for the security id an order needs."""
        for record in self.options_for(underlying):
            if (
                record.expiry == expiry
                and record.strike == strike
                and record.option_type is option_type
            ):
                return record
        return None
