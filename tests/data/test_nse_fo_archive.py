"""NSE F&O bhavcopy adapter — the free historical option chain.

The behaviours pinned here are the ones whose failure would be silent: a
throttled session vanishing as a holiday, and a settlement price presenting
itself as a two-sided book.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date
from decimal import Decimal

import pytest

from index_option_brain.contracts.enums import OptionType
from index_option_brain.data.adapters.base import DataAdapterError
from index_option_brain.data.adapters.nse_fo_archive import (
    EARLIEST_UDIFF_SESSION,
    NseFoArchiveAdapter,
    bhavcopy_url,
    parse_bhavcopy,
    unzip_bhavcopy,
)
from index_option_brain.data.http import HttpResponse

COLUMNS = (
    "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,"
    "FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,"
    "LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,"
    "TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4"
)


def row(
    *,
    symbol: str = "NIFTY",
    expiry: str = "2026-09-15",
    strike: str = "23500",
    opt: str = "CE",
    close: str = "77.00",
    settle: str = "77.00",
    oi: str = "11709555",
    oi_chg: str = "-120",
    vol: str = "2977798",
    lot: str = "65",
) -> str:
    return (
        f"2026-09-10,2026-09-10,FO,NSE,STO,1,,{symbol},,{expiry},{expiry},{strike},{opt},"
        f"{symbol}{expiry},70,80,69,{close},{close},70,23477.80,{settle},{oi},{oi_chg},"
        f"{vol},100,5000,F,{lot},,,,,"
    )


def csv_body(*rows: str) -> str:
    return "\n".join((COLUMNS, *rows)) + "\n"


def zipped(body: str, *, name: str = "BhavCopy.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr(name, body)
    return buf.getvalue()


class StubSession:
    """Answers a scripted sequence of responses and counts the requests."""

    def __init__(self, responses: list[HttpResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    async def get(self, url: str, **_: object) -> HttpResponse:
        self.calls.append(url)
        return self._responses.pop(0) if self._responses else HttpResponse(403, "no")

    async def post(self, *a: object, **k: object) -> HttpResponse:  # pragma: no cover
        raise NotImplementedError

    async def delete(self, *a: object, **k: object) -> HttpResponse:  # pragma: no cover
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


def ok(payload: bytes) -> HttpResponse:
    return HttpResponse(200, "", {}, payload)


class TestParsing:
    def test_builds_a_chain_for_the_nearest_forward_expiry(self) -> None:
        body = csv_body(
            row(strike="23500", opt="CE"),
            row(strike="23500", opt="PE", close="132.80", settle="132.80"),
            row(expiry="2026-09-22", strike="23500", opt="CE", close="150"),
        )
        state = parse_bhavcopy(body, underlying="NIFTY", day=date(2026, 9, 10))
        assert state.expiry == date(2026, 9, 15)
        assert len(state.chain) == 2
        assert state.available_expiries == [date(2026, 9, 15), date(2026, 9, 22)]
        call = next(q for q in state.chain if q.contract.option_type is OptionType.CE)
        assert call.ltp == Decimal("77.00")
        assert call.open_interest == 11709555
        assert call.open_interest_change == -120
        assert call.contract.lot_size == 65

    def test_a_settlement_price_never_presents_as_a_two_sided_book(self) -> None:
        """The single most dangerous thing this adapter could get wrong.

        A bhavcopy has no bid or ask. If they defaulted to the settlement
        price the spread would read as zero, the Strike Engine's liquidity
        filter would pass every strike however untraded, and the backtest
        would discover an edge that is purely an artefact of the fill
        assumption. Unmeasured must stay unmeasured.
        """
        state = parse_bhavcopy(
            csv_body(row()), underlying="NIFTY", day=date(2026, 9, 10)
        )
        quote = state.chain[0]
        assert quote.bid is None
        assert quote.ask is None
        assert quote.spread is None
        assert quote.relative_spread is None
        assert quote.mid == quote.ltp

    def test_an_unpriced_contract_is_dropped_not_zeroed(self) -> None:
        body = csv_body(row(strike="23500"), row(strike="21700", close="", settle=""))
        state = parse_bhavcopy(body, underlying="NIFTY", day=date(2026, 9, 10))
        assert [q.contract.strike for q in state.chain] == [Decimal(23500)]

    def test_an_explicit_expiry_that_is_absent_is_refused(self) -> None:
        with pytest.raises(DataAdapterError, match="Available"):
            parse_bhavcopy(
                csv_body(row()),
                underlying="NIFTY",
                day=date(2026, 9, 10),
                expiry=date(2026, 12, 31),
            )

    def test_an_unknown_underlying_says_so(self) -> None:
        with pytest.raises(DataAdapterError, match="no NIFTYBEES options"):
            parse_bhavcopy(
                csv_body(row()), underlying="NIFTYBEES", day=date(2026, 9, 10)
            )

    def test_futures_rows_are_not_mistaken_for_options(self) -> None:
        body = csv_body(row(), row(opt="", strike="0"))
        state = parse_bhavcopy(body, underlying="NIFTY", day=date(2026, 9, 10))
        assert len(state.chain) == 1


class TestUnzip:
    def test_an_html_interstitial_served_as_200_is_refused(self) -> None:
        with pytest.raises(DataAdapterError, match="not a zip"):
            unzip_bhavcopy(b"<html>Access Denied</html>", day=date(2026, 9, 10))


class TestFetching:
    async def test_a_throttled_weekday_is_retried_not_recorded_as_a_holiday(self) -> None:
        """NSE answers 403 both for a missing file and for a throttled client.

        Measured, not hypothesised: five sequential requests produced 403 on
        two weekdays that each returned 200 with ~1 MB moments later. Treating
        the first answer as final would delete real sessions from a backtest
        and leave the result looking complete.
        """
        payload = zipped(csv_body(row()))
        session = StubSession([HttpResponse(403, "no"), HttpResponse(403, "no"), ok(payload)])
        adapter = NseFoArchiveAdapter(session=session, retry_pause=0.0, polite_pause=0.0)
        assert await adapter.fetch_raw(date(2026, 9, 10)) == payload
        assert len(session.calls) == 3

    async def test_a_weekday_that_refuses_throughout_raises(self) -> None:
        session = StubSession([HttpResponse(403, "no")] * 4)
        adapter = NseFoArchiveAdapter(session=session, retry_pause=0.0, polite_pause=0.0)
        with pytest.raises(DataAdapterError, match="ambiguous"):
            await adapter.fetch_raw(date(2026, 9, 10))

    async def test_a_weekday_404_is_a_holiday(self) -> None:
        session = StubSession([HttpResponse(404, "")])
        adapter = NseFoArchiveAdapter(session=session, retry_pause=0.0, polite_pause=0.0)
        assert await adapter.fetch_raw(date(2026, 9, 10)) is None

    async def test_a_weekend_is_not_requested_at_all(self) -> None:
        session = StubSession([])
        adapter = NseFoArchiveAdapter(session=session)
        assert await adapter.fetch_raw(date(2026, 9, 13)) is None
        assert session.calls == []

    async def test_a_200_with_no_body_is_not_an_empty_session(self) -> None:
        session = StubSession([HttpResponse(200, "", {}, b"")])
        adapter = NseFoArchiveAdapter(session=session, retry_pause=0.0, polite_pause=0.0)
        with pytest.raises(DataAdapterError, match="empty payload is not an empty session"):
            await adapter.fetch_raw(date(2026, 9, 10))

    async def test_a_session_before_the_udiff_format_is_refused(self) -> None:
        adapter = NseFoArchiveAdapter(session=StubSession([]))
        earlier = date(EARLIEST_UDIFF_SESSION.year - 1, 1, 5)
        with pytest.raises(DataAdapterError, match="predates"):
            await adapter.fetch_raw(earlier)

    async def test_bulk_reports_the_days_it_could_not_resolve(self) -> None:
        """A run missing 12 holidays and one missing 40 throttled sessions must
        not look the same from the outside."""
        payload = zipped(csv_body(row()))
        session = StubSession([ok(payload), *[HttpResponse(403, "no")] * 4])
        adapter = NseFoArchiveAdapter(session=session, retry_pause=0.0, polite_pause=0.0)
        chains, unresolved = await adapter.get_many_option_chains(
            "NIFTY", [date(2026, 9, 10), date(2026, 9, 11)]
        )
        assert list(chains) == [date(2026, 9, 10)]
        assert unresolved == [date(2026, 9, 11)]


class TestUrl:
    def test_url_uses_the_udiff_naming(self) -> None:
        assert bhavcopy_url(date(2026, 9, 10)).endswith(
            "BhavCopy_NSE_FO_0_0_0_20260910_F_0000.csv.zip"
        )
