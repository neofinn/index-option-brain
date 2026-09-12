"""Contract specifications from Dhan's public instrument master.

Run against a recorded slice of the real file (see `recorded/README.md`), so
these assert against what Dhan actually publishes rather than a convenient
shape.

The bug that motivated all of this: the bundled NIFTY lot size was 75, and
the exchange's record says 65. Several tests below exist purely to make that
class of drift impossible to reintroduce quietly.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from index_option_brain.contracts.enums import OptionType
from index_option_brain.data.adapters.base import DataAdapterError
from index_option_brain.data.adapters.nse_public import (
    DEFAULT_INDEX_CONFIG,
    index_config_from_master,
)
from index_option_brain.data.dhan_instruments import (
    SCRIP_MASTER_URL,
    DhanInstrumentMaster,
    parse_scrip_master,
)
from index_option_brain.data.http import HttpResponse, RecordedSession

RECORDED = Path(__file__).parent / "recorded" / "dhan_scrip_master.csv"
NEAR_EXPIRY = date(2026, 9, 8)


@pytest.fixture
def master() -> DhanInstrumentMaster:
    return DhanInstrumentMaster(records=parse_scrip_master(RECORDED.read_text()))


class TestParsing:
    def test_only_indices_and_index_options_are_kept(self, master):
        """The full file is ~198,000 rows, almost all single-stock and
        currency contracts. Keeping them would be 25 MB of memory for data
        this system never looks at."""
        kinds = {record.instrument for record in master.records}
        assert kinds == {"INDEX", "OPTIDX"}

    def test_equity_noise_in_the_file_is_skipped(self, master):
        """The recorded slice deliberately contains equity rows."""
        assert not any("EQUITY" == r.instrument for r in master.records)
        assert master.records

    def test_a_malformed_row_does_not_abort_the_load(self):
        """One bad line in 198,000 must not cost the whole file."""
        good = RECORDED.read_text().splitlines()
        broken = "\n".join([good[0], "garbage,row,with,too,few", *good[1:]])
        assert parse_scrip_master(broken)

    def test_an_empty_file_parses_to_nothing_rather_than_guessing(self):
        assert parse_scrip_master("SEM_INSTRUMENT_NAME\n") == []


class TestLotSize:
    def test_nifty_lot_size_is_sixty_five(self, master):
        """The correction. Not 75, which is what was hardcoded."""
        assert master.lot_size("NIFTY") == 65

    def test_banknifty_lot_size_is_thirty(self, master):
        assert master.lot_size("BANKNIFTY") == 30

    def test_the_bundled_fallback_now_agrees_with_the_exchange(self, master):
        """A pin on the fixed bug: if the fallback table drifts from the
        exchange record again, this fails."""
        for symbol in ("NIFTY", "BANKNIFTY"):
            assert DEFAULT_INDEX_CONFIG[symbol].lot_size == master.lot_size(symbol)

    def test_a_lot_size_can_be_asked_for_per_expiry(self, master):
        """A revision applies to newly introduced contracts while existing
        ones keep the old size until they expire. Ignoring that during a
        transition mis-sizes the near expiry by exactly the revision."""
        assert master.lot_size("NIFTY", NEAR_EXPIRY) == 65

    def test_a_revision_in_flight_refuses_rather_than_picking_one(self):
        """Whichever value it picked would be wrong for some expiry, and
        quietly picking one mis-sizes real orders.

        Simulated by giving the far expiry a different lot size, which is
        exactly the shape of a real revision: new contracts get the new size
        while listed ones keep the old until they expire."""
        rows = RECORDED.read_text().splitlines()
        mutated = [
            row.replace(",65.0,", ",75.0,", 1)
            if "2026-09-29" in row and ",65.0," in row
            else row
            for row in rows[1:]
        ]
        conflicted = DhanInstrumentMaster(
            records=parse_scrip_master("\n".join([rows[0], *mutated]))
        )
        with pytest.raises(DataAdapterError, match="revision is in progress"):
            conflicted.lot_size("NIFTY")

    def test_a_revision_in_flight_still_answers_for_a_named_expiry(self):
        """Which is the point of the expiry argument: the near weekly still
        has one correct answer even mid-revision."""
        rows = RECORDED.read_text().splitlines()
        mutated = [
            row.replace(",65.0,", ",75.0,", 1)
            if "2026-09-29" in row and ",65.0," in row
            else row
            for row in rows[1:]
        ]
        conflicted = DhanInstrumentMaster(
            records=parse_scrip_master("\n".join([rows[0], *mutated]))
        )
        assert conflicted.lot_size("NIFTY", NEAR_EXPIRY) == 65
        assert conflicted.lot_size("NIFTY", date(2026, 9, 29)) == 75

    def test_an_unlisted_underlying_raises(self, master):
        with pytest.raises(DataAdapterError, match="No index options listed"):
            master.lot_size("NOTANINDEX")


class TestTickSize:
    def test_tick_size_is_converted_from_paise_to_rupees(self, master):
        """Dhan reports 5.0000, meaning five paise. Sending an order priced
        on the unscaled figure would be off by a factor of a hundred."""
        assert master.tick_size("NIFTY") == Decimal("0.05")

    def test_the_raw_file_really_does_report_paise(self):
        """Guards the conversion against the file changing units."""
        raw = RECORDED.read_text()
        assert ",5.0000," in raw


class TestIdentifiers:
    def test_index_security_ids_are_read(self, master):
        assert master.index_security_id("NIFTY") == "13"
        assert master.index_security_id("BANKNIFTY") == "25"

    def test_an_unknown_index_raises(self, master):
        with pytest.raises(DataAdapterError, match="No index instrument"):
            master.index_security_id("NOTANINDEX")

    def test_one_contract_resolves_to_its_security_id(self, master):
        """What an order actually needs: a broker takes the security id, not
        a strike and an expiry."""
        options = master.options_for("NIFTY")
        sample = next(o for o in options if o.option_type is OptionType.CE)
        found = master.option("NIFTY", sample.expiry, sample.strike, OptionType.CE)
        assert found is not None
        assert found.security_id == sample.security_id
        assert found.lot_size == 65

    def test_a_contract_that_is_not_listed_returns_none(self, master):
        assert (
            master.option("NIFTY", NEAR_EXPIRY, Decimal(99999), OptionType.CE) is None
        )


class TestExpiriesAndStrikes:
    def test_expiries_come_back_sorted(self, master):
        expiries = master.expiries("NIFTY")
        assert expiries == sorted(expiries)

    def test_nifty_weeklies_are_tuesdays(self, master):
        assert master.expiries("NIFTY")[0].strftime("%A") == "Tuesday"

    def test_the_strike_step_is_derived(self, master):
        assert master.strike_step("NIFTY", NEAR_EXPIRY) == Decimal(50)

    def test_too_few_strikes_gives_no_step(self, master):
        assert master.strike_step("NIFTY", date(2099, 1, 1)) is None


class TestBuildingTheAdapterConfig:
    def test_it_produces_a_verified_contract_table(self, master):
        config = index_config_from_master(master)
        assert config["NIFTY"].lot_size == 65
        assert config["NIFTY"].strike_step == Decimal(50)
        assert config["NIFTY"].tick_size == Decimal("0.05")
        assert config["NIFTY"].nse_index_name == "NIFTY 50"

    def test_an_unlisted_symbol_is_skipped_not_defaulted(self, master):
        """An index whose contract size cannot be verified must not enter the
        sizing path at all."""
        config = index_config_from_master(master, symbols=("NIFTY", "MIDCPNIFTY"))
        assert "NIFTY" in config
        assert "MIDCPNIFTY" not in config

    def test_an_unknown_symbol_is_skipped(self, master):
        assert index_config_from_master(master, symbols=("NOTANINDEX",)) == {}


class TestLoading:
    async def test_it_loads_over_http_without_credentials(self):
        """No authentication. That is what makes it usable before any
        subscription."""
        session = RecordedSession({"api-scrip-master": RECORDED.read_text()})
        loaded = await DhanInstrumentMaster.load(session)
        assert loaded.lot_size("NIFTY") == 65
        assert loaded.loaded_at is not None
        assert "Authorization" not in " ".join(session.requests)

    async def test_a_server_error_raises_rather_than_returning_nothing(self):
        session = RecordedSession({"api-scrip-master": HttpResponse(503, "down")})
        with pytest.raises(DataAdapterError, match="HTTP 503"):
            await DhanInstrumentMaster.load(session)

    async def test_a_format_change_raises_rather_than_silently_emptying(self):
        """Zero contracts parsed means the file changed shape. Continuing
        would produce no lot sizes at all, which downstream reads as an
        unlisted index rather than as a broken loader."""
        session = RecordedSession({"api-scrip-master": "col_a,col_b\n1,2\n"})
        with pytest.raises(DataAdapterError, match="zero index contracts"):
            await DhanInstrumentMaster.load(session)

    async def test_a_transport_failure_becomes_a_data_adapter_error(self):
        with pytest.raises(DataAdapterError, match="Could not fetch"):
            await DhanInstrumentMaster.load(RecordedSession({}))

    def test_the_url_is_dhans_public_cdn(self):
        assert SCRIP_MASTER_URL.startswith("https://images.dhan.co/")
