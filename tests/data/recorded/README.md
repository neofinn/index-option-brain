# Recorded NSE payloads

These files are **real responses captured from NSE's public API** on
02-Sep-2026, trimmed to a subset of rows so a test failure is readable. No
value inside them was written by hand: prices, IVs, open interest and
timestamps are exactly what the exchange sent.

They exist because the fragile part of a live adapter is payload parsing, and
parsing can only be pinned down deterministically against a fixed payload.
Testing it against the live endpoint would make the suite fail whenever the
market is closed, the endpoint rate-limits, or NSE changes an unrelated field
— and a test that fails for reasons unrelated to the code gets deleted.

| File | Endpoint |
| --- | --- |
| `nse_all_indices.json` | `/api/allIndices`, trimmed to NIFTY 50, NIFTY BANK and INDIA VIX |
| `nse_contract_info.json` | `/api/option-chain-contract-info?symbol=NIFTY`, first six expiries |
| `nse_option_chain.json` | `/api/option-chain-v3`, eight strikes chosen to exercise every parsing branch |

The eight strikes are not arbitrary. Each one is a case the adapter has to get
right:

* **23900** — the ATM pair, tight two-sided book, IV published by NSE.
* **23850 / 23950** — the strikes either side, for delta ordering.
* **23600** — a tight book (344.25 / 345.45) that NSE published **no** IV for.
  Proves the adapter recovers IV from the book instead of dropping the strike.
* **22900** — NSE published 46.55% IV computed from a stale last trade of
  1,190 while the book stood at 965.20 / 1,082.25. Proves a stale published IV
  does not reach the greeks.
* **22300** — a book 318 points wide on a 1,628 mid, with the mid *below* the
  European lower bound. Proves an unmarkable strike gets no IV rather than a
  fabricated one.
* **25600 / 26400** — far wings priced near the tick floor, where the smile is
  real and premiums are close to zero.

To refresh them, re-request the endpoints listed above and trim to the same
rows. Do not edit values in place: an edited payload is no longer evidence of
what the exchange sends.
