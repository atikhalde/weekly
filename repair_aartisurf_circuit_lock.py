"""
One-time repair for the ONE ledger row confirmed, via real market data, to
be a genuine circuit-lock case that pre-dates the ab_paper.py fix:
AARTISURF, signal_date 2026-08-11, models E_btst / E_btst_wide.

Verified via real yfinance OHLC for 2026-08-11: Open=510.00 High=572.00
Low=500.90 Close=572.00 - Close == High, exactly +10.00% over the real
prior close of 520.00 - the textbook signature of a stock pinned at its
upper circuit with zero further trading (0 sellers).

This script does NOT touch any other flagged row. A broader scan using
just (entry, day_ret) flagged ~39 rows, but cross-checking every unique
(symbol, date) against real yfinance OHLC showed AARTISURF was the ONLY
one that genuinely closed at its day's high - all others (BLSE, MARINE,
KENNAMET, MATRIMONY, HAPPYFORGE, MOLDTECH, RAMCOIND, SAILIFE, SHILPAMED,
SHRIPISTON, SJS, SMLMAH, UNIPARTS, KABRAEXTRU) closed meaningfully below
their real day's high and were legitimately tradeable - repairing them
would have been WRONG (turning real, correct trades into false NO_FILLs).

Run once: `python repair_aartisurf_circuit_lock.py ab_ledger.csv`
"""
import sys
import pandas as pd

def main(path: str) -> int:
    df = pd.read_csv(path)
    mask = (
        (df["symbol"] == "AARTISURF")
        & (df["signal_date"] == "2026-08-11")
        & (df["model"].isin(["E_btst", "E_btst_wide"]))
        & (df["exit_reason"] != "NO_FILL")
    )
    n = int(mask.sum())
    if n == 0:
        print("No matching rows found - nothing to repair (already fixed, or file layout changed).")
        return 0

    print(f"Repairing {n} row(s):")
    print(df.loc[mask, ["model", "symbol", "signal_date", "entry", "exit", "exit_reason", "pnl", "pnl_pct"]].to_string())

    df.loc[mask, "exit_date"] = ""
    df.loc[mask, "exit_time"] = ""
    df.loc[mask, "exit"] = 0.0
    df.loc[mask, "exit_reason"] = "NO_FILL"
    df.loc[mask, "exit_note"] = "locked at upper circuit at signal time - 0 sellers, not fillable (repaired 2026-08-14; verified via real yfinance OHLC: close==high==572.00, prev close 520.00, exactly +10.00%)"
    df.loc[mask, "bars_held"] = 0
    df.loc[mask, "gross_pnl"] = 0.0
    df.loc[mask, "costs"] = 0.0
    df.loc[mask, "pnl"] = 0.0
    df.loc[mask, "pnl_pct"] = 0.0
    df.loc[mask, "r_multiple"] = 0.0
    df.loc[mask, "mfe_pct"] = 0.0
    df.loc[mask, "mae_pct"] = 0.0
    df.loc[mask, "qty"] = 0
    df.loc[mask, "invested"] = 0.0

    df.to_csv(path, index=False)
    print(f"\nRepaired. {path} updated.")
    return 0


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "ab_ledger.csv"
    sys.exit(main(path))
