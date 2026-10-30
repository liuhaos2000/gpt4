from __future__ import annotations

import datetime as dt
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

ETF_CODES = ["510300", "159919", "510330", "510310", "159941"]
START_DATE = "2024-01-01"
END_DATE = dt.date.today().strftime("%Y-%m-%d")


def _symbol_candidates(code: str) -> List[str]:
    code = str(code).zfill(6)
    if code.startswith(("0", "3", "159", "160")):
        return [f"{code}.SZ", f"{code}.SS"]
    if code.startswith(("600", "601", "603", "605", "688", "510", "511", "513", "515")):
        return [f"{code}.SS", f"{code}.SZ"]
    return [f"{code}.SS", f"{code}.SZ"]


def _download_history(code: str, start: str, end: str) -> pd.DataFrame:
    end_dt = pd.Timestamp(end) + pd.Timedelta(days=1)
    for symbol in _symbol_candidates(code):
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end_dt.strftime("%Y-%m-%d"),
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty:
                return df.reset_index().rename(columns={"Date": "date"})
        except Exception:
            continue
    raise RuntimeError(f"无法抓取 {code} 的历史行情数据（yfinance）。")


def _resolve_shares_outstanding(code: str) -> float:
    for symbol in _symbol_candidates(code):
        try:
            ticker = yf.Ticker(symbol)
            info = getattr(ticker, "info", {}) or {}
            shares = info.get("sharesOutstanding") or info.get("floatShares")
            if shares is not None:
                return float(shares)
            fast_info = getattr(ticker, "fast_info", None)
            shares = getattr(fast_info, "shares", None)
            if shares is not None:
                return float(shares)
        except Exception:
            continue
    raise ValueError(f"无法获取 {code} 的流通股数，yfinance 可能未返回该字段。")


def _month_end(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    out["month"] = out["date"].dt.to_period("M").dt.to_timestamp()
    out = out.drop_duplicates(subset=["month"], keep="last")
    out = out.rename(columns={"month": "date"})
    return out[["date", value_col]].reset_index(drop=True)


def get_etf_monthly_market_cap(code: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    hist = _download_history(code, start, end)
    price_col = "Adj Close" if "Adj Close" in hist.columns else "Close"

    out = hist[["date", price_col]].rename(columns={price_col: "close"})
    out["date"] = pd.to_datetime(out["date"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    shares = _resolve_shares_outstanding(code)
    out["market_cap"] = out["close"] * shares
    out = out.dropna(subset=["market_cap"]).sort_values("date")
    return _month_end(out, "market_cap")


def get_etf_market_cap_table(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    frames = [
        get_etf_monthly_market_cap(code, start=start, end=end).rename(columns={"market_cap": code})
        for code in ETF_CODES
    ]
    if not frames:
        return pd.DataFrame(columns=["date"])

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="date", how="outer")
    return merged.sort_values("date").reset_index(drop=True)


def plot_etf_market_cap(start: str = START_DATE, end: str = END_DATE) -> str:
    df = get_etf_market_cap_table(start=start, end=end)
    if df.empty:
        raise RuntimeError("没有可用的 ETF 市值数据。")

    fig, ax = plt.subplots(figsize=(12, 6))
    for code in ETF_CODES:
        if code in df.columns:
            ax.plot(df["date"], df[code], label=code)

    ax.set_title("头部 ETF 月度市值")
    ax.set_xlabel("日期")
    ax.set_ylabel("市值（元）")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    output_path = "hs300_head_etf_market_cap.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"已保存图表: {output_path}")
    return output_path


if __name__ == "__main__":
    data = get_etf_market_cap_table(start=START_DATE, end=END_DATE)
    print(data.head())
    print("\n最近一月数据:")
    print(data.iloc[-1])
    plot_etf_market_cap(start=START_DATE, end=END_DATE)
