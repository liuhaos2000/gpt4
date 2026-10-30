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


def _as_dataframe(obj):
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame().reset_index()
    if isinstance(obj, tuple):
        for item in obj:
            df = _as_dataframe(item)
            if not df.empty:
                return df
        return pd.DataFrame()
    if isinstance(obj, dict):
        return pd.DataFrame(obj)
    try:
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame()


def _symbol_candidates(code: str) -> List[str]:
    code = str(code).zfill(6)
    if code.startswith(("0", "3", "159", "160")):
        return [f"{code}.SZ", f"{code}.SS"]
    if code.startswith(("600", "601", "603", "605", "688", "510", "511", "513", "515")):
        return [f"{code}.SS", f"{code}.SZ"]
    return [f"{code}.SS", f"{code}.SZ"]


def _download_history(code: str, start: str, end: str) -> pd.DataFrame:
    end_dt = pd.Timestamp(end) + pd.Timedelta(days=1)
    last_error = None
    for symbol in _symbol_candidates(code):
        try:
            raw = yf.download(
                symbol,
                start=start,
                end=end_dt.strftime("%Y-%m-%d"),
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
            df = _as_dataframe(raw)
            if df.empty:
                continue
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})
            elif "date" not in df.columns and df.index.name is not None:
                df = df.reset_index().rename(columns={df.index.name: "date"})
            elif "date" not in df.columns and df.index.nlevels == 1:
                df = df.reset_index().rename(columns={"index": "date"})
            if "date" in df.columns:
                return df
        except Exception as exc:  # pragma: no cover - defensive for CI robustness
            last_error = exc
            continue
    if last_error is not None:
        raise RuntimeError(f"无法抓取 {code} 的历史行情数据（yfinance）：{last_error}")
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
    return 1.0


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
    if hist is None or hist.empty:
        raise ValueError(f"{code} 没有返回有效行情数据。")

    hist = hist.copy()
    if "date" not in hist.columns:
        hist = hist.reset_index()
    if "Date" in hist.columns:
        hist = hist.rename(columns={"Date": "date"})
    if "date" not in hist.columns:
        raise ValueError(f"{code} 的行情数据没有 date 列。")

    price_candidates = [c for c in hist.columns if str(c).lower() in {"adj close", "close"}]
    if not price_candidates:
        numeric_cols = [
            c for c in hist.columns
            if c != "date" and pd.api.types.is_numeric_dtype(hist[c])
        ]
        if not numeric_cols:
            raise ValueError(f"{code} 的行情数据没有可用的价格列。")
        price_col = numeric_cols[0]
    else:
        price_col = price_candidates[0]

    out = hist[["date", price_col]].copy().rename(columns={price_col: "close"})
    out["date"] = pd.to_datetime(out["date"])
    close_values = out["close"]
    if not isinstance(close_values, pd.Series):
        close_values = pd.Series(close_values)
    out["close"] = pd.to_numeric(close_values, errors="coerce")

    shares = _resolve_shares_outstanding(code)
    out["market_cap"] = out["close"] * shares
    out = out.dropna(subset=["market_cap"]).sort_values("date")
    return _month_end(out, "market_cap")


def get_etf_market_cap_table(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    frames = []
    for code in ETF_CODES:
        try:
            df = get_etf_monthly_market_cap(code, start=start, end=end).rename(columns={"market_cap": code})
            frames.append(df)
        except Exception as exc:
            print(f"跳过 ETF {code}: {exc}")
            continue

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
