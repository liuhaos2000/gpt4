from __future__ import annotations

import datetime as dt
from typing import Iterable, List

import akshare as ak
import matplotlib.pyplot as plt
import pandas as pd

INDEX_CODE = "399300"  # 沪深300
ETF_CODES = ["510300", "159919", "510330", "510310", "159941"]  # 头部 ETF
START_DATE = "2024-01-01"
END_DATE = dt.date.today().strftime("%Y-%m-%d")


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lowered = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).lower()
        if key in lowered:
            return lowered[key]
    return None

def get_csi300_component_codes() -> List[str]:
    comp = ak.index_zh_a_component(symbol=INDEX_CODE)
    code_col = _pick_col(comp, ["code", "股票代码", "代码", "symbol"])
    if code_col is None:
        raise ValueError("无法识别沪深300成分股代码列，请检查 akshare 版本。")
    codes = comp[code_col].astype(str).str.replace(".", "", regex=False).str.zfill(6)
    return codes.dropna().tolist()

def get_stock_daily_market_cap(code: str, start: str, end: str) -> pd.DataFrame:
    stock = ak.stock_zh_a_daily(
        symbol=code,
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
        adjust="hfq",
    )
    date_col = _pick_col(stock, ["date", "交易日期", "datetime"])
    market_col = _pick_col(stock, ["total_market_cap", "总市值", "market_cap", "总市值（元）", "市值"])

    if date_col is None or market_col is None:
        raise ValueError(f"{code} 无法读取总市值字段，可能是 akshare 版本差异。")

    out = stock[[date_col, market_col]].rename(columns={date_col: "date", market_col: "market_cap"})
    out["date"] = pd.to_datetime(out["date"])
    out["market_cap"] = pd.to_numeric(out["market_cap"], errors="coerce")
    out = out.dropna(subset=["market_cap"]).sort_values("date")
    return out

def get_csi300_total_market_cap(start: str, end: str) -> pd.DataFrame:
    codes = get_csi300_component_codes()
    frames = [get_stock_daily_market_cap(code, start, end).rename(columns={"market_cap": code}) for code in codes]
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="date", how="outer")

    value_cols = [c for c in merged.columns if c != "date"]
    merged["index_market_cap"] = merged[value_cols].sum(axis=1)
    return merged[["date", "index_market_cap"]].dropna().sort_values("date")

def get_etf_daily_close(code: str, start: str, end: str) -> pd.DataFrame:
    for fetch in [
        lambda c: ak.fund_etf_hist_em(symbol=c, period="daily", start_date=start.replace("-", ""), end_date=end.replace("-", "")),
        lambda c: ak.fund_etf_daily(symbol=c, start_date=start.replace("-", ""), end_date=end.replace("-", "")),
        lambda c: ak.fund_em_etf_hist(symbol=c, start_date=start.replace("-", ""), end_date=end.replace("-", "")),
    ]:
        try:
            df = fetch(code)
            if df is None or df.empty:
                continue
            date_col = _pick_col(df, ["date", "交易日期", "datetime", "日期"])
            close_col = _pick_col(df, ["close", "收盘", "收盘价", "close_price"])
            if date_col is None or close_col is None:
                continue
            out = df[[date_col, close_col]].rename(columns={date_col: "date", close_col: "close"})
            out["date"] = pd.to_datetime(out["date"])
            out["close"] = pd.to_numeric(out["close"], errors="coerce")
            return out.dropna(subset=["close"]).sort_values("date")
        except Exception:
            continue
    raise RuntimeError(f"无法抓取 ETF {code} 的行情数据，建议检查 akshare 接口或代码。")

def get_etf_total_market_cap(code: str, start: str, end: str) -> pd.DataFrame:
    close_df = get_etf_daily_close(code, start, end)

    info = ak.fund_etf_fund_info()
    code_col = _pick_col(info, ["fund_code", "基金代码", "code", "代码"])
    share_col = _pick_col(info, ["fund_share", "总份额", "基金份额", "份额", "share", "total_share", "shares"])

    if code_col is None or share_col is None:
        raise ValueError("无法识别 ETF 基金信息字段，可能是 akshare 版本不同。")

    subset = info[info[code_col].astype(str).str.contains(code, na=False)]
    if subset.empty:
        raise ValueError(f"未找到 ETF {code} 的基金信息。")

    total_shares = pd.to_numeric(subset.iloc[0][share_col], errors="coerce")
    if pd.isna(total_shares):
        raise ValueError(f"ETF {code} 的总份额为空。")

    share_value = float(total_shares)
    if "亿" in str(subset.iloc[0][share_col]):
        share_value *= 1e8

    out = close_df.rename(columns={"close": "etf_close"})
    out["market_cap"] = out["etf_close"] * share_value
    return out[["date", "market_cap"]].dropna().sort_values("date")

def get_head_etf_cap_ratio(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    index_df = get_csi300_total_market_cap(start, end)
    etf_dfs = [get_etf_total_market_cap(code, start, end).rename(columns={"market_cap": code}) for code in ETF_CODES]

    merged = etf_dfs[0]
    for df in etf_dfs[1:]:
        merged = merged.merge(df, on="date", how="outer")

    merged["etf_market_cap"] = merged[[c for c in merged.columns if c != "date"]].sum(axis=1)
    ratio_df = merged[["date", "etf_market_cap"]].merge(index_df, on="date", how="inner")
    ratio_df["ratio_pct"] = ratio_df["etf_market_cap"] / ratio_df["index_market_cap"] * 100
    return ratio_df[["date", "ratio_pct"]].sort_values("date")

def plot_ratio(start: str = START_DATE, end: str = END_DATE) -> None:
    ratio_df = get_head_etf_cap_ratio(start=start, end=end)

    plt.figure(figsize=(12, 6))
    plt.plot(ratio_df["date"], ratio_df["ratio_pct"], color="royalblue", linewidth=2)
    plt.title("沪深300头部ETF市值占比（2024年1月起）")
    plt.xlabel("日期")
    plt.ylabel("占沪深300总市值比例 (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("正在获取沪深300及头部 ETF 市值数据...")
    ratio_df = get_head_etf_cap_ratio(start=START_DATE, end=END_DATE)
    print(ratio_df.head())
    print("\n最近一日占比:")
    print(ratio_df.iloc[-1])
    plot_ratio(start=START_DATE, end=END_DATE)
