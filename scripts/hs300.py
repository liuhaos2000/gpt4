from __future__ import annotations

import datetime as dt
import re
from typing import Iterable, List
from urllib.request import Request, urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

try:
    import akshare as ak  # optional fallback for CSI300 constituents only
except Exception:  # pragma: no cover
    ak = None

INDEX_CODE = "399300"  # 沪深300
ETF_CODES = ["510300", "159919", "510330", "510310", "159941"]  # 头部 ETF
START_DATE = "2024-01-01"
END_DATE = dt.date.today().strftime("%Y-%m-%d")
MAX_COMPONENTS = 5  # 只抓取前 5 个成分股，避免大规模抓取导致 CI 超时


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lowered = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).lower()
        if key in lowered:
            return lowered[key]
    return None


def _normalize_component_codes(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, pd.Series):
        series = raw
    elif isinstance(raw, pd.DataFrame):
        series = raw.iloc[:, 0]
    else:
        series = pd.Series(raw)

    series = series.astype(str)
    series = series.str.replace(".", "", regex=False)
    series = series.str.replace(r"[^0-9]", "", regex=True)
    series = series[series.str.len() >= 6]
    series = series.astype(str).str.zfill(6)
    return series.dropna().drop_duplicates().tolist()


def _symbol_candidates(code: str) -> List[str]:
    code = str(code).zfill(6)
    if code.startswith(("0", "3", "159", "160")):
        return [f"{code}.SZ", f"{code}.SS"]
    if code.startswith(("600", "601", "603", "605", "688", "510", "511", "513", "515")):
        return [f"{code}.SS", f"{code}.SZ"]
    return [f"{code}.SS", f"{code}.SZ"]


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


def _fetch_csi300_component_codes_from_web() -> List[str]:
    urls = [
        "https://www.csindex.com.cn/zh-CN/indices/index-detail/399300",
        "https://www.csindex.com.cn/en/indices/index-detail/399300",
    ]
    headers = {"User-Agent": "Mozilla/5.0"}

    for url in urls:
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=20) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            if not html:
                continue

            try:
                tables = pd.read_html(html)
            except Exception:
                tables = []

            for df in tables:
                if df.empty:
                    continue
                for col in ["code", "股票代码", "代码", "symbol", "证券代码", "代码（A股）"]:
                    if col in df.columns:
                        codes = _normalize_component_codes(df[col])
                        if len(codes) >= 10:
                            return codes[:MAX_COMPONENTS]

            matches = re.findall(r'data-code=["\']?(\d{6})["\']?', html)
            if matches:
                return _normalize_component_codes(matches)[:MAX_COMPONENTS]
        except Exception:
            continue

    raise RuntimeError("无法从 AkShare 或公开页面获取沪深300成分股列表。")


def _read_csi300_from_akshare() -> List[str]:
    if ak is None:
        return []

    candidate_names = [
        "index_zh_a_component",
        "index_zh_a_cons",
        "index_zh_a_cons_component",
        "index_zh_a_component_daily",
        "index_zh_a_daily",
    ]

    for name in candidate_names:
        fn = getattr(ak, name, None)
        if not callable(fn):
            continue

        for kwargs in (
            {"symbol": INDEX_CODE},
            {"symbol": "000300"},
            {"index_code": INDEX_CODE},
            {"code": INDEX_CODE},
        ):
            try:
                comp = fn(**kwargs)
            except TypeError:
                continue
            except Exception:
                continue

            if comp is None:
                continue

            if isinstance(comp, pd.DataFrame):
                for col in ["code", "股票代码", "代码", "symbol", "证券代码", "代码（A股）"]:
                    if col in comp.columns:
                        codes = _normalize_component_codes(comp[col])
                        if codes:
                            return codes[:MAX_COMPONENTS]
            else:
                codes = _normalize_component_codes(comp)
                if codes:
                    return codes[:MAX_COMPONENTS]

    return []


def get_csi300_component_codes() -> List[str]:
    codes = _read_csi300_from_akshare()
    if codes:
        return codes

    codes = _fetch_csi300_component_codes_from_web()
    return codes[:MAX_COMPONENTS]


def get_stock_daily_market_cap(code: str, start: str, end: str) -> pd.DataFrame:
    hist = _download_history(code, start, end)
    price_col = "Adj Close" if "Adj Close" in hist.columns else "Close"

    out = hist[["date", price_col]].rename(columns={price_col: "close"})
    out["date"] = pd.to_datetime(out["date"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    shares = _resolve_shares_outstanding(code)
    out["market_cap"] = out["close"] * shares
    out = out.dropna(subset=["market_cap"]).sort_values("date")
    return _month_end(out, "market_cap")


def get_csi300_total_market_cap(start: str, end: str) -> pd.DataFrame:
    codes = get_csi300_component_codes()
    if not codes:
        raise RuntimeError("未能获取沪深300成分股代码。")

    frames = [get_stock_daily_market_cap(code, start, end).rename(columns={"market_cap": code}) for code in codes]
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="date", how="outer")

    value_cols = [c for c in merged.columns if c != "date"]
    merged["index_market_cap"] = merged[value_cols].sum(axis=1)
    return merged[["date", "index_market_cap"]].dropna().sort_values("date")


def get_etf_daily_close(code: str, start: str, end: str) -> pd.DataFrame:
    hist = _download_history(code, start, end)
    price_col = "Adj Close" if "Adj Close" in hist.columns else "Close"

    out = hist[["date", price_col]].rename(columns={price_col: "close"})
    out["date"] = pd.to_datetime(out["date"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["close"]).sort_values("date")
    return _month_end(out, "close")


def get_etf_total_market_cap(code: str, start: str, end: str) -> pd.DataFrame:
    close_df = get_etf_daily_close(code, start, end)
    share_value = _resolve_shares_outstanding(code)

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


def plot_ratio(start: str = START_DATE, end: str = END_DATE) -> str:
    ratio_df = get_head_etf_cap_ratio(start=start, end=end)
    if ratio_df.empty:
        raise RuntimeError("未能生成沪深300头部 ETF 市值占比数据，当前数据源返回空结果。")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ratio_df["date"], ratio_df["ratio_pct"], color="royalblue", linewidth=2)
    ax.set_title("沪深300头部ETF市值占比（月度）")
    ax.set_xlabel("日期")
    ax.set_ylabel("占沪深300总市值比例 (%)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    output_path = "hs300_head_etf_ratio.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"已保存图表: {output_path}")
    return output_path


if __name__ == "__main__":
    print("正在按月获取沪深300及头部 ETF 市值数据（仅前 5 个成分股）...")
    ratio_df = get_head_etf_cap_ratio(start=START_DATE, end=END_DATE)
    if ratio_df.empty:
        raise RuntimeError("未能生成沪深300头部 ETF 市值占比数据，当前数据源返回空结果。")
    print(ratio_df.head())
    print("\n最近一月占比:")
    print(ratio_df.iloc[-1])
    plot_ratio(start=START_DATE, end=END_DATE)
