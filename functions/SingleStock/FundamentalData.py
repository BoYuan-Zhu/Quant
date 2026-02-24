import os
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import akshare as ak
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 根据你文件层级调整
Data_dir = PROJECT_ROOT / "data" / "fundamentals"


@dataclass(frozen=True)
class CacheItem:
    """Simple in-memory cache item container."""
    ts: float
    df: pd.DataFrame


class FundamentalData:
    """
    Unified A-share fundamental data fetcher (AkShare/Eastmoney) with disk caching.

    Key features:
    - Stable class API for common fundamental endpoints
    - Rate limit (sleep), retry
    - Disk cache: save fetched DataFrame to CSV; next run loads from disk first
    - Optional in-memory TTL cache (fast within a single process)
    """

    def __init__(
        self,
        symbol: str,
        source: str = "em",
        sleep: float = 1.0,
        timeout: int = 30,
        cache_ttl: float = 3600.0,
        max_retries: int = 2,
        data_dir: str = str(Data_dir),   # <--- NEW: disk cache root
        prefer_disk: bool = True,              # <--- NEW: disk first
        disk_cache: bool = True,               # <--- NEW: save to disk
    ):
        # Normalize symbol
        self.symbol, self.exchange = self._normalize_stock_symbol(symbol)

        self.source = source.lower()
        self.sleep = float(sleep)
        self.timeout = int(timeout)
        self.cache_ttl = float(cache_ttl)
        self.max_retries = int(max_retries)

        # Disk cache settings
        self.data_dir = Path(data_dir)
        self.prefer_disk = bool(prefer_disk)
        self.disk_cache = bool(disk_cache)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._cache: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], CacheItem] = {}

        self._fetcher_map = {"em": self._fetch_from_em}
        if self.source not in self._fetcher_map:
            raise ValueError(f"Unsupported source: {self.source}")

    # ========================= Public APIs =========================

    def info(self, use_cache: bool = True, force_refresh: bool = False) -> pd.DataFrame:
        return self._fetcher_map[self.source]("info", use_cache=use_cache, force_refresh=force_refresh)

    def balance_sheet(self, mode: str = "report", use_cache: bool = True, force_refresh: bool = False) -> pd.DataFrame:
        return self._fetcher_map[self.source]("balance_sheet", mode=mode, use_cache=use_cache, force_refresh=force_refresh)

    def profit_sheet(self, mode: str = "report", use_cache: bool = True, force_refresh: bool = False) -> pd.DataFrame:
        return self._fetcher_map[self.source]("profit_sheet", mode=mode, use_cache=use_cache, force_refresh=force_refresh)

    def cash_flow_sheet(self, mode: str = "report", use_cache: bool = True, force_refresh: bool = False) -> pd.DataFrame:
        return self._fetcher_map[self.source]("cash_flow_sheet", mode=mode, use_cache=use_cache, force_refresh=force_refresh)

    def financial_analysis_indicator(
        self,
        start_year: str = "2010",
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return self._fetcher_map[self.source](
            "financial_analysis_indicator",
            start_year=start_year,
            use_cache=use_cache,
            force_refresh=force_refresh,
        )

    def financial_abstract(self, use_cache: bool = True, force_refresh: bool = False) -> pd.DataFrame:
        return self._fetcher_map[self.source]("financial_abstract", use_cache=use_cache, force_refresh=force_refresh)

    def valuation_em(self, use_cache: bool = True, force_refresh: bool = False) -> pd.DataFrame:
        # Keep public API name; internally treat as endpoint "valuation_em"
        return self._fetcher_map[self.source]("valuation_em", use_cache=use_cache, force_refresh=force_refresh)

    def dividend_cninfo(self, use_cache: bool = True, force_refresh: bool = False) -> pd.DataFrame:
        return self._fetcher_map[self.source]("dividend_cninfo", use_cache=use_cache, force_refresh=force_refresh)

    def history_dividend(self, use_cache: bool = True, force_refresh: bool = False) -> pd.DataFrame:
        return self._fetcher_map[self.source]("history_dividend", use_cache=use_cache, force_refresh=force_refresh)

    # ========================= Source Implementations =========================

    def _fetch_from_em(self, endpoint: str, **kwargs) -> pd.DataFrame:
        """
        Eastmoney-centric endpoints (most A-share fundamentals live here in AkShare).

        kwargs may include:
        - use_cache: bool (memory+disk)
        - force_refresh: bool (ignore disk/memory and fetch from network)
        - mode/start_year
        """
        # Keep kwargs but avoid leaking internal flags into AkShare calls
        use_cache = kwargs.pop("use_cache", True)
        force_refresh = kwargs.pop("force_refresh", False)

        if endpoint == "info":
            return self._call_ak_cached(
                func_name="stock_individual_info_em",
                func_kwargs={"symbol": self.symbol, "timeout": self.timeout},
                cache_id=("info", {}),
                use_cache=use_cache,
                force_refresh=force_refresh,
            )

        if endpoint == "balance_sheet":
            mode = kwargs.pop("mode", "report")
            if mode == "report":
                func_name = "stock_balance_sheet_by_report_em"
            elif mode == "yearly":
                func_name = "stock_balance_sheet_by_yearly_em"
            else:
                raise ValueError(f"Unsupported balance_sheet mode: {mode}")

            return self._call_ak_cached(
                func_name=func_name,
                func_kwargs={"symbol": self._em_symbol_upper()},
                cache_id=("balance_sheet", {"mode": mode}),
                use_cache=use_cache,
                force_refresh=force_refresh,
            )

        if endpoint == "profit_sheet":
            mode = kwargs.pop("mode", "report")
            if mode == "report":
                func_name = "stock_profit_sheet_by_report_em"
            elif mode == "yearly":
                func_name = "stock_profit_sheet_by_yearly_em"
            elif mode == "quarterly":
                func_name = "stock_profit_sheet_by_quarterly_em"
            else:
                raise ValueError(f"Unsupported profit_sheet mode: {mode}")

            return self._call_ak_cached(
                func_name=func_name,
                func_kwargs={"symbol": self._em_symbol_upper()},
                cache_id=("profit_sheet", {"mode": mode}),
                use_cache=use_cache,
                force_refresh=force_refresh,
            )

        if endpoint == "cash_flow_sheet":
            mode = kwargs.pop("mode", "report")
            if mode == "report":
                func_name = "stock_cash_flow_sheet_by_report_em"
            elif mode == "yearly":
                func_name = "stock_cash_flow_sheet_by_yearly_em"
            elif mode == "quarterly":
                func_name = "stock_cash_flow_sheet_by_quarterly_em"
            else:
                raise ValueError(f"Unsupported cash_flow_sheet mode: {mode}")

            return self._call_ak_cached(
                func_name=func_name,
                func_kwargs={"symbol": self._em_symbol_upper()},
                cache_id=("cash_flow_sheet", {"mode": mode}),
                use_cache=use_cache,
                force_refresh=force_refresh,
            )

        if endpoint == "financial_analysis_indicator":
            start_year = str(kwargs.pop("start_year", "2010"))
            return self._call_ak_cached(
                func_name="stock_financial_analysis_indicator",
                func_kwargs={"symbol": self.symbol, "start_year": start_year},
                cache_id=("financial_analysis_indicator", {"start_year": start_year}),
                use_cache=use_cache,
                force_refresh=force_refresh,
            )

        if endpoint == "financial_abstract":
            return self._call_ak_cached(
                func_name="stock_financial_abstract",
                func_kwargs={"symbol": self.symbol},
                cache_id=("financial_abstract", {}),
                use_cache=use_cache,
                force_refresh=force_refresh,
            )

        if endpoint == "valuation_em":
            return self._call_ak_cached(
                func_name="stock_value_em",
                func_kwargs={"symbol": self.symbol},
                cache_id=("valuation_em", {}),
                use_cache=use_cache,
                force_refresh=force_refresh,
            )

        if endpoint == "dividend_cninfo":
            return self._call_ak_cached(
                func_name="stock_dividend_cninfo",
                func_kwargs={"symbol": self.symbol},
                cache_id=("dividend_cninfo", {}),
                use_cache=use_cache,
                force_refresh=force_refresh,
            )

        if endpoint == "history_dividend":
            # global table, may be huge
            return self._call_ak_cached(
                func_name="stock_history_dividend",
                func_kwargs={},
                cache_id=("history_dividend", {}),
                use_cache=use_cache,
                force_refresh=force_refresh,
            )

        raise ValueError(f"Unsupported endpoint: {endpoint}")

    # ========================= Disk Cache Utilities =========================

    def _cache_file_path(self, endpoint: str, params: Dict[str, Any]) -> Path:
        """
        Build a stable cache file path:
          data_dir / <symbol> / <endpoint>__<params_hash>.csv

        params_hash avoids very long filenames and ensures uniqueness.
        """
        sym_dir = self.data_dir / self.symbol
        sym_dir.mkdir(parents=True, exist_ok=True)

        # Create a stable hash from sorted params
        items = sorted((str(k), str(v)) for k, v in params.items())
        raw = "|".join([f"{k}={v}" for k, v in items])
        h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10] if raw else "noparams"

        fname = f"{endpoint}__{h}.csv"
        return sym_dir / fname

    def _disk_load(self, endpoint: str, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        path = self._cache_file_path(endpoint, params)
        if path.exists():
            try:
                return pd.read_csv(path, dtype=str)  # keep raw strings to avoid dtype surprises
            except Exception:
                # If csv corrupted, ignore and re-fetch
                return None
        return None

    def _disk_save(self, df: pd.DataFrame, endpoint: str, params: Dict[str, Any]) -> Path:
        path = self._cache_file_path(endpoint, params)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    # ========================= Core Utilities =========================

    def _call_ak_cached(
        self,
        func_name: str,
        func_kwargs: Dict[str, Any],
        cache_id: Tuple[str, Dict[str, Any]],
        use_cache: bool,
        force_refresh: bool,
    ) -> pd.DataFrame:
        """
        Cache order:
        - If force_refresh: skip disk & memory, fetch from network
        - Else if use_cache:
            1) disk (if prefer_disk)
            2) memory
          If not found:
            fetch from network -> save disk -> save memory
        """
        endpoint, params = cache_id

        # 1) Disk cache
        if (not force_refresh) and use_cache and self.prefer_disk:
            hit = self._disk_load(endpoint, params)
            if hit is not None:
                return hit

        # 2) Memory cache (optional; still useful within one run)
        mem_key = (func_name, tuple(sorted(func_kwargs.items(), key=lambda x: x[0])))
        if (not force_refresh) and use_cache:
            cached = self._cache_get(mem_key)
            if cached is not None:
                return cached

        # 3) Network fetch (AkShare)
        df = self._call_ak(func_name, use_cache=False, **func_kwargs)

        # 4) Save disk
        if self.disk_cache and use_cache and (not force_refresh):
            self._disk_save(df, endpoint, params)

        # 5) Save memory
        if use_cache and (not force_refresh):
            self._cache_set(mem_key, df)

        return df

    def _call_ak(self, func_name: str, use_cache: bool = True, **kwargs) -> pd.DataFrame:
        """
        R: func_name + kwargs
        M: Resolve AkShare function, call with retries, rate-limit
        E: Return DataFrame
        """
        func = getattr(ak, func_name, None)
        if func is None:
            raise AttributeError(
                f"AkShare has no attribute '{func_name}'. "
                f"Please check your akshare version or endpoint name."
            )

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                df = func(**kwargs)

                if df is None:
                    df = pd.DataFrame()

                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)

                df = self._light_clean(df)

                if self.sleep > 0:
                    time.sleep(self.sleep)

                return df

            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(max(0.2, self.sleep) * (attempt + 1))
                else:
                    break

        raise RuntimeError(f"AkShare call failed: {func_name}, symbol={self.symbol}, err={last_err}") from last_err

    def _cache_get(self, key: Tuple[str, Tuple[Tuple[str, Any], ...]]) -> Optional[pd.DataFrame]:
        item = self._cache.get(key)
        if item is None:
            return None
        if (time.time() - item.ts) > self.cache_ttl:
            self._cache.pop(key, None)
            return None
        return item.df.copy()

    def _cache_set(self, key: Tuple[str, Tuple[Tuple[str, Any], ...]], df: pd.DataFrame) -> None:
        self._cache[key] = CacheItem(ts=time.time(), df=df.copy())

    @staticmethod
    def _light_clean(df: pd.DataFrame) -> pd.DataFrame:
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()].copy()
        return df

    @staticmethod
    def _normalize_stock_symbol(symbol: str) -> Tuple[str, str]:
        raw = str(symbol).strip()

        exch: Optional[str] = None
        upper = raw.upper()
        if upper.startswith("SH"):
            exch = "SH"
            upper = upper[2:]
        elif upper.startswith("SZ"):
            exch = "SZ"
            upper = upper[2:]
        elif upper.startswith("BJ"):
            exch = "BJ"
            upper = upper[2:]

        if upper.endswith(".SH"):
            exch = exch or "SH"
            upper = upper[:-3]
        elif upper.endswith(".SZ"):
            exch = exch or "SZ"
            upper = upper[:-3]
        elif upper.endswith(".BJ"):
            exch = exch or "BJ"
            upper = upper[:-3]

        digits = "".join(ch for ch in upper if ch.isdigit())
        if len(digits) != 6:
            raise ValueError(f"Invalid stock symbol (need 6 digits): {symbol}")

        if exch is None:
            if digits.startswith(("6", "5", "9")):
                exch = "SH"
            elif digits.startswith(("0", "3", "2")):
                exch = "SZ"
            elif digits.startswith(("4", "8")):
                exch = "BJ"
            else:
                exch = "SH"

        return digits, exch

    def _em_symbol_upper(self) -> str:
        return f"{self.exchange.upper()}{self.symbol}"


def print_df(df: pd.DataFrame, title: str, n: int = 10):
    print("\n" + "=" * 80)
    print(title)
    print("shape =", df.shape)
    if df.empty:
        print("(empty)")
    else:
        print(df.head(n))


if __name__ == "__main__":
    # 1) 初始化：建议 sleep >= 0.5，避免 AkShare 接口频繁被限制
    fd = FundamentalData(
        symbol="601398",     # 支持: "601398", "sh601398", "601398.SH", "SZ000001" 等
        source="em",
        sleep=0.8,           # 每次请求后 sleep 秒
        timeout=30,
        cache_ttl=3600,      # 1小时内重复请求会走内存缓存
        max_retries=2,
    )

    # 2) 基础信息（市值/流通市值等，返回 DataFrame）
    df_info = fd.info(use_cache=True)
    print_df(df_info, "1) stock_individual_info_em (info)")

    # 3) 三大报表：资产负债表 / 利润表 / 现金流量表
    #    mode 可选：report / yearly / quarterly（利润表、现金流支持 quarterly）
    df_bs = fd.balance_sheet(mode="report", use_cache=True)
    print_df(df_bs, "2) balance_sheet by report (stock_balance_sheet_by_report_em)")

    df_ps = fd.profit_sheet(mode="report", use_cache=True)
    print_df(df_ps, "3) profit_sheet by report (stock_profit_sheet_by_report_em)")

    df_cf = fd.cash_flow_sheet(mode="report", use_cache=True)
    print_df(df_cf, "4) cash_flow_sheet by report (stock_cash_flow_sheet_by_report_em)")

    # 4) 财务分析指标（常用：ROE、毛利率、周转率、偿债指标等）
    df_fa = fd.financial_analysis_indicator(start_year="2015", use_cache=True)
    print_df(df_fa, "5) financial_analysis_indicator since 2015 (stock_financial_analysis_indicator)")

    # 5) 财务摘要（如果你的 AkShare 版本没有这个 endpoint，会抛 AttributeError / RuntimeError）
    try:
        df_abs = fd.financial_abstract(use_cache=True)
        print_df(df_abs, "6) financial_abstract (stock_financial_abstract)")
    except Exception as e:
        print("\n[financial_abstract] not available in your akshare version or endpoint changed:")
        print("  ", repr(e))

    # 6) 估值序列（PE/PB/PS/市值等，东财 stock_value_em）
    df_val = fd.valuation_em(use_cache=True)
    print_df(df_val, "7) valuation_em (stock_value_em)")

    # 7) 分红（CNINFO）
    df_div = fd.dividend_cninfo(use_cache=True)
    print_df(df_div, "8) dividend_cninfo (stock_dividend_cninfo)")

    # 8) 历史分红（注意：这个 endpoint 通常返回“全市场”，数据可能很大）
    #    如果你只是想看单个股票，建议先不调这个，或在返回后自己筛选
    try:
        df_hist_div = fd.history_dividend(use_cache=True)
        print_df(df_hist_div, "9) history_dividend (stock_history_dividend) - may be huge", n=3)
    except Exception as e:
        print("\n[history_dividend] failed or too large:")
        print("  ", repr(e))

    # 9) 缓存演示：再次调用 info，应该命中内存缓存（更快，且不会再 sleep）
    df_info2 = fd.info(use_cache=True)
    print("\nCache demo: second info() call, equal to first? ->", df_info2.equals(df_info))