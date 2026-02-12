import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import akshare as ak
import pandas as pd


@dataclass(frozen=True)
class CacheItem:
    """Simple cache item container."""
    ts: float
    df: pd.DataFrame


class FundamentalData:
    """
    Unified A-share fundamental data fetcher based on AkShare.

    Design:
    - Provide a stable class API even if AkShare endpoints rename.
    - Built-in rate limit (sleep), retry, and in-memory TTL cache.
    - Return raw DataFrame from AkShare by default (minimal assumptions).
    """

    def __init__(
        self,
        symbol: str,
        source: str = "em",
        sleep: float = 1.0,
        timeout: int = 30,
        cache_ttl: float = 3600.0,
        max_retries: int = 2,
    ):
        # Store normalized 6-digit code and inferred/explicit exchange.
        self.symbol, self.exchange = self._normalize_stock_symbol(symbol)
        self.source = source.lower()
        self.sleep = float(sleep)
        self.timeout = int(timeout)
        self.cache_ttl = float(cache_ttl)
        self.max_retries = int(max_retries)

        # In-memory cache: key -> CacheItem
        self._cache: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], CacheItem] = {}

        # Source routing (easy to extend later if you add other providers)
        self._fetcher_map = {
            "em": self._fetch_from_em,
        }
        if self.source not in self._fetcher_map:
            raise ValueError(f"Unsupported source: {self.source}")

    # ========================= Public APIs =========================

    def info(self, use_cache: bool = True) -> pd.DataFrame:
        """
        R: self.symbol
        M: Fetch basic individual info (market cap, float cap, etc.)
        E: Return DataFrame
        """
        return self._fetcher_map[self.source]("info", use_cache=use_cache)

    def balance_sheet(self, mode: str = "report", use_cache: bool = True) -> pd.DataFrame:
        """
        mode:
          - "report": by report period
          - "yearly": by yearly
        """
        return self._fetcher_map[self.source]("balance_sheet", mode=mode, use_cache=use_cache)

    def profit_sheet(self, mode: str = "report", use_cache: bool = True) -> pd.DataFrame:
        """
        mode:
          - "report": by report period
          - "yearly": by yearly
          - "quarterly": by single quarter
        """
        return self._fetcher_map[self.source]("profit_sheet", mode=mode, use_cache=use_cache)

    def cash_flow_sheet(self, mode: str = "report", use_cache: bool = True) -> pd.DataFrame:
        """
        mode:
          - "report": by report period
          - "yearly": by yearly
          - "quarterly": by single quarter
        """
        return self._fetcher_map[self.source]("cash_flow_sheet", mode=mode, use_cache=use_cache)

    def financial_analysis_indicator(
            self,
            start_year: str = "2010",
            use_cache: bool = True
    ) -> pd.DataFrame:
        """
        R: start_year, use_cache
        M: Fetch financial analysis indicators from AkShare
        E: Return DataFrame
        """
        return self._fetcher_map[self.source](
            "financial_analysis_indicator",
            start_year=start_year,
            use_cache=use_cache
        )

    def financial_abstract(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Financial abstract (key summary fields).
        Note: Some AkShare versions may not expose this endpoint.
        """
        return self._fetcher_map[self.source]("financial_abstract", use_cache=use_cache)

    def valuation_em(self, use_cache: bool = True) -> pd.DataFrame:
        """
        R: use_cache
        M: Fetch valuation and market-based indicators from Eastmoney
        E: Return DataFrame

        Underlying AkShare function:
        - stock_value_em(symbol="601398")

        Notes:
        - This provides valuation-related time series such as PE, PB, PS,
          total market value, circulating market value, etc.
        - Symbol format: 6-digit stock code without exchange prefix.
        """
        return self._call_ak(
            "stock_value_em",
            symbol=self.symbol,
            use_cache=use_cache
        )

    def dividend_cninfo(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Dividend history from CNINFO.
        """
        return self._fetcher_map[self.source]("dividend_cninfo", use_cache=use_cache)

    def history_dividend(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Historical dividend (AkShare provides multiple endpoints; this one is common).
        If not available in your version, you will get a helpful error.
        """
        return self._fetcher_map[self.source]("history_dividend", use_cache=use_cache)

    # ========================= Source Implementations =========================

    def _fetch_from_em(self, endpoint: str, **kwargs) -> pd.DataFrame:
        """
        Eastmoney-centric endpoints (most A-share fundamentals live here in AkShare).
        """
        if endpoint == "info":
            # Doc: symbol is 6-digit code, e.g. "603777".
            return self._call_ak("stock_individual_info_em", symbol=self.symbol, timeout=self.timeout, **kwargs)

        if endpoint == "balance_sheet":
            mode = kwargs.pop("mode", "report")
            if mode == "report":
                # Doc: symbol is prefixed, e.g. "SH600519".
                return self._call_ak("stock_balance_sheet_by_report_em", symbol=self._em_symbol_upper(), **kwargs)
            if mode == "yearly":
                return self._call_ak("stock_balance_sheet_by_yearly_em", symbol=self._em_symbol_upper(), **kwargs)
            raise ValueError(f"Unsupported balance_sheet mode: {mode}")

        if endpoint == "profit_sheet":
            mode = kwargs.pop("mode", "report")
            if mode == "report":
                # Doc: symbol is prefixed, e.g. "SH600519".
                return self._call_ak("stock_profit_sheet_by_report_em", symbol=self._em_symbol_upper(), **kwargs)
            if mode == "yearly":
                return self._call_ak("stock_profit_sheet_by_yearly_em", symbol=self._em_symbol_upper(), **kwargs)
            if mode == "quarterly":
                return self._call_ak("stock_profit_sheet_by_quarterly_em", symbol=self._em_symbol_upper(), **kwargs)
            raise ValueError(f"Unsupported profit_sheet mode: {mode}")

        if endpoint == "cash_flow_sheet":
            mode = kwargs.pop("mode", "report")
            if mode == "report":
                # Doc: symbol is prefixed, e.g. "SH600519".
                return self._call_ak("stock_cash_flow_sheet_by_report_em", symbol=self._em_symbol_upper(), **kwargs)
            if mode == "yearly":
                return self._call_ak("stock_cash_flow_sheet_by_yearly_em", symbol=self._em_symbol_upper(), **kwargs)
            if mode == "quarterly":
                return self._call_ak("stock_cash_flow_sheet_by_quarterly_em", symbol=self._em_symbol_upper(), **kwargs)
            raise ValueError(f"Unsupported cash_flow_sheet mode: {mode}")

        if endpoint == "financial_analysis_indicator":
            use_cache = kwargs.pop("use_cache", True)

            # AkShare requires start_year in some versions
            start_year = kwargs.pop("start_year", "2010")

            return self._call_ak(
                "stock_financial_analysis_indicator",
                symbol=self.symbol,
                start_year=str(start_year),
                use_cache=use_cache,
                **kwargs
            )

        if endpoint == "financial_abstract":
            # Some versions expose it as stock_financial_abstract
            # Doc: Sina endpoint uses 6-digit code, e.g. "600004".
            return self._call_ak("stock_financial_abstract", symbol=self.symbol, **kwargs)

        if endpoint == "valuation_em":
            use_cache = kwargs.pop("use_cache", True)
            return self._call_ak(
                "stock_value_em",
                symbol=self.symbol,  # digits only
                use_cache=use_cache,
                **kwargs
            )

        if endpoint == "dividend_cninfo":
            # Doc: CNINFO endpoint uses 6-digit code, e.g. "600009".
            return self._call_ak("stock_dividend_cninfo", symbol=self.symbol, **kwargs)

        if endpoint == "history_dividend":
            # Doc: this endpoint returns ALL stocks and takes no parameters.
            return self._call_ak("stock_history_dividend", **kwargs)

        raise ValueError(f"Unsupported endpoint: {endpoint}")

    # ========================= Core Utilities =========================

    def _call_ak(self, func_name: str, use_cache: bool = True, **kwargs) -> pd.DataFrame:
        """
        R: func_name + kwargs
        M: Resolve AkShare function, call with retries, rate-limit, cache
        E: Return DataFrame
        """
        cache_key = (func_name, tuple(sorted(kwargs.items(), key=lambda x: x[0])))

        if use_cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

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
                    # Some endpoints might return dict/list; normalize to DataFrame.
                    df = pd.DataFrame(df)

                # Light normalization: remove duplicated columns and reset weird indexes
                df = self._light_clean(df)

                if use_cache:
                    self._cache_set(cache_key, df)

                if self.sleep > 0:
                    time.sleep(self.sleep)

                return df

            except Exception as e:
                last_err = e
                # Exponential-ish backoff (very mild)
                if attempt < self.max_retries:
                    time.sleep(max(0.2, self.sleep) * (attempt + 1))
                else:
                    break

        raise RuntimeError(f"AkShare call failed: {func_name}, symbol={self.symbol}, err={last_err}") from last_err

    def _cache_get(self, key: Tuple[str, Tuple[Tuple[str, Any], ...]]) -> Optional[pd.DataFrame]:
        """Return cached DataFrame if not expired."""
        item = self._cache.get(key)
        if item is None:
            return None
        if (time.time() - item.ts) > self.cache_ttl:
            self._cache.pop(key, None)
            return None
        return item.df.copy()

    def _cache_set(self, key: Tuple[str, Tuple[Tuple[str, Any], ...]], df: pd.DataFrame) -> None:
        """Store DataFrame into cache."""
        self._cache[key] = CacheItem(ts=time.time(), df=df.copy())

    @staticmethod
    def _light_clean(df: pd.DataFrame) -> pd.DataFrame:
        """
        Minimal cleaning:
        - Drop duplicated columns
        - Reset index if it looks like a meaningless RangeIndex is missing
        """
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()].copy()

        # Some endpoints return index as dates; keep as-is to avoid assumptions.
        return df

    @staticmethod
    def _normalize_stock_symbol(symbol: str) -> Tuple[str, str]:
        """Normalize stock symbol into (code_6digit, exchange).

        Notes
        - Many AkShare endpoints accept a plain 6-digit code ("600004").
        - Some Eastmoney financial-statement endpoints require an exchange prefix
          ("SH600519" / "SZ000001").

        Accepted inputs
        - "601398", "sh601398", "SH601398", "601398.SH"
        - "000001", "sz000001", "000001.SZ"
        - "BJ430047", "430047.BJ" (will keep BJ)
        """
        raw = str(symbol).strip()

        # Extract explicit exchange if provided.
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

        # Remove suffix like ".SH" / ".SZ" / ".BJ".
        if upper.endswith(".SH"):
            exch = exch or "SH"
            upper = upper[:-3]
        elif upper.endswith(".SZ"):
            exch = exch or "SZ"
            upper = upper[:-3]
        elif upper.endswith(".BJ"):
            exch = exch or "BJ"
            upper = upper[:-3]

        # Keep only digits.
        digits = "".join(ch for ch in upper if ch.isdigit())
        if len(digits) != 6:
            raise ValueError(f"Invalid stock symbol (need 6 digits): {symbol}")

        # Infer exchange if missing.
        if exch is None:
            # Heuristic mapping for CN markets.
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
        """Build Eastmoney-style prefixed symbol: 'SH600519' / 'SZ000001'."""
        return f"{self.exchange.upper()}{self.symbol}"
