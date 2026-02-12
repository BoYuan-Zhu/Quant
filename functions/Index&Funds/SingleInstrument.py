import os
import time
import hashlib
import pandas as pd
import akshare as ak


class SingleInstrument:
    """
    One object = one instrument (Index / ETF / Open-end Fund).

    kind:
      - "auto" (default): try index -> etf -> fund
      - "index": use index_zh_a_hist
      - "etf":   use fund_etf_hist_em
      - "fund":  use fund_open_fund_info_em (NAV)

    Examples:
      - HS300 index: code="000300" or "sh000300"
      - ETF: code="510300"
      - Open-end fund: code="110022"
    """

    def __init__(
        self,
        code: str,
        kind: str = "auto",
        sleep: float = 0.5,
        timeout: int = 30,
        cache_dir: str = ".cache_ak_instruments",
    ):
        self.raw_code = str(code).strip()
        self.code = self._normalize_code(self.raw_code)  # keep flexible; may include 'sh'
        self.kind = kind.lower().strip()
        self.sleep = float(sleep)
        self.timeout = int(timeout)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.detected_kind = None  # "index"/"etf"/"fund"
        self.data = None

        if self.kind not in ("auto", "index", "etf", "fund"):
            raise ValueError("kind must be one of: auto/index/etf/fund")

    # ===================== unified entry =====================

    def hist(
        self,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
        use_cache: bool = True,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Unified 'history' interface:
          - index / etf -> OHLCV history
          - fund -> Not supported (use nav())

        start_date/end_date:
          - For index_zh_a_hist: typically 'YYYYMMDD'
          - For fund_etf_hist_em: 'YYYYMMDD' too
        """
        k = self._resolve_kind_for_price_history(start_date, end_date, adjust, use_cache, force)

        if k == "index":
            df = self._index_hist(start_date, end_date, use_cache, force)
        elif k == "etf":
            df = self._etf_hist(start_date, end_date, adjust, use_cache, force)
        else:
            raise ValueError("Open-end fund has no OHLCV 'hist'. Use nav().")

        self.data = df
        return df

    def nav(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        use_cache: bool = True,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Open-end fund NAV history.
        If kind=auto, it will resolve to fund if index/etf endpoints don't fit.
        """
        k = self._resolve_kind_for_nav(use_cache, force)

        if k != "fund":
            raise ValueError(f"nav() is only for open-end funds, but detected kind='{k}'. Use hist().")

        df = self._fund_nav(use_cache, force)
        df = self._normalize_date_index(df)

        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        self.data = df
        return df

    def get_kind(self) -> str:
        """
        Return detected kind if available; otherwise return configured kind.
        """
        return self.detected_kind or self.kind

    # ===================== internal: kind resolution =====================

    def _resolve_kind_for_price_history(self, start_date, end_date, adjust, use_cache, force) -> str:
        if self.kind in ("index", "etf"):
            self.detected_kind = self.kind
            return self.kind
        if self.kind == "fund":
            self.detected_kind = "fund"
            return "fund"

        # auto detect: index -> etf
        if self._can_call_index_hist(start_date, end_date, use_cache, force):
            self.detected_kind = "index"
            return "index"

        if self._can_call_etf_hist(start_date, end_date, adjust, use_cache, force):
            self.detected_kind = "etf"
            return "etf"

        # fallback
        self.detected_kind = "fund"
        return "fund"

    def _resolve_kind_for_nav(self, use_cache, force) -> str:
        if self.kind == "fund":
            self.detected_kind = "fund"
            return "fund"
        if self.kind in ("index", "etf"):
            self.detected_kind = self.kind
            return self.kind

        # auto detect: if it's clearly index/etf, treat as non-fund
        # (we keep this strict: nav() should not silently return something else)
        # So for auto, we try fund endpoint directly; if it fails -> it isn't a fund.
        if self._can_call_fund_nav(use_cache, force):
            self.detected_kind = "fund"
            return "fund"

        # if fund endpoint fails, assume it's not a fund
        # pick index first as a "more common" for codes like 000300
        self.detected_kind = "index"
        return "index"

    # ===================== internal: endpoint wrappers =====================

    def _index_hist(self, start_date, end_date, use_cache, force) -> pd.DataFrame:
        df = self._call_ak(
            "index_zh_a_hist",
            use_cache=use_cache,
            force=force,
            symbol=self._normalize_index_symbol(self.raw_code),
            start_date=start_date,
            end_date=end_date,
            timeout=self.timeout,
        )
        return self._normalize_ohlcv(df)

    def _etf_hist(self, start_date, end_date, adjust, use_cache, force) -> pd.DataFrame:
        df = self._call_ak(
            "fund_etf_hist_em",
            use_cache=use_cache,
            force=force,
            symbol=self._normalize_six_digit(self.raw_code),
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
            timeout=self.timeout,
        )
        return self._normalize_ohlcv(df)

    def _fund_nav(self, use_cache, force) -> pd.DataFrame:
        df = self._call_ak(
            "fund_open_fund_info_em",
            use_cache=use_cache,
            force=force,
            fund=self._normalize_six_digit(self.raw_code),
            timeout=self.timeout,
        )
        return df

    # ===================== internal: "can call" probes (safe, cached) =====================

    def _can_call_index_hist(self, start_date, end_date, use_cache, force) -> bool:
        try:
            df = self._index_hist(start_date, end_date, use_cache, force)
            ok = df is not None and not df.empty
            print(f"[detect] index OK={ok} code={self.raw_code}")
            return ok
        except Exception as e:
            print(f"[detect] index FAIL code={self.raw_code} err={type(e).__name__}: {e}")
            return False

    def _can_call_etf_hist(self, start_date, end_date, adjust, use_cache, force) -> bool:
        try:
            df = self._etf_hist(start_date, end_date, adjust, use_cache, force)
            ok = df is not None and not df.empty
            print(f"[detect] etf OK={ok} code={self.raw_code}")
            return ok
        except Exception as e:
            print(f"[detect] etf FAIL code={self.raw_code} err={type(e).__name__}: {e}")
            return False

    def _can_call_fund_nav(self, use_cache, force) -> bool:
        try:
            df = self._fund_nav(use_cache, force)
            ok = df is not None and not df.empty
            print(f"[detect] fund OK={ok} code={self.raw_code}")
            return ok
        except Exception as e:
            print(f"[detect] fund FAIL code={self.raw_code} err={type(e).__name__}: {e}")
            return False

    # ===================== normalization =====================

    @staticmethod
    def _normalize_code(code: str) -> str:
        return str(code).strip().lower()

    @staticmethod
    def _normalize_six_digit(code: str) -> str:
        c = str(code).strip().lower()
        c = c.replace("sh", "").replace("sz", "")
        if not c.isdigit():
            raise ValueError(f"Invalid 6-digit code: {code}")
        return c.zfill(6)

    @staticmethod
    def _normalize_index_symbol(symbol: str) -> str:
        """
        Many AkShare index endpoints accept:
          - '000300' or
          - 'sh000300' / 'sz399001'
        We'll keep 'sh/sz' if user provided, otherwise pad to 6 digits.
        """
        s = str(symbol).strip().lower()
        if s.startswith(("sh", "sz")) and len(s) == 8 and s[2:].isdigit():
            return s
        if s.isdigit():
            return s.zfill(6)
        return s

    @staticmethod
    def _normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        for col in ["净值日期", "日期", "date", "Date"]:
            if col in df.columns:
                out = df.copy()
                out[col] = pd.to_datetime(out[col], errors="coerce")
                out = out.sort_values(col).set_index(col)
                return out
        return df

    @staticmethod
    def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        column_map = {
            "date": "日期",
            "open": "开盘",
            "close": "收盘",
            "high": "最高",
            "low": "最低",
            "volume": "成交量",
            "amount": "成交额",
            "turnover": "换手率",
        }
        out = df.rename(columns=column_map).copy()

        if "日期" in out.columns:
            out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
            out = out.sort_values("日期").set_index("日期")

        for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    # ===================== AkShare caller + cache =====================

    def _call_ak(self, func_name: str, use_cache: bool = True, force: bool = False, **kwargs) -> pd.DataFrame:
        import inspect

        cache_path = None
        if use_cache:
            cache_path = self._cache_path(func_name, kwargs)
            if (not force) and os.path.exists(cache_path):
                try:
                    return pd.read_parquet(cache_path)
                except Exception:
                    try:
                        os.remove(cache_path)
                    except Exception:
                        pass

        func = getattr(ak, func_name, None)
        if func is None:
            raise AttributeError(f"AkShare has no function named: {func_name}")

        # -------- key fix: drop unsupported kwargs --------
        try:
            sig = inspect.signature(func)
            allowed = set(sig.parameters.keys())
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        except Exception:
            # If signature introspection fails, fall back to original kwargs
            filtered_kwargs = kwargs

        df = func(**filtered_kwargs)

        if use_cache and cache_path and df is not None and not df.empty:
            try:
                df.to_parquet(cache_path, index=False)
            except Exception:
                pass

        if self.sleep > 0:
            time.sleep(self.sleep)

        return df

    def _cache_path(self, func_name: str, params: dict) -> str:
        key = f"{func_name}|{self.raw_code}|{sorted(params.items(), key=lambda x: x[0])}"
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{func_name}_{h}.parquet")
