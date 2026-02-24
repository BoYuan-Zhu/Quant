import akshare as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta

class SingleStock:
    def __init__(
        self,
        symbol: str,
        period: str = "daily",
        adjust: str = "qfq",
        source: str = "em",
        sleep: float = 1.0,
        auto_fetch: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
        lookback_days: int = 365,
    ):
        self.symbol = self.normalize_symbol(symbol)
        self.period = period
        self.adjust = adjust
        self.source = source.lower()
        self.sleep = sleep
        self.data = None

        self._fetcher_map = {
            "em": self._fetch_from_em,
            "sina": self._fetch_from_sina,
            "tencent": self._fetch_from_tencent,
        }
        if self.source not in self._fetcher_map:
            raise ValueError(f"Unsupported data source: {self.source}")

        # --- auto fetch ---
        if auto_fetch:
            if end_date is None:
                end_date = datetime.now().strftime("%Y%m%d")
            if start_date is None:
                sd = datetime.strptime(end_date, "%Y%m%d") - timedelta(days=lookback_days)
                start_date = sd.strftime("%Y%m%d")
            self.fetch(start_date, end_date)

    # ===================== public API =====================
    def _symbol_for_api(self) -> str:
        # FIX: call the staticmethod via class/self
        return self.symbol_for_source(self.symbol, self.source)

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """
        Normalize stock symbol to internal format: 6-digit numeric string.

        Accepts examples:
            "601398"       -> "601398"
            "sh601398"     -> "601398"
            "sz000001"     -> "000001"
            "SH600000"     -> "600000"
            "600000.SH"    -> "600000"
            "000001.SZ"    -> "000001"
        """
        if symbol is None:
            raise ValueError("symbol cannot be None")

        s = str(symbol).strip().lower()

        # handle suffix like ".sh" ".sz"
        if s.endswith((".sh", ".sz")):
            s = s.split(".")[0]

        # handle prefix like "sh" "sz"
        if s.startswith(("sh", "sz")):
            s = s[2:]

        # keep digits only (defensive)
        s = "".join(ch for ch in s if ch.isdigit())

        if len(s) != 6:
            raise ValueError(f"Invalid A-share symbol after normalization: {symbol} -> {s}")

        return s

    @staticmethod
    def symbol_for_source(symbol: str, source: str) -> str:
        """
        Convert internal symbol (6-digit) to data-source-specific format.
        """
        s = SingleStock.normalize_symbol(symbol)
        src = str(source).strip().lower()

        if src in {"em", "eastmoney"}:
            # Eastmoney expects pure numeric code
            return s

        if src in {"sina", "tencent"}:
            # Basic SH/SZ routing by leading digit
            if s.startswith(("6", "9", "5")):
                return f"sh{s}"
            if s.startswith(("0", "2", "3")):
                return f"sz{s}"

            # If you need BJ (Beijing exchange) support later, handle it here.
            raise ValueError(f"Cannot infer exchange prefix for symbol={s} (source={src})")

        raise ValueError(f"Unsupported data source: {source}")

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        R: Input symbol/period/start/end/adjust
        M: Fetch OHLCV data from selected source
        E: self.data updated and returned
        """
        df = self._fetcher_map[self.source](start_date, end_date)

        if df is None or df.empty:
            raise RuntimeError(f"No data returned from source={self.source}")

        df = self._normalize_dataframe(df)
        self.data = df

        if self.sleep > 0:
            time.sleep(self.sleep)

        return df

    # ===================== data sources =====================

    def _fetch_from_em(self, start_date: str, end_date: str) -> pd.DataFrame:
        return ak.stock_zh_a_hist(
            symbol=self.symbol,
            period=self.period,
            start_date=start_date,
            end_date=end_date,
            adjust=self.adjust,
            timeout=30,
        )

    def _fetch_from_sina(self, start_date: str, end_date: str) -> pd.DataFrame:
        if self.period != "daily":
            raise ValueError("Sina only supports daily data")

        symbol_api = self._symbol_for_api()  # <- 关键：sina 通常也需要 sh/sz 前缀
        df = ak.stock_zh_a_daily(
            symbol=symbol_api,
            start_date=start_date,
            end_date=end_date,
            adjust=self.adjust
        )
        return df.reset_index()

    def _fetch_from_tencent(self, start_date: str, end_date: str) -> pd.DataFrame:
        if self.period != "daily":
            raise ValueError("Tencent only supports daily data")

        symbol_api = self._symbol_for_api()  # <- 关键：tencent 需要 sh/sz 前缀
        df = ak.stock_zh_a_hist_tx(
            symbol=symbol_api,
            start_date=start_date,
            end_date=end_date,
            adjust=self.adjust
        )
        return df.reset_index()

    # ===================== normalization =====================

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names and index for downstream usage.
        All internal columns use English semantic names.
        """
        column_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交额": "volume",
            "成交量": "amount",
            "换手率": "turnover",
        }

        # rename columns if present
        df = df.rename(columns=column_map)

        # handle date index
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")

        # numeric columns
        for col in ["open", "close", "high", "low", "volume", "amount", "turnover"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _require_data(self):
        if self.data is None or self.data.empty:
            raise ValueError("No data loaded. Call fetch(start_date, end_date) first.")

    ########################################     基本数据调用      ####################################################

    def get_column(self, col: str) -> pd.Series:
        self._require_data()
        if col not in self.data.columns:
            raise KeyError(f"Column '{col}' not found in data.")
        return self.data[col]

    def open(self) -> pd.Series:
        return self.get_column("open")

    def close(self) -> pd.Series:
        return self.get_column("close")

    def high(self) -> pd.Series:
        return self.get_column("high")

    def low(self) -> pd.Series:
        return self.get_column("low")

    def volume(self) -> pd.Series:
        return self.get_column("volume")

    def amount(self) -> pd.Series:
        return self.get_column("amount")
    # def amplitude(self) -> pd.Series:
    #     """Price amplitude (%)"""
    #     return self.get_column("振幅")
    #
    # def pct_change(self) -> pd.Series:
    #     """Daily percentage change (%)"""
    #     return self.get_column("涨跌幅")
    #
    # def change(self) -> pd.Series:
    #     """Daily price change"""
    #     return self.get_column("涨跌额")

    def turnover(self) -> pd.Series:
        """Turnover rate (%)"""
        return self.get_column("turnover")

    # def financial_report(self, date: str):


    ##############################################   基本数据计算  #####################################################


    ## N日均线
    #ma5 = stock.ma(5)                 # 默认收盘价5日均线，前4天是NaN
    #ma20_open = stock.ma(20, "开盘")  # 用开盘价做20日均线
    def ma_convolve(self, N: int, price: str = "close", inplace: bool = True, force: bool = False) -> pd.Series:
        self._require_data()

        if N <= 0:
            raise ValueError("N must be positive.")

        col_name = f"MA{N}_{price}"

        # Reuse if exists and not forcing recompute
        if (not force) and (col_name in self.data.columns):
            s_exist = self.data[col_name]
            if s_exist.notna().any():  # avoid returning an all-NaN placeholder
                return s_exist.rename(col_name)

        x = self.get_column(price).to_numpy(dtype=float)
        weights = np.ones(N, dtype=float) / N

        sma_valid = np.convolve(weights, x, mode="valid")
        pad = np.full(N - 1, np.nan, dtype=float)
        sma_full = np.concatenate([pad, sma_valid])

        s = pd.Series(sma_full, index=self.data.index, name=col_name)

        if inplace:
            self.data[col_name] = s

        return s

    ## 指数移动平均
    def ema(self, N: int, price: str = "close", adjust: bool = False, inplace: bool = True,
            force: bool = False) -> pd.Series:
        self._require_data()

        if N <= 0:
            raise ValueError("N must be positive.")

        col_name = f"EMA{N}_{price}"

        if (not force) and (col_name in self.data.columns):
            s_exist = self.data[col_name]
            if s_exist.notna().any():
                return s_exist.rename(col_name)

        s = self.get_column(price).ewm(span=N, adjust=adjust).mean().rename(col_name)

        if inplace:
            self.data[col_name] = s

        return s

    ## MACD

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9, price: str = "close",
             inplace: bool = True, force: bool = False):
        self._require_data()

        dif_name = f"MACD_DIF_{fast}_{slow}_{price}"
        dea_name = f"MACD_DEA_{signal}_{fast}_{slow}_{price}"
        hist_name = f"MACD_HIST_{fast}_{slow}_{signal}_{price}"

        # If all exist and look valid, reuse
        if not force:
            if (dif_name in self.data.columns) and (dea_name in self.data.columns) and (hist_name in self.data.columns):
                dif_exist = self.data[dif_name]
                dea_exist = self.data[dea_name]
                hist_exist = self.data[hist_name]
                # if any has real values, assume computed
                if dif_exist.notna().any() or dea_exist.notna().any() or hist_exist.notna().any():
                    return dif_exist.rename(dif_name), dea_exist.rename(dea_name), hist_exist.rename(hist_name)

        # Compute
        ema_fast = self.ema(fast, price=price, inplace=False, force=force)
        ema_slow = self.ema(slow, price=price, inplace=False, force=force)

        dif = (ema_fast - ema_slow).rename(dif_name)
        dea = dif.ewm(span=signal, adjust=False).mean().rename(dea_name)
        hist = (dif - dea).rename(hist_name)

        if inplace:
            self.data[dif_name] = dif
            self.data[dea_name] = dea
            self.data[hist_name] = hist

        return dif, dea, hist

    ## KDJ
    def kdj(self, N: int = 9, k_smooth: int = 3, d_smooth: int = 3, init: float = 50.0,
            use_ema: bool = False, inplace: bool = True, force: bool = False):
        self._require_data()
        if N <= 0 or k_smooth <= 0 or d_smooth <= 0:
            raise ValueError("N, k_smooth, d_smooth must be positive.")

        base = f"N{N}_K{k_smooth}_D{d_smooth}" + ("_EMA" if use_ema else "_SMA")
        k_name = f"KDJ_K_{base}"
        d_name = f"KDJ_D_{base}"
        j_name = f"KDJ_J_{base}"

        if not force:
            if (k_name in self.data.columns) and (d_name in self.data.columns) and (j_name in self.data.columns):
                K_exist = self.data[k_name]
                D_exist = self.data[d_name]
                J_exist = self.data[j_name]
                if K_exist.notna().any() or D_exist.notna().any() or J_exist.notna().any():
                    return K_exist.rename(k_name), D_exist.rename(d_name), J_exist.rename(j_name)

        high = self.get_column("high")
        low = self.get_column("low")
        close = self.get_column("close")

        llv = low.rolling(window=N, min_periods=N).min()
        hhv = high.rolling(window=N, min_periods=N).max()

        denom = (hhv - llv).replace(0, np.nan)
        rsv = (close - llv) / denom * 100.0

        if use_ema:
            K = rsv.ewm(span=k_smooth, adjust=False).mean()
            D = K.ewm(span=d_smooth, adjust=False).mean()
        else:
            def sma_cn(x, m, init_value):
                out = np.full(len(x), np.nan, dtype=float)
                prev = init_value
                arr = x.to_numpy(dtype=float)
                for i, v in enumerate(arr):
                    if np.isnan(v):
                        out[i] = np.nan
                    else:
                        prev = (prev * (m - 1) + v) / m
                        out[i] = prev
                return pd.Series(out, index=x.index)

            K = sma_cn(rsv, k_smooth, init)
            D = sma_cn(K, d_smooth, init)

        J = 3.0 * K - 2.0 * D

        K = K.rename(k_name)
        D = D.rename(d_name)
        J = J.rename(j_name)

        if inplace:
            self.data[k_name] = K
            self.data[d_name] = D
            self.data[j_name] = J

        return K, D, J

    def TD(
            self,
            lookback: int = 4,
            setup_len: int = 9,
            price: str = "close",
            shift_days: int = 1,
            inplace: bool = True,
            force: bool = False,
    ):
        """
        TD Sequential simplified signals:
          - buy[setup_len]_signal : next trading day after Buy Setup reaches setup_len
          - sell[setup_len]_signal: next trading day after Sell Setup reaches setup_len

        Returns:
          buy_signal, sell_signal  (int Series 0/1)
        """
        self._require_data()

        # Column names like: buy[9]_signal, sell[9]_signal (setup_len changes them)
        buy_sig_col = f"buy[{setup_len}]_signal"
        sell_sig_col = f"sell[{setup_len}]_signal"

        if not force and (buy_sig_col in self.data.columns) and (sell_sig_col in self.data.columns):
            bs, ss = self.data[buy_sig_col], self.data[sell_sig_col]
            if bs.notna().any() or ss.notna().any():
                return bs, ss

        s = self.get_column(price)
        ref = s.shift(lookback)

        # bool arrays (avoid FutureWarning)
        buy_cond = (s < ref).astype("boolean").fillna(False).to_numpy(dtype=bool)
        sell_cond = (s > ref).astype("boolean").fillna(False).to_numpy(dtype=bool)

        n = len(s)
        buy_setup = np.zeros(n, dtype=np.int32)
        sell_setup = np.zeros(n, dtype=np.int32)

        bcnt = 0
        scnt = 0
        for i in range(n):
            if buy_cond[i]:
                bcnt += 1
                if bcnt > setup_len:
                    bcnt = setup_len
            else:
                bcnt = 0
            buy_setup[i] = bcnt

            if sell_cond[i]:
                scnt += 1
                if scnt > setup_len:
                    scnt = setup_len
            else:
                scnt = 0
            sell_setup[i] = scnt

        buy_hit = (pd.Series(buy_setup, index=self.data.index) == setup_len)
        sell_hit = (pd.Series(sell_setup, index=self.data.index) == setup_len)

        # next-day signals
        buy_signal = buy_hit.shift(shift_days).astype("boolean").fillna(False).astype("int8").rename(buy_sig_col)
        sell_signal = sell_hit.shift(shift_days).astype("boolean").fillna(False).astype("int8").rename(sell_sig_col)

        if inplace:
            self.data[buy_sig_col] = buy_signal
            self.data[sell_sig_col] = sell_signal

        return buy_signal, sell_signal
