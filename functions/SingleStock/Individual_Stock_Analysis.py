import akshare as ak
import numpy as np
import pandas as pd
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

    def td(
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

    def impulse_macd(
            self,
            fast: int = 12,
            slow: int = 26,
            signal: int = 9,
            trend_ema: int = 13,
            price: str = "close",
            inplace: bool = True,
            force: bool = False,
    ):
        """
        Impulse MACD (Elder Impulse System) based on:
          - Trend: slope of EMA(trend_ema)
          - Momentum: slope of MACD histogram (HIST)

        State encoding:
          +1 : Green  (trend up AND momentum up)
          -1 : Red    (trend down AND momentum down)
           0 : Neutral otherwise

        Returns:
          impulse_state, trend_ema_series, hist_series
        """
        self._require_data()

        # ---- names (aligned with your macd naming) ----
        ema_name = f"IMPULSE_EMA{trend_ema}_{price}"
        state_name = f"IMPULSE_STATE_EMA{trend_ema}_MACD_{fast}_{slow}_{signal}_{price}"

        # Reuse if exists
        if not force and (state_name in self.data.columns) and (ema_name in self.data.columns):
            st = self.data[state_name]
            ema_s = self.data[ema_name]
            # hist name is created by your macd(); we recompute/reuse below for safety
            _, _, hist = self.macd(fast=fast, slow=slow, signal=signal, price=price, inplace=inplace, force=force)
            if st.notna().any() or ema_s.notna().any():
                return st.rename(state_name), ema_s.rename(ema_name), hist

        # ---- compute components ----
        ema_tr = self.ema(trend_ema, price=price, inplace=False, force=force).rename(ema_name)
        _, _, hist = self.macd(fast=fast, slow=slow, signal=signal, price=price, inplace=inplace, force=force)

        # ---- slopes ----
        trend_slope = ema_tr.diff()
        mom_slope = hist.diff()

        trend_up = trend_slope > 0
        trend_dn = trend_slope < 0
        mom_up = mom_slope > 0
        mom_dn = mom_slope < 0

        # ---- impulse state ----
        state = pd.Series(0, index=self.data.index, dtype="int8")
        state[trend_up & mom_up] = 1
        state[trend_dn & mom_dn] = -1
        state = state.rename(state_name)

        if inplace:
            self.data[ema_name] = ema_tr
            self.data[state_name] = state

        return state, ema_tr, hist

    def rsi(
            self,
            N: int = 14,
            price: str = "close",
            oversold: float = 30.0,
            overbought: float = 70.0,
            inplace: bool = True,
            force: bool = False,
    ):
        """
        R: Compute RSI value + RSI overbought/oversold signal; add both to self.data.
        M: Wilder RSI. Signal:
           RSI <= oversold  ->  1
           RSI >= overbought-> -1
           else             ->  0
        E: Return (rsi_series, signal_series)
        """
        self._require_data()
        if N <= 0:
            raise ValueError("N must be positive.")
        if not (0 < oversold < overbought < 100):
            raise ValueError("Require 0 < oversold < overbought < 100.")

        rsi_col = f"RSI_{N}_{price}"
        sig_col = f"RSI_SIGNAL_{N}_{int(oversold)}_{int(overbought)}_{price}"

        # Reuse if exists and looks valid
        if not force and (rsi_col in self.data.columns) and (sig_col in self.data.columns):
            rsi_exist = self.data[rsi_col]
            sig_exist = self.data[sig_col]
            if rsi_exist.notna().any() or sig_exist.notna().any():
                return rsi_exist.rename(rsi_col), sig_exist.rename(sig_col)

        s = self.get_column(price).astype(float)
        delta = s.diff()

        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        rsi_vals = np.full(len(s), np.nan, dtype=float)
        g = gain.to_numpy(dtype=float)
        l = loss.to_numpy(dtype=float)

        if len(s) >= N + 1:
            # Seed with first N simple averages (from 1..N)
            avg_gain = np.nanmean(g[1: N + 1])
            avg_loss = np.nanmean(l[1: N + 1])

            # First RSI value at index N
            if avg_loss == 0 and avg_gain == 0:
                rsi_vals[N] = 50.0
            elif avg_loss == 0:
                rsi_vals[N] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_vals[N] = 100.0 - 100.0 / (1.0 + rs)

            # Wilder smoothing recursion
            for i in range(N + 1, len(s)):
                if np.isnan(g[i]) or np.isnan(l[i]):
                    rsi_vals[i] = np.nan
                    continue

                avg_gain = (avg_gain * (N - 1) + g[i]) / N
                avg_loss = (avg_loss * (N - 1) + l[i]) / N

                if avg_loss == 0 and avg_gain == 0:
                    rsi_vals[i] = 50.0
                elif avg_loss == 0:
                    rsi_vals[i] = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi_vals[i] = 100.0 - 100.0 / (1.0 + rs)

        rsi = pd.Series(rsi_vals, index=self.data.index, name=rsi_col)

        signal = pd.Series(0, index=self.data.index, dtype="int8", name=sig_col)
        signal[rsi <= oversold] = 1
        signal[rsi >= overbought] = -1
        signal[rsi.isna()] = 0

        if inplace:
            self.data[rsi_col] = rsi
            self.data[sig_col] = signal

        return rsi, signal

    def bollinger(
            self,
            N: int = 20,
            k: float = 2.0,
            price: str = "close",
            ddof: int = 0,
            inplace: bool = True,
            force: bool = False,
            with_extras: bool = True,
    ):
        """
        R: Compute Bollinger Bands based on rolling mean and std of `price`.
        M: mid = MA(N), std = rolling_std(N), upper = mid + k*std, lower = mid - k*std
        E: Return (mid, upper, lower) and optionally (bandwidth, percent_b)

        Notes:
          - ddof=0 uses population std (pandas default); ddof=1 uses sample std.
          - bandwidth = (upper - lower) / mid
          - percent_b = (price - lower) / (upper - lower)
        """
        self._require_data()

        if N <= 0:
            raise ValueError("N must be positive.")
        if k <= 0:
            raise ValueError("k must be positive.")
        if ddof not in (0, 1):
            raise ValueError("ddof must be 0 or 1.")

        # Column names
        mid_col = f"BOLL_MID_{N}_{k}_{price}"
        up_col = f"BOLL_UP_{N}_{k}_{price}"
        low_col = f"BOLL_LOW_{N}_{k}_{price}"
        bw_col = f"BOLL_BW_{N}_{k}_{price}"  # bandwidth
        pb_col = f"BOLL_PB_{N}_{k}_{price}"  # percent_b

        # Reuse if exists
        if not force:
            needed = [mid_col, up_col, low_col] + ([bw_col, pb_col] if with_extras else [])
            if all(c in self.data.columns for c in needed):
                # If any has real values, assume computed
                if any(self.data[c].notna().any() for c in needed):
                    mid = self.data[mid_col].rename(mid_col)
                    up = self.data[up_col].rename(up_col)
                    low = self.data[low_col].rename(low_col)
                    if with_extras:
                        bw = self.data[bw_col].rename(bw_col)
                        pb = self.data[pb_col].rename(pb_col)
                        return mid, up, low, bw, pb
                    return mid, up, low

        s = self.get_column(price).astype(float)

        mid = s.rolling(window=N, min_periods=N).mean().rename(mid_col)
        std = s.rolling(window=N, min_periods=N).std(ddof=ddof)
        up = (mid + k * std).rename(up_col)
        low = (mid - k * std).rename(low_col)

        if with_extras:
            denom = (up - low).replace(0, np.nan)
            bw = ((up - low) / mid.replace(0, np.nan)).rename(bw_col)
            pb = ((s - low) / denom).rename(pb_col)

            if inplace:
                self.data[mid_col] = mid
                self.data[up_col] = up
                self.data[low_col] = low
                self.data[bw_col] = bw
                self.data[pb_col] = pb

            return mid, up, low, bw, pb

        if inplace:
            self.data[mid_col] = mid
            self.data[up_col] = up
            self.data[low_col] = low

        return mid, up, low