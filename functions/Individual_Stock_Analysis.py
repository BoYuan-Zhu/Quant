import akshare as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time



class SingleStock:
    """
    Unified A-share stock data fetcher.

    source:
        - "em"      : Eastmoney (default)
        - "sina"    : Sina
        - "tencent" : Tencent
    """

    def __init__(
        self,
        symbol: str,
        period: str = "daily",
        adjust: str = "qfq",
        source: str = "em",
        sleep: float = 1.0,
    ):
        self.symbol = symbol
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

    # ===================== public API =====================

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

        df = ak.stock_zh_a_daily(symbol=self.symbol,start_date=start_date, end_date=end_date, adjust = self.adjust)
        return df.reset_index()

    def _fetch_from_tencent(self, start_date: str, end_date: str) -> pd.DataFrame:
        if self.period != "daily":
            raise ValueError("Tencent only supports daily data")

        df = ak.stock_zh_a_hist_tx(symbol=self.symbol,start_date=start_date, end_date=end_date, adjust = self.adjust)
        return df.reset_index()

    # ===================== normalization =====================

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names and index for downstream usage.
        """
        column_map = {
            "date": "日期",
            "open": "开盘",
            "close": "收盘",
            "high": "最高",
            "low": "最低",
            "volume": "成交量",
            "amount": "成交额",
            "turnover":"换手率"
        }

        # 新浪 / 腾讯是英文字段
        df = df.rename(columns=column_map)

        if "日期" in df.columns:
            df["日期"] = pd.to_datetime(df["日期"])
            df = df.sort_values("日期").set_index("日期")

        for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额"]:
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
        return self.get_column("开盘")

    def close(self) -> pd.Series:
        return self.get_column("收盘")

    def high(self) -> pd.Series:
        return self.get_column("最高")

    def low(self) -> pd.Series:
        return self.get_column("最低")

    def volume(self) -> pd.Series:
        return self.get_column("成交量")

    def amount(self) -> pd.Series:
        return self.get_column("成交额")
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
        return self.get_column("换手率")

    # def financial_report(self, date: str):


    ##############################################   基本数据计算  #####################################################


    ## N日均线
    #ma5 = stock.ma(5)                 # 默认收盘价5日均线，前4天是NaN
    #ma20_open = stock.ma(20, "开盘")  # 用开盘价做20日均线
    def ma_convolve(self, N: int, price: str = "收盘") -> pd.Series:
        """
        R: N-day moving average using convolution, aligned to original index
        M: convolve(valid) then pad NaN at the front
        E: Return Series aligned with 日期 index (length == len(self.data))
        """
        self._require_data()

        if N <= 0:
            raise ValueError("N must be positive.")

        x = self.get_column(price).to_numpy(dtype=float)
        weights = np.ones(N, dtype=float) / N

        sma_valid = np.convolve(weights, x, mode="valid")  # length M-N+1
        pad = np.full(N - 1, np.nan, dtype=float)

        sma_full = np.concatenate([pad, sma_valid])        # length M

        return pd.Series(sma_full, index=self.data.index, name=f"MA{N}_{price}")



    ## 指数移动平均
    def ema(self, N: int, price: str = "收盘", adjust: bool = False) -> pd.Series:
        """
        R: N-period exponential moving average on column `price`
        M: Use pandas ewm to compute EMA with full length
        E: Return Series aligned with 日期 index
        """
        self._require_data()

        if N <= 0:
            raise ValueError("N must be positive.")

        s = self.get_column(price)

        # span = N 是金融里最常见的 EMA 定义
        return s.ewm(span=N, adjust=adjust).mean()

    ## MACD

    def macd(self, fast=12, slow=26, signal=9, price="收盘"):
        self._require_data()

        ema_fast = self.ema(fast, price=price)
        ema_slow = self.ema(slow, price=price)

        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        hist = dif - dea

        dif.name = f"DIF({fast},{slow})"
        dea.name = f"DEA({signal})"
        hist.name = "HIST"

        return dif, dea, hist

    ## KDJ
    def kdj(
        self,
        N: int = 9,
        k_smooth: int = 3,
        d_smooth: int = 3,
        init: float = 50.0,
        use_ema: bool = False,
    ):
        """
        R: N=RSV window; k_smooth/d_smooth are smoothing periods; init is initial K/D
        M: Compute RSV then smooth to get K, D; J = 3K - 2D
        E: Return (K, D, J) as pd.Series aligned with 日期 index
        """
        self._require_data()
        if N <= 0 or k_smooth <= 0 or d_smooth <= 0:
            raise ValueError("N, k_smooth, d_smooth must be positive.")

        high = self.get_column("最高")
        low = self.get_column("最低")
        close = self.get_column("收盘")

        llv = low.rolling(window=N, min_periods=N).min()
        hhv = high.rolling(window=N, min_periods=N).max()

        denom = (hhv - llv).replace(0, np.nan)
        rsv = (close - llv) / denom * 100.0
        rsv.name = "RSV"


        # --- Smooth RSV -> K, then K -> D ---
        # Two common smoothing styles:
        # 1) "SMA" style (Chinese market): K_t = (K_{t-1}*(m-1) + RSV_t) / m
        # 2) EMA style: K = EWM(RSV), D = EWM(K)
        if use_ema:
            K = rsv.ewm(span=k_smooth, adjust=False).mean()
            D = K.ewm(span=d_smooth, adjust=False).mean()
        else:
            def sma_cn(x: pd.Series, m: int, init_value: float) -> pd.Series:
                out = np.full(len(x), np.nan, dtype=float)
                prev = init_value
                for i, v in enumerate(x.to_numpy(dtype=float)):
                    if np.isnan(v):
                        out[i] = np.nan
                    else:
                        prev = (prev * (m - 1) + v) / m
                        out[i] = prev
                return pd.Series(out, index=x.index)

            K = sma_cn(rsv, k_smooth, init)
            D = sma_cn(K, d_smooth, init)

        J = 3.0 * K - 2.0 * D

        K.name = f"K(N={N})"
        D.name = f"D(N={N})"
        J.name = f"J(N={N})"

        return K, D, J
