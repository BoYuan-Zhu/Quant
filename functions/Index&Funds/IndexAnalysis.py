import time
from typing import Optional, Tuple, Dict, Callable

import akshare as ak
import numpy as np
import pandas as pd


class IndexAnalysis:
    """
    China index data fetcher + analytics (English output schema).

    Daily bars sources:
        - "sina": ak.stock_zh_index_daily (default)
        - "tx"  : ak.stock_zh_index_daily_tx
        - "em"  : ak.stock_zh_index_daily_em

    Normalized output:
        index: DatetimeIndex named "date"
        columns (when available): open, close, high, low, volume, amount
    """

    def __init__(self, symbol: str, source: str = "sina", sleep: float = 0.5):
        self.symbol = str(symbol).strip()
        self.source = source.lower().strip()
        self.sleep = float(sleep)
        self.data: Optional[pd.DataFrame] = None

        self._fetcher_map: Dict[str, Callable[[], pd.DataFrame]] = {
            "sina": self._fetch_from_sina,
            "tx": self._fetch_from_tencent,
            "em": self._fetch_from_eastmoney,
        }
        if self.source not in self._fetcher_map:
            raise ValueError(f"Unsupported source={self.source}, choose from {list(self._fetcher_map.keys())}")

    # ----------------------------- basic utils -----------------------------

    @staticmethod
    def digits_code(symbol: str) -> str:
        """Return digits-only code, e.g. 'sh000300' -> '000300'."""
        return "".join([c for c in str(symbol) if c.isdigit()])

    @staticmethod
    def _to_ts(s: Optional[str]) -> Optional[pd.Timestamp]:
        if s is None or str(s).strip() == "":
            return None
        return pd.to_datetime(s)

    def _require_data(self) -> None:
        if self.data is None or self.data.empty:
            raise ValueError("No data loaded. Call fetch() first.")

    def get_series(self, col: str) -> pd.Series:
        """Get a column as Series from normalized data."""
        self._require_data()
        if col not in self.data.columns:
            raise KeyError(f"Column '{col}' not found. Available: {list(self.data.columns)}")
        return self.data[col]

    # ----------------------------- fetch daily bars -----------------------------

    def fetch(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        R: start_date/end_date optional, 'YYYY-MM-DD'
        M: fetch raw -> normalize -> filter by date -> store in self.data
        E: returns normalized DataFrame
        """
        raw = self._fetcher_map[self.source]()
        if raw is None or raw.empty:
            raise RuntimeError(f"No data returned for symbol={self.symbol}, source={self.source}")

        df = self._normalize_daily(raw)

        sd = self._to_ts(start_date)
        ed = self._to_ts(end_date)
        if sd is not None:
            df = df[df.index >= sd]
        if ed is not None:
            df = df[df.index <= ed]

        self.data = df

        if self.sleep > 0:
            time.sleep(self.sleep)

        return df

    def _fetch_from_sina(self) -> pd.DataFrame:
        # ak.stock_zh_index_daily(symbol="sh000300")
        return ak.stock_zh_index_daily(symbol=self.symbol)

    def _fetch_from_tencent(self) -> pd.DataFrame:
        # ak.stock_zh_index_daily_tx(symbol="sh000300")
        return ak.stock_zh_index_daily_tx(symbol=self.symbol)

    def _fetch_from_eastmoney(self) -> pd.DataFrame:
        # ak.stock_zh_index_daily_em(symbol="sh000300")
        return ak.stock_zh_index_daily_em(symbol=self.symbol)

    def _normalize_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize various column naming conventions from different sources into:
            date index + open/close/high/low/volume/amount columns
        """
        df = df.copy()

        # Common column names across akshare index daily functions
        # Some sources use English, some use Chinese-like naming.
        rename_map = {
            "date": "date",
            "日期": "date",
            "open": "open",
            "开盘": "open",
            "今开": "open",
            "close": "close",
            "收盘": "close",
            "high": "high",
            "最高": "high",
            "low": "low",
            "最低": "low",
            "volume": "volume",
            "成交量": "volume",
            "amount": "amount",
            "成交额": "amount",
            "成交额(元)": "amount",
        }
        df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)

        # Set date index
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")
        else:
            # Already indexed
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df.index.name = "date"

        # Keep only relevant columns if they exist
        keep = [c for c in ["open", "close", "high", "low", "volume", "amount"] if c in df.columns]
        df = df[keep]

        # Cast to numeric
        for c in keep:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    # ----------------------------- convenience getters -----------------------------

    def open(self) -> pd.Series:
        return self.get_series("open")

    def close(self) -> pd.Series:
        return self.get_series("close")

    def high(self) -> pd.Series:
        return self.get_series("high")

    def low(self) -> pd.Series:
        return self.get_series("low")

    def volume(self) -> pd.Series:
        return self.get_series("volume")

    def amount(self) -> pd.Series:
        return self.get_series("amount")

    # ----------------------------- analytics (index细分内容) -----------------------------

    def returns(self, price: str = "close") -> pd.Series:
        """Daily simple returns."""
        s = self.get_series(price)
        out = s.pct_change()
        out.name = f"ret_{price}"
        return out

    def log_returns(self, price: str = "close") -> pd.Series:
        """Daily log returns."""
        s = self.get_series(price).astype(float)
        out = np.log(s / s.shift(1))
        out.name = f"logret_{price}"
        return out

    def cum_returns(self, price: str = "close") -> pd.Series:
        """Cumulative returns starting from 0."""
        r = self.returns(price).fillna(0.0)
        out = (1.0 + r).cumprod() - 1.0
        out.name = f"cumret_{price}"
        return out

    def rolling_vol(self, window: int = 20, price: str = "close", annualize: bool = True, trading_days: int = 252) -> pd.Series:
        """Rolling volatility of returns."""
        r = self.returns(price)
        out = r.rolling(window).std()
        if annualize:
            out = out * np.sqrt(trading_days)
        out.name = f"vol_{window}_{price}"
        return out

    def max_drawdown(self, price: str = "close") -> Tuple[float, pd.Series]:
        """Return (mdd_value, drawdown_series)."""
        s = self.get_series(price).astype(float)
        peak = s.cummax()
        dd = s / peak - 1.0
        dd.name = f"drawdown_{price}"
        mdd = float(dd.min()) if len(dd) else np.nan
        return mdd, dd

    # ----------------------------- indicators -----------------------------

    def sma(self, n: int, price: str = "close") -> pd.Series:
        """Simple moving average."""
        if n <= 0:
            raise ValueError("n must be positive.")
        s = self.get_series(price)
        out = s.rolling(n).mean()
        out.name = f"sma_{n}_{price}"
        return out

    def ema(self, n: int, price: str = "close", adjust: bool = False) -> pd.Series:
        """Exponential moving average."""
        if n <= 0:
            raise ValueError("n must be positive.")
        s = self.get_series(price)
        out = s.ewm(span=n, adjust=adjust).mean()
        out.name = f"ema_{n}_{price}"
        return out

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9, price: str = "close") -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD:
            dif = EMA(fast) - EMA(slow)
            dea = EMA(dif, signal)
            hist = dif - dea
        """
        if not (fast > 0 and slow > 0 and signal > 0):
            raise ValueError("fast/slow/signal must be positive.")
        if fast >= slow:
            raise ValueError("fast should be < slow for typical MACD.")

        ef = self.ema(fast, price=price, adjust=False)
        es = self.ema(slow, price=price, adjust=False)

        dif = ef - es
        dea = dif.ewm(span=signal, adjust=False).mean()
        hist = dif - dea

        dif.name = f"macd_dif_{fast}_{slow}_{price}"
        dea.name = f"macd_dea_{signal}_{fast}_{slow}_{price}"
        hist.name = f"macd_hist_{fast}_{slow}_{signal}_{price}"
        return dif, dea, hist

    def kdj(self, n: int = 9, k_smooth: int = 3, d_smooth: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        KDJ (classic):
            RSV = (close - LLV(low,n)) / (HHV(high,n) - LLV(low,n)) * 100
            K = EMA/SMOOTH(RSV, k_smooth)
            D = EMA/SMOOTH(K, d_smooth)
            J = 3K - 2D
        """
        if n <= 0 or k_smooth <= 0 or d_smooth <= 0:
            raise ValueError("n/k_smooth/d_smooth must be positive.")

        high = self.get_series("high")
        low = self.get_series("low")
        close = self.get_series("close")

        llv = low.rolling(n, min_periods=n).min()
        hhv = high.rolling(n, min_periods=n).max()
        denom = (hhv - llv).replace(0, np.nan)
        rsv = (close - llv) / denom * 100.0

        # Use EMA smoothing to keep stable and simple
        k = rsv.ewm(span=k_smooth, adjust=False).mean()
        d = k.ewm(span=d_smooth, adjust=False).mean()
        j = 3.0 * k - 2.0 * d

        k.name = f"kdj_k_{n}"
        d.name = f"kdj_d_{n}"
        j.name = f"kdj_j_{n}"
        return k, d, j

    # ----------------------------- index-specific data -----------------------------
    # These are independent from daily bars and do NOT require fetch().

    @staticmethod
    def index_info() -> pd.DataFrame:
        """
        Index list/info table.
        Underlying AkShare: ak.index_stock_info()
        """
        return ak.index_stock_info()

    def constituents_csindex(self) -> pd.DataFrame:
        """
        Constituents from CSI index website.
        Underlying AkShare: ak.index_stock_cons_csindex(symbol="000300")
        """
        code = self.digits_code(self.symbol)
        return ak.index_stock_cons_csindex(symbol=code)

    def constituents_sina(self) -> pd.DataFrame:
        """
        Constituents from Sina (mainstream indices).
        Underlying AkShare: ak.index_stock_cons_sina(symbol="000300")
        """
        code = self.digits_code(self.symbol)
        return ak.index_stock_cons_sina(symbol=code)

    def constituents_weight_csindex(self) -> pd.DataFrame:
        """
        Constituents weights from CSI index website.
        Underlying AkShare: ak.index_stock_cons_weight_csindex(symbol="000300")
        """
        code = self.digits_code(self.symbol)
        return ak.index_stock_cons_weight_csindex(symbol=code)

    def spot_em(self, group: str = "沪深重要指数") -> pd.DataFrame:
        """
        Realtime index spot quotes from Eastmoney.
        Underlying AkShare: ak.stock_zh_index_spot_em(symbol=group)

        This method tries to match exact code (digits), e.g. 000300.
        """
        df = ak.stock_zh_index_spot_em(symbol=group)
        code = self.digits_code(self.symbol)

        if "代码" in df.columns:
            # Exact match is safer than 'contains'
            df = df[df["代码"].astype(str).str.strip() == code]

        return df.reset_index(drop=True)



