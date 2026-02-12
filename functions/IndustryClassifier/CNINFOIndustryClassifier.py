"""
One-file pipeline: A-share universe (AkShare) + CNINFO industry mapping + CSV cache

Outputs (offline-first):
  - data/universe_spot_sina.csv              (universe snapshot)
  - data/cninfo_stock_industry_latest.csv    (stock -> industry latest)
  - data/universe_with_industry.csv          (optional merge)

Universe source:
  - ak.stock_zh_a_spot()  # Sina spot snapshot

Industry source:
  - ak.stock_industry_change_cninfo(symbol=..., start_date, end_date)

English columns:
  universe: symbol_raw, symbol, name, close, open, high, low, volume, amount
  mapping : symbol, industry_standard, industry_l1, industry_l2, industry_l3, change_date
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict
from collections import Counter

import pandas as pd
import akshare as ak
from akshare.utils.demjson import JSONDecodeError


# =========================
# Helpers
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def norm_symbol_6(x: str) -> str:
    s = str(x).strip()
    s = "".join(ch for ch in s if ch.isdigit())
    return s.zfill(6)

def _clean_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s

def retry_call(fn, *args, retries: int = 5, backoff: float = 1.6, sleep0: float = 0.3, **kwargs):
    last = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            time.sleep(sleep0 * (backoff ** i))
    raise RuntimeError(f"AKShare call failed after {retries} retries: {repr(last)}")


# =========================
# Universe (AkShare / Sina)
# =========================

@dataclass
class UniverseSchema:
    symbol_raw: str = "symbol_raw"
    symbol: str = "symbol"
    name: str = "name"
    close: str = "close"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    volume: str = "volume"
    amount: str = "amount"


class AShareUniverse:
    """
    Offline-first A-share universe snapshot from Sina spot (ak.stock_zh_a_spot).

    Output columns (English):
      symbol_raw, symbol, name, close, open, high, low, volume, amount
    """

    def __init__(self, schema: UniverseSchema = UniverseSchema()):
        self.schema = schema
        self.df: Optional[pd.DataFrame] = None

    @staticmethod
    def _pick_col(cols: List[str], candidates: List[str], required: bool = True) -> Optional[str]:
        for c in candidates:
            if c in cols:
                return c
        lower_map = {str(c).lower(): c for c in cols}
        for cand in candidates:
            cand_l = str(cand).lower()
            for lc, orig in lower_map.items():
                if cand_l in lc:
                    return orig
        if required:
            raise KeyError(f"Cannot find required column: candidates={candidates}, got={cols}")
        return None

    def load_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, dtype=str)
        if self.schema.symbol not in df.columns:
            raise KeyError(f"'{self.schema.symbol}' not found in {path}")
        df = df.copy()
        df[self.schema.symbol] = df[self.schema.symbol].map(norm_symbol_6)
        self.df = df
        return df

    def save_csv(self, path: str) -> None:
        if self.df is None:
            raise ValueError("Universe is empty.")
        ensure_dir(os.path.dirname(path) or ".")
        self.df.to_csv(path, index=False, encoding="utf-8-sig")

    def fetch_spot_sina(self, retries: int = 6, backoff: float = 1.7) -> pd.DataFrame:
        """
        Fetch Sina spot snapshot with retry because ak.stock_zh_a_spot sometimes returns HTML
        which triggers JSONDecodeError.
        """
        last_err = None
        for i in range(retries):
            try:
                df = ak.stock_zh_a_spot()
                if df is None or df.empty:
                    raise RuntimeError("ak.stock_zh_a_spot() returned empty dataframe.")
                out = self._normalize_sina_spot(df)
                self.df = out
                return out
            except JSONDecodeError as e:
                last_err = e
                time.sleep(backoff ** i)
            except Exception as e:
                last_err = e
                time.sleep(backoff ** i)
        raise RuntimeError(f"Sina spot failed after {retries} retries. Last error: {repr(last_err)}")

    def fetch_or_load(self, cache_csv: str = "data/universe_spot_sina.csv", force_fetch: bool = False) -> pd.DataFrame:
        if (not force_fetch) and os.path.exists(cache_csv):
            return self.load_csv(cache_csv)
        df = self.fetch_spot_sina()
        self.save_csv(cache_csv)
        return df

    def _normalize_sina_spot(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols = list(df.columns)

        c_symbol = self._pick_col(cols, ["代码", "symbol", "证券代码", "股票代码"])
        c_name = self._pick_col(cols, ["名称", "name", "股票简称", "证券简称"], required=False)

        c_close = self._pick_col(cols, ["最新价", "close", "现价", "最新", "收盘"], required=False)
        c_open = self._pick_col(cols, ["今开", "open", "开盘"], required=False)
        c_high = self._pick_col(cols, ["最高", "high"], required=False)
        c_low = self._pick_col(cols, ["最低", "low"], required=False)

        c_vol = self._pick_col(cols, ["成交量", "volume"], required=False)
        c_amt = self._pick_col(cols, ["成交额", "amount", "成交额(元)"], required=False)

        out = pd.DataFrame()
        out[self.schema.symbol_raw] = df[c_symbol].astype(str).str.strip()
        out[self.schema.symbol] = out[self.schema.symbol_raw].str.replace(r"\D", "", regex=True).map(norm_symbol_6)

        if c_name is not None:
            out[self.schema.name] = df[c_name].astype(str).str.strip()
        if c_close is not None:
            out[self.schema.close] = pd.to_numeric(df[c_close], errors="coerce")
        if c_open is not None:
            out[self.schema.open] = pd.to_numeric(df[c_open], errors="coerce")
        if c_high is not None:
            out[self.schema.high] = pd.to_numeric(df[c_high], errors="coerce")
        if c_low is not None:
            out[self.schema.low] = pd.to_numeric(df[c_low], errors="coerce")
        if c_vol is not None:
            out[self.schema.volume] = pd.to_numeric(df[c_vol], errors="coerce")
        if c_amt is not None:
            out[self.schema.amount] = pd.to_numeric(df[c_amt], errors="coerce")

        return out

    def symbols(self) -> List[str]:
        if self.df is None or self.df.empty:
            raise ValueError("Universe is empty.")
        return self.df[self.schema.symbol].dropna().astype(str).map(norm_symbol_6).tolist()


# =========================
# CNINFO industry mapping
# =========================

@dataclass
class CNINFOIndustrySchema:
    symbol: str = "symbol"
    industry_standard: str = "industry_standard"
    industry_l1: str = "industry_l1"
    industry_l2: str = "industry_l2"
    industry_l3: str = "industry_l3"
    change_date: str = "change_date"


class CNINFOIndustryClassifier:
    """
    Offline-first stock -> industry latest mapping via CNINFO.
    """

    def __init__(
        self,
        cache_csv: str = "data/cninfo_stock_industry_latest.csv",
        use_cache: bool = True,
        schema: CNINFOIndustrySchema = CNINFOIndustrySchema(),
        retries: int = 4,
        backoff: float = 1.6,
        sleep_each: float = 0.03,
    ):
        self.cache_csv = cache_csv
        self.use_cache = use_cache
        self.schema = schema
        self.retries = retries
        self.backoff = backoff
        self.sleep_each = sleep_each
        self.df: Optional[pd.DataFrame] = None

    def load_cache(self) -> pd.DataFrame:
        df = pd.read_csv(self.cache_csv, dtype=str)
        need = {
            self.schema.symbol,
            self.schema.industry_standard,
            self.schema.industry_l1,
            self.schema.industry_l2,
            self.schema.industry_l3,
            self.schema.change_date,
        }
        missing = need - set(df.columns)
        if missing:
            raise KeyError(f"Cache missing columns: {missing}")
        df = df.copy()
        df[self.schema.symbol] = df[self.schema.symbol].map(norm_symbol_6)
        self.df = df
        return df

    def save_cache(self) -> None:
        ensure_dir(os.path.dirname(self.cache_csv) or ".")
        if self.df is None:
            cols = [
                self.schema.symbol,
                self.schema.industry_standard,
                self.schema.industry_l1,
                self.schema.industry_l2,
                self.schema.industry_l3,
                self.schema.change_date,
            ]
            pd.DataFrame(columns=cols).to_csv(self.cache_csv, index=False, encoding="utf-8-sig")
            return
        self.df.to_csv(self.cache_csv, index=False, encoding="utf-8-sig")

    def fetch_one_latest(
        self,
        symbol: str,
        priority_standards: Optional[List[str]] = None,
        start_date: str = "19900101",
        end_date: str = "20300101",
    ) -> Optional[Dict]:
        s = norm_symbol_6(symbol)
        df = retry_call(
            ak.stock_industry_change_cninfo,
            symbol=s,
            start_date=start_date,
            end_date=end_date,
            retries=self.retries,
            backoff=self.backoff,
        )
        if df is None or df.empty:
            return None

        df = df.copy()
        if "变更日期" in df.columns:
            df["变更日期"] = pd.to_datetime(df["变更日期"], errors="coerce")

        chosen_std = None
        if priority_standards and "分类标准" in df.columns:
            stds = set(df["分类标准"].dropna().astype(str).str.strip().tolist())
            for cand in priority_standards:
                if cand in stds:
                    chosen_std = cand
                    break

        df2 = df
        if chosen_std is not None and "分类标准" in df.columns:
            tmp = df[df["分类标准"].astype(str).str.strip() == chosen_std].copy()
            if not tmp.empty:
                df2 = tmp

        if "变更日期" in df2.columns:
            df2 = df2.sort_values("变更日期", ascending=False)

        r = df2.iloc[0].to_dict()
        r["symbol"] = s
        return r

    def build_latest(
            self,
            symbols: Iterable[str],
            priority_standards: Optional[List[str]] = None,
            start_date: str = "19900101",
            end_date: str = "20300101",
            force: bool = False,
            save_every: int = 200,  # NEW: periodic save
            max_fail_ratio: float = 0.6,  # NEW: if fail explodes, slow down
    ) -> pd.DataFrame:
        """
        Offline-first + resume + periodic save + fail retry.

        Behavior:
          - If cache exists and not force: load it
          - If cache exists but force=False: resume from it (skip done symbols)
          - Periodically save to cache_csv
          - Collect failures, retry them once with slower pacing
        """
        # Load existing cache to resume
        existing = None
        done = set()
        if self.use_cache and os.path.exists(self.cache_csv) and (not force):
            try:
                existing = self.load_cache()
                done = set(existing[self.schema.symbol].astype(str).map(norm_symbol_6).tolist())
                print(f"[resume] loaded cache rows={len(existing)} done={len(done)}")
            except Exception:
                existing = None
                done = set()

        err = Counter()
        rows = [] if existing is None else existing.to_dict("records")
        fail_syms = []

        sym_list = [norm_symbol_6(x) for x in symbols]
        sym_list = [x for x in sym_list if len(x) == 6]
        total = len(sym_list)

        processed = 0
        success = 0
        fail = 0

        for i, s in enumerate(sym_list, 1):
            if s in done:
                continue

            processed += 1
            try:
                r = self.fetch_one_latest(
                    s,
                    priority_standards=priority_standards,
                    start_date=start_date,
                    end_date=end_date,
                )
                if r is None:
                    # treat as fail but keep for retry
                    fail_syms.append(s)
                    fail += 1
                else:
                    out = {
                        self.schema.symbol: norm_symbol_6(r.get("symbol", s)),
                        self.schema.industry_standard: _clean_str(r.get("分类标准", "")),
                        self.schema.industry_l1: _clean_str(r.get("行业大类", "")),
                        self.schema.industry_l2: _clean_str(r.get("行业中类", "")),
                        self.schema.industry_l3: _clean_str(r.get("行业次类", "")),
                        self.schema.change_date: _clean_str(r.get("变更日期", "")),
                    }
                    rows.append(out)
                    done.add(s)
                    success += 1

            except Exception as e:
                err[type(e).__name__] += 1
                fail_syms.append(s)
                fail += 1

            # periodic progress
            if (success + fail) % 500 == 0:
                print(f"processed {len(done)}/{total} new_success={success} new_fail={fail} errors={dict(err)}")

            # NEW: periodic save
            if self.use_cache and (success + fail) % save_every == 0:
                df_tmp = pd.DataFrame(rows)
                if not df_tmp.empty:
                    df_tmp[self.schema.symbol] = df_tmp[self.schema.symbol].map(norm_symbol_6)
                    df_tmp = df_tmp.drop_duplicates(self.schema.symbol, keep="last").reset_index(drop=True)
                self.df = df_tmp
                self.save_cache()

            # pacing
            time.sleep(self.sleep_each)

            # NEW: if failure ratio spikes, slow down to avoid getting blocked
            if (success + fail) >= 500:
                ratio = fail / max(success + fail, 1)
                if ratio > max_fail_ratio:
                    # slow down progressively
                    self.sleep_each = min(0.30, max(self.sleep_each, 0.10))
                    # also add extra pause occasionally
                    if (success + fail) % 200 == 0:
                        time.sleep(2.0)

        # First pass finalize save
        df_out = pd.DataFrame(rows)
        if not df_out.empty:
            df_out[self.schema.symbol] = df_out[self.schema.symbol].map(norm_symbol_6)
            df_out = df_out.drop_duplicates(self.schema.symbol, keep="last").reset_index(drop=True)
        self.df = df_out
        if self.use_cache:
            self.save_cache()

        print("saved ->", self.cache_csv, "rows=", len(df_out), "errors=", dict(err), "fail_syms=", len(fail_syms))

        # ===== second pass: retry failures with slower pacing =====
        if fail_syms:
            print(f"[retry] retry failed symbols once: n={len(fail_syms)} (slower pacing)")
            old_sleep = self.sleep_each
            self.sleep_each = max(old_sleep, 0.12)  # slow down
            retry_rows = []
            retry_err = Counter()

            for j, s in enumerate(fail_syms, 1):
                if s in set(df_out[self.schema.symbol].astype(str).tolist()):
                    continue
                try:
                    r = self.fetch_one_latest(
                        s,
                        priority_standards=priority_standards,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if r is not None:
                        retry_rows.append({
                            self.schema.symbol: norm_symbol_6(r.get("symbol", s)),
                            self.schema.industry_standard: _clean_str(r.get("分类标准", "")),
                            self.schema.industry_l1: _clean_str(r.get("行业大类", "")),
                            self.schema.industry_l2: _clean_str(r.get("行业中类", "")),
                            self.schema.industry_l3: _clean_str(r.get("行业次类", "")),
                            self.schema.change_date: _clean_str(r.get("变更日期", "")),
                        })
                except Exception as e:
                    retry_err[type(e).__name__] += 1

                if j % 200 == 0:
                    print(f"[retry] {j}/{len(fail_syms)} gained={len(retry_rows)} retry_errors={dict(retry_err)}")
                time.sleep(self.sleep_each)

            if retry_rows:
                df_retry = pd.DataFrame(retry_rows)
                df_out = pd.concat([df_out, df_retry], ignore_index=True)
                df_out[self.schema.symbol] = df_out[self.schema.symbol].map(norm_symbol_6)
                df_out = df_out.drop_duplicates(self.schema.symbol, keep="last").reset_index(drop=True)
                self.df = df_out
                if self.use_cache:
                    self.save_cache()

            self.sleep_each = old_sleep
            print("[retry] done. final rows=", len(df_out), "retry_errors=", dict(retry_err))

        return df_out


# =========================
# Pipeline utilities
# =========================

def build_universe_and_industry_csv(
    universe_csv: str = "data/universe_spot_sina.csv",
    industry_csv: str = "data/cninfo_stock_industry_latest.csv",
    merged_csv: Optional[str] = "data/universe_with_industry.csv",
    force_fetch_universe: bool = False,
    force_build_industry: bool = False,
    priority_standards: Optional[List[str]] = None,
):
    """
    One-click:
      1) fetch/load universe -> save universe_csv
      2) build/load industry mapping -> save industry_csv
      3) optional merge -> save merged_csv
    """
    priority = priority_standards or [
        "申银万国行业分类标准",
        "证监会行业分类标准（2012）",
        "巨潮行业分类标准",
        "中证行业分类标准",
        "中国上市公司协会上市公司行业分类标准",
    ]

    uni = AShareUniverse()
    u = uni.fetch_or_load(universe_csv, force_fetch=force_fetch_universe)
    syms = uni.symbols()
    print("Universe symbols:", len(syms))

    cls = CNINFOIndustryClassifier(cache_csv=industry_csv, use_cache=True)
    m = cls.build_latest(syms, priority_standards=priority, force=force_build_industry)

    if merged_csv is not None:
        out = u.merge(m, on="symbol", how="left")
        ensure_dir(os.path.dirname(merged_csv) or ".")
        out.to_csv(merged_csv, index=False, encoding="utf-8-sig")
        print("saved ->", merged_csv, "rows=", len(out))

    return u, m


# =========================
# Example main
# =========================

if __name__ == "__main__":
    # First time:
    #   - force_fetch_universe=True (optional)
    #   - force_build_industry=True (recommended to rebuild full mapping)
    #
    # Next times:
    #   - set both to False to run fully offline from cached CSVs.
    build_universe_and_industry_csv(
        universe_csv="data/universe_spot_sina.csv",
        industry_csv="data/cninfo_stock_industry_latest.csv",
        merged_csv="data/universe_with_industry.csv",
        force_fetch_universe=False,
        force_build_industry=False,   # <-- set True once you want to build full 5481 mapping
    )
