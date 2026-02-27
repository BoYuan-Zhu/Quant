import backtrader as bt
import numpy as np
import math
from collections import defaultdict

class MultiBenchAnalyzer(bt.Analyzer):
    """
    Enhanced MultiBenchAnalyzer:
      - Keeps original outputs (cum_port / benches alpha beta / per_asset)
      - Adds:
          portfolio.stats: sharpe_ann, vol_ann, cagr, calmar, avg_daily_ret, n_days
          portfolio.drawdown: max_dd_pct, max_dd_len
          portfolio.trades: closed, won, lost, winrate, avg_win, avg_loss, profit_factor, expectancy, avg_hold_bars
    """

    params = dict(
        annual=252,
        bench_names=('hs300',),
        strict_align=True,

        # performance params
        riskfreerate=0.02,     # annual risk-free rate
        ddof=1,                # std ddof
    )

    def start(self):
        if len(self.strategy.datas) < 2:
            raise ValueError("Need at least 1 asset + 1 benchmark")

        self.benches = []
        self.assets = []
        for d in self.strategy.datas:
            name = getattr(d, "_name", None)
            if name in self.p.bench_names:
                self.benches.append(d)
            else:
                self.assets.append(d)

        if len(self.assets) < 1:
            raise ValueError("No assets detected (all datas are treated as benchmarks?)")
        if len(self.benches) < 1:
            raise ValueError("No benchmarks detected. Please set bench_names correctly.")

        # --- portfolio series ---
        self.port_prev = None
        self.port_rets = []
        self.port_values = []      # NEW: store value path for drawdown/CAGR
        self.sample_dts = []       # optional
        self.seen_dt = set()       # NEW: avoid double-sampling in non-strict multi-bench mode

        # --- per-benchmark series ---
        self.bench_prev = {self._dname(b): None for b in self.benches}
        self.bench_rets = {self._dname(b): [] for b in self.benches}

        # --- per-asset series ---
        self.prev_close = {}
        self.asset_rets = defaultdict(list)
        self.prev_weight = {}
        self.weight_hist = defaultdict(list)
        self.contrib_hist = defaultdict(list)

        # --- trade stats (NEW) ---
        self.trade_pnls = []
        self.trade_hold_bars = []

        self._rets = {}

    def _dname(self, d):
        return getattr(d, "_name", d._dataname if hasattr(d, "_dataname") else "data")

    def _all_assets_aligned_to(self, dt):
        for d in self.assets:
            if len(d) == 0:
                return False
            if d.datetime.datetime(0) != dt:
                return False
        return True

    # NEW: capture closed trades for winrate etc.
    def notify_trade(self, trade):
        if trade.isclosed:
            pnl = float(trade.pnlcomm)
            if math.isfinite(pnl):
                self.trade_pnls.append(pnl)
            # barlen is holding period in bars (may be 0/None in some cases)
            try:
                bl = int(trade.barlen)
                if bl >= 0:
                    self.trade_hold_bars.append(bl)
            except Exception:
                pass

    def next(self):
        if self.p.strict_align:
            b0 = self.benches[0]
            if len(b0) == 0:
                return
            dt = b0.datetime.datetime(0)

            for b in self.benches:
                if len(b) == 0 or b.datetime.datetime(0) != dt:
                    return

            if not self._all_assets_aligned_to(dt):
                return

            self._sample_once(dt, only_bench=None)

        else:
            # sample separately per benchmark calendar (tolerant)
            for b in self.benches:
                if len(b) == 0:
                    continue
                dt = b.datetime.datetime(0)
                if not self._all_assets_aligned_to(dt):
                    continue
                self._sample_once(dt, only_bench=self._dname(b))

    def _sample_once(self, dt, only_bench=None):
        """
        If only_bench is provided:
          - update that benchmark return series
          - but portfolio/assets are sampled ONLY ONCE per dt (avoid double count)
        """

        # ----- portfolio return/value (sample once per dt) -----
        sample_port = (dt not in self.seen_dt)
        if sample_port:
            v = float(self.strategy.broker.getvalue())

            # store value path
            self.port_values.append(v)
            self.sample_dts.append(dt)

            if self.port_prev is not None and self.port_prev > 0:
                r = v / self.port_prev - 1.0
                if np.isfinite(r):
                    self.port_rets.append(float(r))
            self.port_prev = v

            self.seen_dt.add(dt)
        else:
            # still need v for weights if already sampled? skip assets if already sampled
            v = None

        # ----- benchmark returns -----
        for b in self.benches:
            bname = self._dname(b)
            if only_bench is not None and bname != only_bench:
                continue

            px = float(b.close[0])
            prev = self.bench_prev[bname]
            if prev is not None and prev > 0:
                r = px / prev - 1.0
                if np.isfinite(r):
                    self.bench_rets[bname].append(float(r))
            self.bench_prev[bname] = px

        # ----- per-asset return & weights & contribution (sample once per dt) -----
        if not sample_port:
            return

        total_value = float(self.strategy.broker.getvalue())
        total_value = total_value if total_value != 0 else np.nan

        for d in self.assets:
            name = self._dname(d)
            close = float(d.close[0])

            # asset return
            if name in self.prev_close and self.prev_close[name] > 0:
                ri = close / self.prev_close[name] - 1.0
                if np.isfinite(ri):
                    self.asset_rets[name].append(float(ri))
            self.prev_close[name] = close

            # weight
            pos = self.strategy.getposition(d)
            mv = float(pos.size) * close
            wi = mv / total_value if np.isfinite(total_value) else np.nan
            if np.isfinite(wi):
                self.weight_hist[name].append(float(wi))

            # contribution (prev_weight * last_return)
            if name in self.prev_weight and len(self.asset_rets[name]) > 0:
                contrib = self.prev_weight[name] * self.asset_rets[name][-1]
                if np.isfinite(contrib):
                    self.contrib_hist[name].append(float(contrib))

            if np.isfinite(wi):
                self.prev_weight[name] = float(wi)

    def stop(self):
        port = np.asarray(self.port_rets, dtype=float)
        port_n = int(len(port))

        if port_n == 0:
            self._rets = dict(portfolio={}, per_asset={})
            return

        # ---------- original cum_port ----------
        cum_port = float(np.prod(1.0 + port) - 1.0)

        # ---------- NEW: drawdown from value path ----------
        vals = np.asarray(self.port_values, dtype=float)
        max_dd_pct = float("nan")
        max_dd_len = 0
        if len(vals) >= 2 and np.all(np.isfinite(vals)) and np.nanmin(vals) > 0:
            peak = np.maximum.accumulate(vals)
            dd = (vals / peak - 1.0)  # negative or 0
            max_dd_pct = float(-np.min(dd) * 100.0)

            # drawdown length in bars
            in_dd = vals < peak
            cur = 0
            best = 0
            for x in in_dd:
                if x:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
            max_dd_len = int(best)

        # ---------- NEW: sharpe/vol/cagr/calmar ----------
        rf_daily = (1.0 + float(self.p.riskfreerate)) ** (1.0 / float(self.p.annual)) - 1.0
        excess = port - rf_daily

        sharpe_ann = float("nan")
        vol_ann = float("nan")
        avg_daily_ret = float(np.mean(port))

        if port_n >= 2:
            std_ex = float(np.std(excess, ddof=self.p.ddof))
            if std_ex > 0 and math.isfinite(std_ex):
                sharpe_ann = float((np.mean(excess) / std_ex) * math.sqrt(self.p.annual))

            std_r = float(np.std(port, ddof=self.p.ddof))
            if std_r >= 0 and math.isfinite(std_r):
                vol_ann = float(std_r * math.sqrt(self.p.annual))

        # CAGR uses value path
        cagr = float("nan")
        calmar = float("nan")
        if len(vals) >= 2 and vals[0] > 0 and vals[-1] > 0:
            years = port_n / float(self.p.annual)
            if years > 0:
                cagr = float((vals[-1] / vals[0]) ** (1.0 / years) - 1.0)
                if max_dd_pct and math.isfinite(max_dd_pct) and max_dd_pct > 0:
                    calmar = float(cagr / (max_dd_pct / 100.0))

        # ---------- NEW: trade winrate ----------
        pnls = np.asarray(self.trade_pnls, dtype=float) if len(self.trade_pnls) else np.asarray([], dtype=float)
        closed = int(len(pnls))
        won = int(np.sum(pnls > 0)) if closed else 0
        lost = int(np.sum(pnls < 0)) if closed else 0
        winrate = float(won / closed) if closed else float("nan")

        avg_win = float(np.mean(pnls[pnls > 0])) if won else float("nan")
        avg_loss = float(np.mean(pnls[pnls < 0])) if lost else float("nan")  # negative
        profit_factor = float("nan")
        if won and lost:
            gp = float(np.sum(pnls[pnls > 0]))
            gl = float(-np.sum(pnls[pnls < 0]))
            if gl > 0:
                profit_factor = float(gp / gl)

        expectancy = float(np.mean(pnls)) if closed else float("nan")

        hold = np.asarray(self.trade_hold_bars, dtype=float) if len(self.trade_hold_bars) else np.asarray([], dtype=float)
        avg_hold_bars = float(np.mean(hold)) if len(hold) else float("nan")

        # ---------- original benchmark block (kept) ----------
        benches_out = {}
        for bname, brets_list in self.bench_rets.items():
            bench = np.asarray(brets_list, dtype=float)
            n = int(min(len(port), len(bench)))
            if n <= 0:
                benches_out[bname] = dict(
                    cum_bench=float("nan"),
                    excess=float("nan"),
                    beta=float("nan"),
                    alpha_ann=float("nan"),
                    n_days=0,
                )
                continue

            p = port[-n:]
            b = bench[-n:]

            cum_bench = float(np.prod(1.0 + b) - 1.0)
            excess_ret = float(cum_port - cum_bench)

            beta, alpha_ann = float("nan"), float("nan")
            if n >= 2:
                var_b = float(np.var(b, ddof=1))
                if var_b > 0:
                    cov = float(np.cov(p, b, ddof=1)[0, 1])
                    beta = cov / var_b
                    alpha_daily = float(np.mean(p) - beta * np.mean(b))
                    alpha_ann = float((1.0 + alpha_daily) ** self.p.annual - 1.0)

            benches_out[bname] = dict(
                cum_bench=float(cum_bench),
                excess=float(excess_ret),
                beta=float(beta),
                alpha_ann=float(alpha_ann),
                n_days=int(n),
            )

        # ---------- original per-asset block (kept) ----------
        per_asset = {}
        for name, rets in self.asset_rets.items():
            arr = np.asarray(rets, dtype=float)
            cum_ret = float(np.prod(1.0 + arr) - 1.0) if len(arr) else 0.0
            ws = np.asarray(self.weight_hist[name], dtype=float)
            avg_w = float(np.nanmean(ws)) if len(ws) else float("nan")
            cs = np.asarray(self.contrib_hist[name], dtype=float)
            cum_contrib = float(np.nansum(cs)) if len(cs) else 0.0

            per_asset[name] = dict(
                cum_ret=float(cum_ret),
                avg_weight=float(avg_w),
                cum_contrib=float(cum_contrib),
            )

        self._rets = dict(
            portfolio=dict(
                cum_port=float(cum_port),
                n_days=int(port_n),
                benches=benches_out,

                # NEW blocks
                stats=dict(
                    sharpe_ann=float(sharpe_ann),
                    vol_ann=float(vol_ann),
                    cagr=float(cagr),
                    calmar=float(calmar),
                    avg_daily_ret=float(avg_daily_ret),
                ),
                drawdown=dict(
                    max_dd_pct=float(max_dd_pct),
                    max_dd_len=int(max_dd_len),
                ),
                trades=dict(
                    closed=int(closed),
                    won=int(won),
                    lost=int(lost),
                    winrate=float(winrate),
                    avg_win=float(avg_win),
                    avg_loss=float(avg_loss),
                    profit_factor=float(profit_factor),
                    expectancy=float(expectancy),
                    avg_hold_bars=float(avg_hold_bars),
                ),
            ),
            per_asset=per_asset,
        )

    def get_analysis(self):
        return self._rets