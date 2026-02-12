import backtrader as bt
import numpy as np
from collections import defaultdict

class MultiBenchAnalyzer(bt.Analyzer):
    """
    Assumption:
      datas[:-1] = assets universe
      datas[-1]  = benchmark
    Output:
      - portfolio: cum_port, cum_bench, excess, beta, alpha_ann
      - per_asset: {name: {cum_ret, avg_w, cum_contrib}}
    """

    params = dict(annual=252)

    def start(self):
        if len(self.strategy.datas) < 2:
            raise ValueError("Need at least 1 asset + 1 benchmark")

        self.assets = list(self.strategy.datas[:-1])
        self.bench  = self.strategy.datas[-1]

        # --- portfolio series ---
        self.port_prev = None
        self.bench_prev = None
        self.port_rets = []
        self.bench_rets = []

        # --- per-asset series ---
        self.prev_close = {}
        self.asset_rets = defaultdict(list)
        self.prev_weight = {}
        self.weight_hist = defaultdict(list)
        self.contrib_hist = defaultdict(list)

        self._rets = {}

    def next(self):
        # Use benchmark datetime as sampling calendar
        if len(self.bench) == 0:
            return
        dt = self.bench.datetime.datetime(0)

        # require all assets aligned to this dt (strict intersection)
        for d in self.assets:
            if len(d) == 0:
                return
            if d.datetime.datetime(0) != dt:
                return

        # ----- portfolio return -----
        v = float(self.strategy.broker.getvalue())
        if self.port_prev is not None and self.port_prev > 0:
            r = v / self.port_prev - 1.0
            if np.isfinite(r):
                self.port_rets.append(float(r))
        self.port_prev = v

        # ----- benchmark return -----
        b = float(self.bench.close[0])
        if self.bench_prev is not None and self.bench_prev > 0:
            r = b / self.bench_prev - 1.0
            if np.isfinite(r):
                self.bench_rets.append(float(r))
        self.bench_prev = b

        # ----- per-asset return & weights & contribution -----
        total_value = v if v != 0 else np.nan

        for d in self.assets:
            name = getattr(d, "_name", d._dataname if hasattr(d, "_dataname") else "asset")
            close = float(d.close[0])

            # asset return
            if name in self.prev_close and self.prev_close[name] > 0:
                ri = close / self.prev_close[name] - 1.0
                if np.isfinite(ri):
                    self.asset_rets[name].append(float(ri))
            self.prev_close[name] = close

            # weight (end-of-day mark-to-market)
            pos = self.strategy.getposition(d)
            mv = float(pos.size) * close
            wi = mv / total_value if np.isfinite(total_value) else np.nan
            if np.isfinite(wi):
                self.weight_hist[name].append(float(wi))

            # contribution (prev weight * today return)
            if name in self.prev_weight and len(self.asset_rets[name]) > 0:
                contrib = self.prev_weight[name] * self.asset_rets[name][-1]
                if np.isfinite(contrib):
                    self.contrib_hist[name].append(float(contrib))

            # update prev weight at end
            if np.isfinite(wi):
                self.prev_weight[name] = float(wi)

    def stop(self):
        port = np.asarray(self.port_rets, dtype=float)
        bench = np.asarray(self.bench_rets, dtype=float)
        n = int(min(len(port), len(bench)))

        if n <= 0:
            self._rets = dict(portfolio={}, per_asset={})
            return

        port = port[-n:]
        bench = bench[-n:]

        cum_port = float(np.prod(1.0 + port) - 1.0)
        cum_bench = float(np.prod(1.0 + bench) - 1.0)
        excess = float(cum_port - cum_bench)

        # beta/alpha
        beta, alpha_ann = float("nan"), float("nan")
        if n >= 2:
            var_b = float(np.var(bench, ddof=1))
            if var_b > 0:
                cov = float(np.cov(port, bench, ddof=1)[0, 1])
                beta = cov / var_b
                alpha_daily = float(np.mean(port) - beta * np.mean(bench))
                alpha_ann = float((1.0 + alpha_daily) ** self.p.annual - 1.0)

        per_asset = {}
        for name, rets in self.asset_rets.items():
            arr = np.asarray(rets, dtype=float)
            cum_ret = float(np.prod(1.0 + arr) - 1.0) if len(arr) else 0.0
            ws = np.asarray(self.weight_hist[name], dtype=float)
            avg_w = float(np.nanmean(ws)) if len(ws) else float("nan")
            cs = np.asarray(self.contrib_hist[name], dtype=float)
            cum_contrib = float(np.nansum(cs)) if len(cs) else 0.0

            per_asset[name] = dict(
                cum_ret=cum_ret,
                avg_weight=avg_w,
                cum_contrib=cum_contrib
            )

        self._rets = dict(
            portfolio=dict(
                cum_port=cum_port,
                cum_bench=cum_bench,
                excess=excess,
                beta=float(beta),
                alpha_ann=float(alpha_ann),
                n_days=int(n),
            ),
            per_asset=per_asset,
        )

    def get_analysis(self):
        return self._rets
