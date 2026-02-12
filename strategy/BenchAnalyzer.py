import backtrader as bt
import numpy as np


class BenchAnalyzer(bt.Analyzer):
    """
    Output:
      - cum_port:   strategy cumulative return
      - cum_bench:  benchmark cumulative return
      - excess:     cum_port - cum_bench
      - beta:       OLS beta of port vs bench (daily)
      - alpha_ann:  annualized alpha estimated from daily alpha

    Assumption (no change to Strategy usage):
      datas[0] = asset (the one Strategy trades, i.e. self.data)
      datas[1] = benchmark (e.g. HS300)
    """

    def start(self):
        # Ensure we bind the right feeds by position (since Strategy uses self.data)
        if len(self.strategy.datas) < 2:
            raise ValueError("BenchAnalyzer requires 2 data feeds: asset (datas[0]) and benchmark (datas[1]).")

        self.asset = self.strategy.datas[0]
        self.bench = self.strategy.datas[1]

        print("[BenchAnalyzer] asset name =", getattr(self.asset, "_name", "asset"))
        print("[BenchAnalyzer] bench name =", getattr(self.bench, "_name", "bench"))

        self.port_prev = None
        self.bench_prev = None
        self.port_rets = []
        self.bench_rets = []
        self._rets = {}

    def next(self):
        """
        Record one-day returns ONLY when:
          - asset has a bar (always true for next())
          - benchmark also has a bar for the same datetime (avoid misalignment)
        """
        # Backtrader time alignment guard:
        # If benchmark hasn't advanced to the asset's current datetime, skip
        dt_asset = self.asset.datetime.datetime(0)

        # If bench has no bar loaded yet, skip
        if len(self.bench) == 0:
            return

        dt_bench = self.bench.datetime.datetime(0)
        if dt_bench != dt_asset:
            # Not aligned on this step (benchmark missing this day or different calendar)
            return

        # ----- portfolio return (by broker equity curve) -----
        v = float(self.strategy.broker.getvalue())
        if self.port_prev is not None and self.port_prev > 0:
            r = v / self.port_prev - 1.0
            if np.isfinite(r):
                self.port_rets.append(float(r))
        self.port_prev = v

        # ----- benchmark return (by close price) -----
        bench_close = float(self.bench.close[0])
        if self.bench_prev is not None and self.bench_prev > 0:
            r = bench_close / self.bench_prev - 1.0
            if np.isfinite(r):
                self.bench_rets.append(float(r))
        self.bench_prev = bench_close

    def stop(self):
        """
        Compute final metrics once at the end.
        """
        port = np.asarray(self.port_rets, dtype=float)
        bench = np.asarray(self.bench_rets, dtype=float)

        n = int(min(len(port), len(bench)))
        if n <= 0:
            self._rets = dict(
                cum_port=0.0,
                cum_bench=0.0,
                excess=0.0,
                beta=float("nan"),
                alpha_ann=float("nan"),
            )
            return

        port = port[-n:]
        bench = bench[-n:]

        cum_port = float(np.prod(1.0 + port) - 1.0)
        cum_bench = float(np.prod(1.0 + bench) - 1.0)
        excess = float(cum_port - cum_bench)

        # beta & alpha
        if n >= 2:
            var_b = float(np.var(bench, ddof=1))
            if var_b > 0:
                cov = float(np.cov(port, bench, ddof=1)[0, 1])
                beta = cov / var_b
                alpha_daily = float(np.mean(port) - beta * np.mean(bench))
                alpha_ann = float((1.0 + alpha_daily) ** 252 - 1.0)
            else:
                beta, alpha_ann = float("nan"), float("nan")
        else:
            beta, alpha_ann = float("nan"), float("nan")

        self._rets = dict(
            cum_port=cum_port,
            cum_bench=cum_bench,
            excess=excess,
            beta=float(beta),
            alpha_ann=float(alpha_ann),
        )

    def get_analysis(self):
        return self._rets
