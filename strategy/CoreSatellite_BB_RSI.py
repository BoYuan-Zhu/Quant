import math
import backtrader as bt


class CoreSatellite_BB_RSI(bt.Strategy):
    params = dict(
        # ===== Trend filter (price-derived) =====
        trend_ma_period=200,      # long-term trend filter
        trend_use_ema=False,      # True -> EMA, False -> SMA

        # ===== Bollinger =====
        bb_period=20,
        bb_dev=2.0,

        # ===== RSI =====
        rsi_period=14,
        rsi_buy=45.0,             # satellite add when RSI low
        rsi_sell=60.0,            # satellite reduce when RSI high

        # ===== Core / Satellite exposure =====
        core_target=0.60,         # core exposure when uptrend
        sat_step=0.20,            # each add/reduce step
        sat_max=0.40,             # max satellite exposure
        max_target=0.98,
        min_target=0.00,
        cash_buffer=0.02,         # only when increasing exposure

        # ===== Triggers =====
        # satellite add: price <= mid (or <= bot for more conservative)
        sat_add_on="mid",         # "mid" or "bot"
        # satellite reduce: price >= top
        sat_reduce_on="top",

        # ===== Risk / churn control =====
        stop_loss_pct=0.10,       # hard stop from avg entry
        trail_enable=True,
        trail_dd=0.12,            # peak drawdown trailing exit
        min_bars_between_orders=5,

        debug_reject=True,
        printlog=False,
    )

    def __init__(self):
        d = self.datas[0]
        self.data0 = d
        self.order = None
        self._last_order_bar = -10**9

        # indicators
        if self.p.trend_use_ema:
            self.trend_ma = bt.ind.EMA(d.close, period=int(self.p.trend_ma_period))
        else:
            self.trend_ma = bt.ind.SMA(d.close, period=int(self.p.trend_ma_period))

        self.bb = bt.ind.BollingerBands(d.close, period=int(self.p.bb_period), devfactor=float(self.p.bb_dev))
        self.rsi = bt.ind.RSI(d.close, period=int(self.p.rsi_period))

        # satellite state
        self._sat_target = 0.0
        self._peak_since_entry = float("nan")

    # -------- helpers --------
    def _bars_since(self, bar_index: int) -> int:
        return int(len(self) - bar_index)

    def _cur_exposure(self) -> float:
        price = float(self.data0.close[0])
        portv = float(self.broker.getvalue())
        if portv <= 0:
            return 0.0
        posv = float(self.position.size) * price
        return posv / portv

    def _submit_target(self, target: float, reason: str):
        if self._bars_since(self._last_order_bar) < int(self.p.min_bars_between_orders):
            return

        cur_exp = self._cur_exposure()
        tgt = float(target)

        # cash buffer only when increasing exposure
        if tgt > cur_exp + 1e-9:
            tgt = max(0.0, min(1.0, tgt - float(self.p.cash_buffer)))

        o = self.order_target_percent(data=self.data0, target=tgt)
        if o is not None and self.p.printlog:
            dt = self.data0.datetime.date(0).isoformat()
            print(dt, f"TARGET={tgt:.2f}", reason)
        self.order = o

    def _submit_close(self, reason: str):
        if self._bars_since(self._last_order_bar) < int(self.p.min_bars_between_orders):
            return
        o = self.close(data=self.data0)
        if o is not None and self.p.printlog:
            dt = self.data0.datetime.date(0).isoformat()
            print(dt, "CLOSE", reason)
        self.order = o

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        dt = self.data0.datetime.date(0).isoformat()

        if order.status in [order.Completed]:
            self._last_order_bar = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.p.debug_reject:
                print(dt, "ORDER", order.getstatusname(),
                      "isbuy=", order.isbuy(),
                      "created_size=", order.created.size,
                      "price=", order.created.price)

        self.order = None

    def next(self):
        if self.order:
            return

        close = float(self.data0.close[0])
        ma = float(self.trend_ma[0])
        rsi = float(self.rsi[0])

        if any(map(math.isnan, [close, ma, rsi])):
            return

        uptrend = close >= ma

        # ----- risk management first -----
        if self.position:
            # update peak
            h0 = float(self.data0.high[0])
            if not math.isnan(h0):
                if math.isnan(self._peak_since_entry):
                    self._peak_since_entry = h0
                else:
                    self._peak_since_entry = max(self._peak_since_entry, h0)

            # stop loss from avg entry
            avg = float(self.position.price)
            if avg > 0:
                sl = avg * (1.0 - float(self.p.stop_loss_pct))
                if close <= sl:
                    self._sat_target = 0.0
                    self._peak_since_entry = float("nan")
                    self._submit_close(f"STOP LOSS: close={close:.2f} <= {sl:.2f}")
                    return

            # trailing exit
            if self.p.trail_enable and (not math.isnan(self._peak_since_entry)) and self._peak_since_entry > 0:
                line = self._peak_since_entry * (1.0 - float(self.p.trail_dd))
                if close <= line:
                    self._sat_target = 0.0
                    self._peak_since_entry = float("nan")
                    self._submit_close(f"TRAIL EXIT: close={close:.2f} <= {line:.2f}")
                    return
        else:
            self._peak_since_entry = float("nan")

        # ----- if not uptrend: go flat (or you can keep small core if you want) -----
        if not uptrend:
            self._sat_target = 0.0
            if self.position:
                self._submit_close("DOWN TREND: exit all")
            return

        # ----- in uptrend: core stays + satellite trades -----
        core = float(self.p.core_target)

        # choose add line
        add_line = float(self.bb.mid[0]) if self.p.sat_add_on == "mid" else float(self.bb.bot[0])
        reduce_line = float(self.bb.top[0])

        # add satellite on pullback + low RSI
        if (close <= add_line) and (rsi <= float(self.p.rsi_buy)):
            self._sat_target = min(float(self.p.sat_max), self._sat_target + float(self.p.sat_step))

        # reduce satellite on extension + high RSI
        if (close >= reduce_line) and (rsi >= float(self.p.rsi_sell)):
            self._sat_target = max(0.0, self._sat_target - float(self.p.sat_step))

        total = core + self._sat_target
        total = min(float(self.p.max_target), max(float(self.p.min_target), total))

        # avoid tiny adjustments
        cur = self._cur_exposure()
        if abs(total - cur) >= 0.05:
            self._submit_target(total, f"UPTREND core={core:.2f} sat={self._sat_target:.2f} rsi={rsi:.1f}")