import math
import backtrader as bt


# =========================
# 1) Indicators
# =========================
class SuperTrend(bt.Indicator):
    lines = ("st", "trend", "fu", "fl")
    params = dict(
        atr_period=10,
        atr_mult=2.3,
        flip_atr_buffer=0.3,   # ATR buffer to reduce flip noise
    )

    def __init__(self):
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)
        self.hl2 = (self.data.high + self.data.low) / 2.0
        self.upper = self.hl2 + self.p.atr_mult * self.atr
        self.lower = self.hl2 - self.p.atr_mult * self.atr
        self._started = False

    def next(self):
        a = float(self.atr[0])
        if math.isnan(a):
            self.l.st[0] = float("nan")
            self.l.trend[0] = float("nan")
            self.l.fu[0] = float("nan")
            self.l.fl[0] = float("nan")
            return

        if (not self._started) or math.isnan(float(self.l.trend[-1])):
            self._started = True
            self.l.fu[0] = float(self.upper[0])
            self.l.fl[0] = float(self.lower[0])
            self.l.trend[0] = 1.0
            self.l.st[0] = self.l.fl[0]
            return

        prev_fu = float(self.l.fu[-1])
        prev_fl = float(self.l.fl[-1])
        prev_trend = float(self.l.trend[-1])
        prev_close = float(self.data.close[-1])

        up = float(self.upper[0])
        lo = float(self.lower[0])

        fu = up if (up < prev_fu or prev_close > prev_fu) else prev_fu
        fl = lo if (lo > prev_fl or prev_close < prev_fl) else prev_fl

        c = float(self.data.close[0])
        buf = float(self.atr[0]) * float(self.p.flip_atr_buffer)

        if prev_trend == 1.0:
            if c < (fl - buf):
                trend = -1.0
                st = fu
            else:
                trend = 1.0
                st = fl
        else:
            if c > (fu + buf):
                trend = 1.0
                st = fl
            else:
                trend = -1.0
                st = fu

        self.l.fu[0] = fu
        self.l.fl[0] = fl
        self.l.trend[0] = trend
        self.l.st[0] = st


class DebounceBinary(bt.Indicator):
    lines = ("out",)
    params = dict(confirm_days=3)

    def __init__(self):
        self._last = None
        self._cand = None
        self._cnt = 0

    def next(self):
        x = float(self.data[0])
        if math.isnan(x):
            self.l.out[0] = float("nan")
            return

        if self._last is None:
            self._last = x
            self.l.out[0] = self._last
            return

        if x == self._last:
            self._cand = None
            self._cnt = 0
            self.l.out[0] = self._last
            return

        if self._cand is None or self._cand != x:
            self._cand = x
            self._cnt = 1
        else:
            self._cnt += 1

        if self._cnt >= self.p.confirm_days:
            self._last = self._cand
            self._cand = None
            self._cnt = 0

        self.l.out[0] = self._last


# =========================
# 2) Strategy (Bull/Bear only + StopLoss + TakeProfit)
# =========================
class BullBear_ST_StopLoss_TakeProfit_Strategy(bt.Strategy):
    params = dict(
        # --- SuperTrend ---
        st_atr_period=10,
        st_atr_mult=2.3,
        st_flip_atr_buffer=0.3,
        bullbear_confirm=2,

        # --- Position ---
        bull_buy_target=0.98,   # bull -> target exposure
        cash_buffer=0.02,       # when increasing exposure: target -= cash_buffer (avoid Margin)

        # --- Stop loss ---
        stop_loss_pct=0.15,     # -10% from average entry price

        # --- Take Profit ---
        tp_enable=True,
        tp_lookback=200,         # N-day new high breakout
        tp_drawdown=0.10,        # drawdown percentage after peak
        tp_breakout_price="high",  # breakout check uses "high" or "close"
        tp_peak_price="high",      # peak update uses "high" or "close"
        tp_trigger_price="close",  # exit trigger uses "close" or "low"

        # --- Cooldown (optional) ---
        switch_cooldown=5,      # bars after filled order before allowing another buy increase

        # --- Debug ---
        debug_reject=True,
        printlog=False,
    )

    def __init__(self):
        self.data0 = self.datas[0]
        self.order = None
        self._order_reason = {}
        self.trade_marks = []

        self._last_trend = None
        self._last_exec_bar = -10**9

        self.st = SuperTrend(
            self.data0,
            atr_period=self.p.st_atr_period,
            atr_mult=self.p.st_atr_mult,
            flip_atr_buffer=self.p.st_flip_atr_buffer,
        )
        self.trend_db = DebounceBinary(self.st.trend, confirm_days=self.p.bullbear_confirm)

        # =========================
        # Take-profit state (after breakout)
        # =========================
        self._tp_active = False
        self._tp_peak = float("nan")

        self._tp_breakout_line = getattr(self.data0, self.p.tp_breakout_price)
        self._tp_peak_line = getattr(self.data0, self.p.tp_peak_price)
        self._tp_trigger_line = getattr(self.data0, self.p.tp_trigger_price)

        # Rolling highest line for breakout detection (exclude current bar by using [-1])
        self._tp_hh = bt.ind.Highest(self._tp_breakout_line, period=int(self.p.tp_lookback))

    def _in_cooldown(self) -> bool:
        return (len(self) - self._last_exec_bar) < self.p.switch_cooldown

    def _cur_exposure(self) -> float:
        price = float(self.data0.close[0])
        portv = float(self.broker.getvalue())
        if portv <= 0:
            return 0.0
        posv = float(self.position.size) * price
        return posv / portv

    def _submit_target(self, target: float, reason: str):
        cur_exp = self._cur_exposure()
        tgt = float(target)

        # Only apply buffer when increasing exposure
        if tgt > cur_exp + 1e-9:
            tgt = max(0.0, min(1.0, tgt - float(self.p.cash_buffer)))

        o = self.order_target_percent(data=self.data0, target=tgt)
        if o is not None:
            self._order_reason[o.ref] = f"{reason} | target={tgt:.2f}"
        self.order = o

    def _submit_close(self, reason: str):
        o = self.close(data=self.data0)
        if o is not None:
            self._order_reason[o.ref] = reason
        self.order = o

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        dt = self.data0.datetime.datetime(0)

        if order.status in [order.Completed]:
            self._last_exec_bar = len(self)

            reason = self._order_reason.pop(order.ref, "UNKNOWN")
            side = "BUY" if order.isbuy() else "SELL"
            size = float(order.executed.size)
            price = float(order.executed.price)

            self.trade_marks.append({
                "dt": dt,
                "side": side,
                "size": size,
                "price": price,
                "reason": reason,
            })

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            reason = self._order_reason.pop(order.ref, "UNKNOWN")
            if self.p.debug_reject:
                print(dt.date().isoformat(), "ORDER", order.getstatusname(),
                      "isbuy=", order.isbuy(),
                      "created_size=", order.created.size,
                      "price=", order.created.price,
                      "reason=", reason)

        self.order = None

    def next(self):
        if self.order:
            return

        trend = float(self.trend_db.out[0])
        if math.isnan(trend):
            return

        close = float(self.data0.close[0])

        # =========================
        # Stop loss first
        # =========================
        if self.position:
            avg = float(self.position.price)  # avg entry price of current position
            if avg > 0:
                sl_px = avg * (1.0 - float(self.p.stop_loss_pct))
                if close <= sl_px:
                    self._submit_close(f"STOP LOSS: close (close={close:.2f} <= {sl_px:.2f})")
                    self._last_trend = trend
                    return

        # =========================
        # Take-profit: breakout then trailing drawdown
        # =========================
        if self.p.tp_enable and self.position and self.position.size > 0:
            brk = float(self._tp_breakout_line[0])
            pk_src = float(self._tp_peak_line[0])
            trig = float(self._tp_trigger_line[0])

            prev_hh = float("nan")
            if len(self) > int(self.p.tp_lookback):
                prev_hh = float(self._tp_hh[-1])  # highest up to previous bar

            # Arm trailing TP only after breakout happens while holding
            if (not self._tp_active) and (not math.isnan(prev_hh)) and (not math.isnan(brk)) and (brk > prev_hh):
                self._tp_active = True
                self._tp_peak = pk_src

            # Update peak if active
            if self._tp_active and (not math.isnan(pk_src)):
                if math.isnan(self._tp_peak):
                    self._tp_peak = pk_src
                else:
                    self._tp_peak = max(self._tp_peak, pk_src)

            # Trigger exit on drawdown
            if self._tp_active and (not math.isnan(self._tp_peak)) and (not math.isnan(trig)):
                tp_line = self._tp_peak * (1.0 - float(self.p.tp_drawdown))
                if trig <= tp_line:
                    self._submit_close(
                        f"TAKE PROFIT: {int(self.p.tp_lookback)}D breakout then DD {self.p.tp_drawdown:.0%} | "
                        f"trig={trig:.2f} <= line={tp_line:.2f} (peak={self._tp_peak:.2f})"
                    )
                    # Avoid repeated close submissions before notify_order
                    self._tp_active = False
                    self._tp_peak = float("nan")
                    return

        # =========================
        # Trend-based trading
        # =========================
        if self._last_trend is None:
            self._last_trend = trend
            if trend == 1.0 and not self.position:
                self._submit_target(self.p.bull_buy_target, "INIT BULL: buy")
            elif trend == -1.0 and self.position:
                self._submit_close("INIT BEAR: close")
            return

        prev = self._last_trend

        # trend switch actions
        if trend != prev:
            if trend == -1.0:
                if self.position:
                    self._submit_close("SWITCH -> BEAR: close")
                self._last_trend = trend
                return

            if trend == 1.0:
                if (not self.position) and (not self._in_cooldown()):
                    self._submit_target(self.p.bull_buy_target, "SWITCH -> BULL: buy")
                self._last_trend = trend
                return

        # same bull trend: if flat (e.g., stopped out), allow re-enter (cooldown)
        if trend == 1.0 and (not self.position) and (not self._in_cooldown()):
            self._submit_target(self.p.bull_buy_target, "BULL: re-enter")

        self._last_trend = trend