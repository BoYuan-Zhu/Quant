import math
import backtrader as bt


class TrendDonchian_ATR_Pyramid_Strategy(bt.Strategy):
    params = dict(
        # -------------------------
        # Trend filter (faster MAs => earlier entry)
        # -------------------------
        ema_fast=10,
        ema_slow=30,

        # -------------------------
        # ADX filter (more aggressive)
        # -------------------------
        adx_filter_enable=True,   # set False to ignore ADX completely (most aggressive)
        adx_period=14,
        adx_enter=15,             # lower threshold => earlier entry

        # -------------------------
        # Donchian breakout (shorter => earlier entry)
        # -------------------------
        dc_entry_period=15,
        dc_exit_period=8,
        entry_on_high_break=True,   # True: today's HIGH > donchian_high -> enter earlier than close
        exit_on_low_break=True,     # True: today's LOW < donchian_low -> exit earlier than close

        # -------------------------
        # ATR risk management
        # -------------------------
        atr_period=14,
        stop_atr_mult=2.2,        # slightly wider than 2.0 to avoid easy stop-out
        trail_atr_mult=2.6,       # tighter trailing => lock profits earlier (more "active")

        # -------------------------
        # Position sizing (more aggressive)
        # -------------------------
        base_target=0.70,         # entry exposure
        max_target=0.98,
        min_target=0.00,
        cash_buffer=0.02,         # avoid margin when increasing

        # -------------------------
        # Pyramiding (more aggressive)
        # -------------------------
        pyramid_enable=True,
        pyr_step_atr=0.7,                 # smaller step => add faster
        pyr_incs=(0.15, 0.15, 0.20, 0.20),# bigger & more levels
        pyr_cooldown_bars=2,
        pyr_allow_jump=True,              # if price jumps multiple steps, add multiple levels at once

        # -------------------------
        # Order gating (avoid tiny orders)
        # -------------------------
        min_bars_between_orders=1,
        min_target_change=0.02,   # ignore target change < 2%

        # -------------------------
        # Debug
        # -------------------------
        debug_reject=True,
        printlog=False,
    )

    def __init__(self):
        self.data0 = self.datas[0]
        self.order = None
        self._order_reason = {}
        self.trade_marks = []
        self._last_order_bar = -10**9
        self._last_add_bar = -10**9

        # Indicators
        self.ema_f = bt.ind.EMA(self.data0.close, period=self.p.ema_fast)
        self.ema_s = bt.ind.EMA(self.data0.close, period=self.p.ema_slow)
        self.adx = bt.ind.ADX(self.data0, period=self.p.adx_period)
        self.atr = bt.ind.ATR(self.data0, period=self.p.atr_period)

        # Donchian excluding current bar
        self.dc_high = bt.ind.Highest(self.data0.high(-1), period=self.p.dc_entry_period)
        self.dc_low = bt.ind.Lowest(self.data0.low(-1), period=self.p.dc_exit_period)

        # State
        self._entry_price = None
        self._highest_close = None
        self._last_add_price = None
        self._pyr_stage = 0
        self._prev_pos_size = 0.0

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

        # ignore tiny changes (avoid micro orders)
        if abs(tgt - cur_exp) < float(self.p.min_target_change):
            return

        # cash buffer only when increasing
        if tgt > cur_exp + 1e-9:
            tgt = max(0.0, min(1.0, tgt - float(self.p.cash_buffer)))

        tgt = max(float(self.p.min_target), min(float(self.p.max_target), tgt))

        # re-check after buffer/cap
        if abs(tgt - cur_exp) < float(self.p.min_target_change):
            return

        o = self.order_target_percent(data=self.data0, target=tgt)
        if o is not None:
            self._order_reason[o.ref] = f"{reason} | target={tgt:.2f}"
            self.order = o

    def _submit_close(self, reason: str):
        if self._bars_since(self._last_order_bar) < int(self.p.min_bars_between_orders):
            return
        o = self.close(data=self.data0)
        if o is not None:
            self._order_reason[o.ref] = reason
            self.order = o

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        dt = self.data0.datetime.datetime(0)

        if order.status in [order.Completed]:
            self._last_order_bar = len(self)
            reason = self._order_reason.pop(order.ref, "UNKNOWN")

            self.trade_marks.append({
                "dt": dt,
                "side": "BUY" if order.isbuy() else "SELL",
                "size": float(order.executed.size),
                "price": float(order.executed.price),
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

    def _sync_position_state(self):
        cur_size = float(self.position.size)

        if self._prev_pos_size == 0.0 and cur_size != 0.0:
            # new entry
            self._entry_price = float(self.position.price)
            self._highest_close = float(self.data0.close[0])
            self._last_add_price = float(self.data0.close[0])
            self._pyr_stage = 0
            self._last_add_bar = -10**9

        if self._prev_pos_size != 0.0 and cur_size == 0.0:
            # flat reset
            self._entry_price = None
            self._highest_close = None
            self._last_add_price = None
            self._pyr_stage = 0
            self._last_add_bar = -10**9

        self._prev_pos_size = cur_size

        if cur_size != 0.0 and self._highest_close is not None:
            self._highest_close = max(self._highest_close, float(self.data0.close[0]))

    def next(self):
        self._sync_position_state()
        if self.order:
            return

        c = float(self.data0.close[0])
        h = float(self.data0.high[0])
        l = float(self.data0.low[0])

        ef = float(self.ema_f[0])
        es = float(self.ema_s[0])
        adx = float(self.adx[0])
        atr = float(self.atr[0])
        dh = float(self.dc_high[0])
        dl = float(self.dc_low[0])

        if any(math.isnan(x) for x in [c, h, l, ef, es, adx, atr, dh, dl]):
            return

        trend_up = ef > es
        trend_dn = ef < es

        adx_ok = True
        if self.p.adx_filter_enable:
            adx_ok = adx >= float(self.p.adx_enter)

        # =========================
        # Exits first
        # =========================
        if self.position and self._entry_price is not None:
            stop_px = self._entry_price - float(self.p.stop_atr_mult) * atr
            if c <= stop_px:
                self._submit_close(f"STOP (ATR): close={c:.2f} <= {stop_px:.2f}")
                return

            if self._highest_close is not None:
                trail_px = self._highest_close - float(self.p.trail_atr_mult) * atr
                if c <= trail_px:
                    self._submit_close(f"TRAIL (Chandelier): close={c:.2f} <= {trail_px:.2f}")
                    return

            # Donchian exit (more aggressive: low breaks)
            if self.p.exit_on_low_break:
                if l < dl:
                    self._submit_close(f"EXIT (Donchian-low): low={l:.2f} < dc_low={dl:.2f}")
                    return
            else:
                if c < dl:
                    self._submit_close(f"EXIT (Donchian): close={c:.2f} < dc_low={dl:.2f}")
                    return

            if trend_dn:
                self._submit_close("EXIT (MA): EMA_fast < EMA_slow")
                return

        # =========================
        # Entry
        # =========================
        if not self.position:
            if trend_up and adx_ok:
                if self.p.entry_on_high_break:
                    if h > dh:
                        self._submit_target(float(self.p.base_target), "ENTRY: High breaks Donchian + MA up (+ADX)")
                else:
                    if c > dh:
                        self._submit_target(float(self.p.base_target), "ENTRY: Close breaks Donchian + MA up (+ADX)")
            return

        # =========================
        # Pyramiding (faster & bigger)
        # =========================
        if self.p.pyramid_enable and self.position and trend_up and adx_ok:
            if self._pyr_stage < len(self.p.pyr_incs):
                if self._bars_since(self._last_add_bar) >= int(self.p.pyr_cooldown_bars):
                    step_px = float(self.p.pyr_step_atr) * atr
                    if (self._last_add_price is not None) and (c >= self._last_add_price + step_px):
                        cur_exp = self._cur_exposure()

                        if self.p.pyr_allow_jump:
                            k = int((c - self._last_add_price) // step_px)
                            k = max(1, k)
                            k = min(k, len(self.p.pyr_incs) - self._pyr_stage)

                            inc_sum = 0.0
                            for j in range(k):
                                inc_sum += float(self.p.pyr_incs[self._pyr_stage + j])

                            new_target = min(float(self.p.max_target), cur_exp + inc_sum)
                            self._submit_target(new_target, f"PYRAMID ADD x{k}: +{inc_sum:.2f}")
                            self._pyr_stage += k
                            self._last_add_bar = len(self)
                            self._last_add_price = c
                            return
                        else:
                            inc = float(self.p.pyr_incs[self._pyr_stage])
                            new_target = min(float(self.p.max_target), cur_exp + inc)
                            self._submit_target(new_target, f"PYRAMID ADD L{self._pyr_stage+1}: +{inc:.2f}")
                            self._pyr_stage += 1
                            self._last_add_bar = len(self)
                            self._last_add_price = c
                            return