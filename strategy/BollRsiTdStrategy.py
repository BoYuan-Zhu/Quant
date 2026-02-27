import backtrader as bt
import math


class BollRsiTdStepStrategy(bt.Strategy):
    params = dict(
        # ===== indicators =====
        bb_period=20,
        rsi_period=14,

        td_lookback=4,
        td_setup_len=5,
        td_shift_days=1,

        # ===== BB ladder (independent for buy / sell) =====
        # Each level: (sigma, rsi_threshold, step)
        # Buy: close cross BELOW lower band AND RSI < threshold -> +step
        buy_levels=(
            (1.5, 45, 0.1),
            (2.0, 40, 0.25),
            (2.5, 35, 0.5),
            (3.0, 30, 1.00),
        ),
        # Sell: close cross ABOVE upper band AND RSI > threshold -> -step
        sell_levels=(
            (2.5, 60, 0.05),
            (3.0, 65, 0.15),
            (3.5, 70, 0.50),
            (4.0, 75, 1.00),
        ),

        # ===== TD steps (ABSOLUTE position delta) =====
        td_buy_step=0.15,   # td_buy -> +0.15
        td_sell_step=0.10,  # td_sell -> -0.10

        # ===== cooldown (same-side) =====
        min_gap_same_side=5,

        # ===== stop loss =====
        stop_loss_pct=0.1,

        # ===== execution =====
        lot=100,
        comm_buffer=0.001,

        # allow short by setting min_pos negative
        min_pos=-1.0,
        max_pos=+1.0,
        allow_short=True,   # if False: sells won't go beyond existing long size

        printlog=True,
    )

    def __init__(self):
        self.data0 = self.datas[0]
        self.order = None

        self.rsi = bt.ind.RSI(self.data0.close, period=self.p.rsi_period)

        # Build BB indicators for all sigmas used in buy/sell ladders
        sigmas = set()
        for s, _, _ in self.p.buy_levels:
            sigmas.add(float(s))
        for s, _, _ in self.p.sell_levels:
            sigmas.add(float(s))

        self._bb = {}
        for s in sorted(sigmas):
            self._bb[s] = bt.ind.BollingerBands(self.data0.close, period=self.p.bb_period, devfactor=float(s))

        # TD internal state (same as your original simplified TD)
        self._td_buy_setup = 0
        self._td_sell_setup = 0
        self._td_buy_fire = []
        self._td_sell_fire = []

        # cooldown tracked by EXECUTED bar index
        self._last_buy_bar = -10**9
        self._last_sell_bar = -10**9

    def log(self, txt):
        if self.p.printlog:
            dt = self.data0.datetime.date(0).isoformat()
            print(f"{dt} | {txt}")

    # -------------------------
    # TD (simplified): returns one-shot td_buy/td_sell
    # -------------------------
    def _td_signals_today(self):
        if len(self) <= self.p.td_lookback:
            return 0, 0

        now_bar = len(self) - 1
        c0 = float(self.data0.close[0])
        cL = float(self.data0.close[-self.p.td_lookback])

        buy_cond = c0 < cL   # relative weakness
        sell_cond = c0 > cL  # relative strength

        self._td_buy_setup = min(self._td_buy_setup + 1, self.p.td_setup_len) if buy_cond else 0
        self._td_sell_setup = min(self._td_sell_setup + 1, self.p.td_setup_len) if sell_cond else 0

        if self._td_buy_setup == self.p.td_setup_len:
            self._td_buy_fire.append(now_bar + int(self.p.td_shift_days))
        if self._td_sell_setup == self.p.td_setup_len:
            self._td_sell_fire.append(now_bar + int(self.p.td_shift_days))

        td_buy = 1 if (self._td_buy_fire and self._td_buy_fire[0] == now_bar) else 0
        td_sell = 1 if (self._td_sell_fire and self._td_sell_fire[0] == now_bar) else 0

        if td_buy:
            self._td_buy_fire.pop(0)
        if td_sell:
            self._td_sell_fire.pop(0)

        return td_buy, td_sell

    # -------------------------
    # Position helpers (SIGNED, supports short)
    # -------------------------
    def _round_to_lot(self, size_abs: int) -> int:
        lot = int(self.p.lot)
        if lot <= 1:
            return int(size_abs)
        return (int(size_abs) // lot) * lot

    def _pos_value_signed(self) -> float:
        pos = self.getposition(self.data0)
        price = float(self.data0.close[0])
        if not (price > 0 and math.isfinite(price)):
            return 0.0
        return float(pos.size) * price  # can be negative

    def _pos_pct_signed(self) -> float:
        eq = float(self.broker.getvalue())
        if not (eq > 0 and math.isfinite(eq)):
            return 0.0
        pct = self._pos_value_signed() / eq
        return max(float(self.p.min_pos), min(pct, float(self.p.max_pos)))

    def _delta_size_to_target_pct(self, target_pct: float) -> int:
        """
        Convert target_pct (can be negative) to share delta:
          +delta -> buy
          -delta -> sell
        """
        price = float(self.data0.close[0])
        eq = float(self.broker.getvalue())
        if not (price > 0 and eq > 0 and math.isfinite(price) and math.isfinite(eq)):
            return 0

        target_pct = max(float(self.p.min_pos), min(float(target_pct), float(self.p.max_pos)))
        target_value = target_pct * eq

        pos = self.getposition(self.data0)
        cur_value = float(pos.size) * price
        delta_value = target_value - cur_value

        # Need to BUY (increase position, or cover shorts)
        if delta_value > 0:
            cash_avail = float(self.broker.getcash())
            spend = min(delta_value, cash_avail) * (1.0 - float(self.p.comm_buffer))
            if spend <= 0:
                return 0
            raw = int(spend / price)
            size = self._round_to_lot(raw)
            return max(size, 0)

        # Need to SELL (reduce long or increase short)
        if delta_value < 0:
            raw = int((-delta_value) / price)
            size = self._round_to_lot(raw)
            if size <= 0:
                return 0

            if not bool(self.p.allow_short):
                # long-only: cap sell by current long size
                size = min(size, max(int(pos.size), 0))
                if size <= 0:
                    return 0

            return -size

        return 0

    # -------------------------
    # Stop loss (long/short symmetric)
    # -------------------------
    def _stop_loss_check(self) -> bool:
        pos = self.getposition(self.data0)
        if pos.size == 0:
            return False

        entry = float(pos.price)     # avg entry
        close = float(self.data0.close[0])
        if not (entry > 0 and math.isfinite(entry) and math.isfinite(close)):
            return False

        sl = float(self.p.stop_loss_pct)

        if pos.size > 0:
            stop_px = entry * (1.0 - sl)
            if close <= stop_px:
                self.log(f"STOP LOSS LONG -> CLOSE (close={close:.2f} <= stop={stop_px:.2f})")
                self.order = self.close(data=self.data0)
                return True
        else:
            stop_px = entry * (1.0 + sl)
            if close >= stop_px:
                self.log(f"STOP LOSS SHORT -> CLOSE (close={close:.2f} >= stop={stop_px:.2f})")
                self.order = self.close(data=self.data0)
                return True

        return False

    # -------------------------
    # BB ladder pickers (choose deepest sigma triggered)
    # -------------------------
    def _pick_buy_level(self, c0: float, c1: float, r0: float):
        best = None  # (sigma, step)
        for sigma, rsi_th, step in self.p.buy_levels:
            sigma = float(sigma)
            rsi_th = float(rsi_th)
            step = float(step)
            bb = self._bb[sigma]
            cross_below = (c1 >= float(bb.bot[-1])) and (c0 < float(bb.bot[0]))
            if cross_below and (r0 < rsi_th):
                if best is None or sigma > best[0]:
                    best = (sigma, step)
        return best

    def _pick_sell_level(self, c0: float, c1: float, r0: float):
        best = None  # (sigma, step)
        for sigma, rsi_th, step in self.p.sell_levels:
            sigma = float(sigma)
            rsi_th = float(rsi_th)
            step = float(step)
            bb = self._bb[sigma]
            cross_above = (c1 <= float(bb.top[-1])) and (c0 > float(bb.top[0]))
            if cross_above and (r0 > rsi_th):
                if best is None or sigma > best[0]:
                    best = (sigma, step)
        return best

    # -------------------------
    # Core trading logic (TD is ABSOLUTE delta like before)
    # -------------------------
    def next(self):
        if self.order:
            return

        if self._stop_loss_check():
            return

        min_need = max(self.p.bb_period, self.p.rsi_period) + 2
        if len(self) < min_need:
            return

        now_bar = len(self) - 1
        c0 = float(self.data0.close[0])
        c1 = float(self.data0.close[-1])
        r0 = float(self.rsi[0])

        cur_pct = self._pos_pct_signed()
        delta = 0.0
        reasons = []

        # BB ladder
        buy_pick = self._pick_buy_level(c0, c1, r0)
        if buy_pick is not None:
            sigma, step = buy_pick
            delta += float(step)
            reasons.append(f"BB_BUY_{sigma:g}σ(+{step:g})")

        sell_pick = self._pick_sell_level(c0, c1, r0)
        if sell_pick is not None:
            sigma, step = sell_pick
            delta -= float(step)
            reasons.append(f"BB_SELL_{sigma:g}σ(-{step:g})")

        # TD absolute delta (same as before)
        td_buy, td_sell = self._td_signals_today()
        if td_buy:
            delta += float(self.p.td_buy_step)
            reasons.append(f"TD_BUY(+{float(self.p.td_buy_step):g})")
        if td_sell:
            delta -= float(self.p.td_sell_step)
            reasons.append(f"TD_SELL(-{float(self.p.td_sell_step):g})")

        if abs(delta) <= 1e-12:
            return

        target_pct = max(float(self.p.min_pos), min(cur_pct + delta, float(self.p.max_pos)))
        if abs(target_pct - cur_pct) <= 1e-6:
            return

        delta_size = self._delta_size_to_target_pct(target_pct)
        if delta_size == 0:
            return

        # cooldown
        gap = int(self.p.min_gap_same_side)
        if delta_size > 0:
            if now_bar - self._last_buy_bar < gap:
                return
        else:
            if now_bar - self._last_sell_bar < gap:
                return

        reason = "+".join(reasons)
        if delta_size > 0:
            self.log(f"{reason} | BUY cur={cur_pct:+.2f} -> tgt={target_pct:+.2f} size={delta_size}")
            self.order = self.buy(data=self.data0, size=delta_size)
        else:
            self.log(f"{reason} | SELL cur={cur_pct:+.2f} -> tgt={target_pct:+.2f} size={-delta_size}")
            self.order = self.sell(data=self.data0, size=-delta_size)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            now_bar = len(self) - 1
            if order.isbuy():
                self._last_buy_bar = now_bar
                self.log(f"BUY EXEC price={order.executed.price:.2f} size={order.executed.size}")
            else:
                self._last_sell_bar = now_bar
                self.log(f"SELL EXEC price={order.executed.price:.2f} size={order.executed.size}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER FAILED: {order.getstatusname()}")

        self.order = None