import math
import backtrader as bt


class TurtleTradingStrategy(bt.Strategy):
    """
    Turtle Trading (classic):
      - Entry: breakout of Donchian channel (highest high / lowest low)
      - Exit: breakout of shorter Donchian channel in opposite direction
      - Position sizing: risk per trade = equity * risk_pct, based on ATR (N)
      - Stop loss: 2*N against entry
      - Pyramiding: add every 0.5*N in favorable move, up to max_units
    """

    params = dict(
        # ===== Donchian channels =====
        entry_period=20,   # System 1: 20  (System 2: 55)
        exit_period=10,    # System 1: 10  (System 2: 20)
        trade_long=True,
        trade_short=False,  # A股/现货一般不开空

        # ===== ATR (N) =====
        atr_period=20,     # classic turtle uses 20-day ATR
        stop_atr_mult=2.0, # stop = entry - 2N (long)

        # ===== Risk / sizing =====
        risk_pct=0.01,     # risk per trade = 1% of equity
        unit_atr_mult=1.0, # unit size uses N * unit_atr_mult
        max_units=4,       # pyramiding units

        # ===== Pyramiding =====
        add_atr_step=0.5,  # add every +0.5N
        allow_pyramid=True,

        # ===== Execution / safety =====
        cash_buffer=0.02,       # keep some cash when increasing exposure
        min_bars_between_orders=1,
        debug_reject=True,
        printlog=False,
    )

    def __init__(self):
        d = self.datas[0]
        self.data0 = d
        self.order = None
        self._last_order_bar = -10**9

        # Donchian channels (use highest/lowest; we will reference [-1] to avoid same-bar lookahead)
        self.entry_high = bt.ind.Highest(d.high, period=int(self.p.entry_period))
        self.entry_low = bt.ind.Lowest(d.low, period=int(self.p.entry_period))
        self.exit_high = bt.ind.Highest(d.high, period=int(self.p.exit_period))
        self.exit_low = bt.ind.Lowest(d.low, period=int(self.p.exit_period))

        # ATR (N)
        self.atr = bt.ind.ATR(d, period=int(self.p.atr_period))

        # Turtle state
        self._units = 0               # how many units currently held (long side)
        self._last_add_price = float("nan")
        self._entry_price = float("nan")
        self._stop_price = float("nan")

    # ---------- helpers ----------
    def _bars_since(self, bar_index: int) -> int:
        return int(len(self) - bar_index)

    def _can_order(self) -> bool:
        return self._bars_since(self._last_order_bar) >= int(self.p.min_bars_between_orders)

    def _equity(self) -> float:
        return float(self.broker.getvalue())

    def _unit_size_shares(self) -> int:
        """
        Classic turtle: Unit = (Equity * risk_pct) / (N * dollar_per_point)
        Here dollar_per_point ~ price per share, so shares = risk_cash / (N)
        """
        N = float(self.atr[0])
        if math.isnan(N) or N <= 0:
            return 0
        risk_cash = self._equity() * float(self.p.risk_pct)
        shares = int(risk_cash / (N * float(self.p.unit_atr_mult)))
        return max(shares, 0)

    def _place_buy(self, size: int, reason: str):
        if size <= 0:
            return
        if not self._can_order():
            return
        o = self.buy(data=self.data0, size=size)
        self.order = o
        if self.p.printlog:
            dt = self.data0.datetime.date(0).isoformat()
            print(dt, "BUY", size, reason)

    def _place_sell(self, size: int, reason: str):
        if size <= 0:
            return
        if not self._can_order():
            return
        o = self.sell(data=self.data0, size=size)
        self.order = o
        if self.p.printlog:
            dt = self.data0.datetime.date(0).isoformat()
            print(dt, "SELL", size, reason)

    def _close_long(self, reason: str):
        if not self.position:
            return
        if not self._can_order():
            return
        o = self.close(data=self.data0)
        self.order = o
        if self.p.printlog:
            dt = self.data0.datetime.date(0).isoformat()
            print(dt, "CLOSE", reason)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        dt = self.data0.datetime.date(0).isoformat()

        if order.status in [order.Completed]:
            self._last_order_bar = len(self)

            # update turtle state after fills
            if order.isbuy():
                fill = float(order.executed.price)
                if math.isnan(self._entry_price):
                    self._entry_price = fill
                self._last_add_price = fill
                self._units += 1

                N = float(self.atr[0])
                if (not math.isnan(N)) and N > 0:
                    # stop based on FIRST entry (classic uses last entry sometimes; we use first for simplicity)
                    base = self._entry_price if not math.isnan(self._entry_price) else fill
                    self._stop_price = base - float(self.p.stop_atr_mult) * N

            else:
                # sell completion (if we are using partial sells)
                pass

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.p.debug_reject:
                print(dt, "ORDER", order.getstatusname(),
                      "isbuy=", order.isbuy(),
                      "created_size=", order.created.size,
                      "price=", order.created.price)

        self.order = None

    # ---------- main ----------
    def next(self):
        if self.order:
            return

        # need enough bars
        if len(self) <= max(int(self.p.entry_period), int(self.p.exit_period), int(self.p.atr_period)):
            return

        close = float(self.data0.close[0])
        hi_prev = float(self.entry_high[-1])
        lo_prev = float(self.entry_low[-1])
        ex_lo_prev = float(self.exit_low[-1])
        N = float(self.atr[0])

        if any(map(math.isnan, [close, hi_prev, lo_prev, ex_lo_prev, N])) or N <= 0:
            return

        # ========== LONG SIDE ==========
        if self.p.trade_long:
            # 1) Exit conditions (risk-first)
            if self.position:
                # Stop loss: close below stop
                if (not math.isnan(self._stop_price)) and close <= float(self._stop_price):
                    self._close_long(f"STOP: close={close:.2f} <= stop={self._stop_price:.2f}")
                    self._reset_state()
                    return

                # Donchian exit: close breaks below exit channel low
                if close < ex_lo_prev:
                    self._close_long(f"EXIT: close={close:.2f} < DonchianLow({self.p.exit_period})={ex_lo_prev:.2f}")
                    self._reset_state()
                    return

            # 2) Entry
            if not self.position:
                # breakout above prior entry high
                if close > hi_prev:
                    unit_shares = self._unit_size_shares()
                    if unit_shares > 0:
                        self._place_buy(unit_shares, f"ENTRY: close={close:.2f} > DonchianHigh({self.p.entry_period})={hi_prev:.2f}")
                        # set entry/stop tentatively; final price updated in notify_order
                        self._entry_price = close
                        self._last_add_price = close
                        self._units = 0
                        self._stop_price = close - float(self.p.stop_atr_mult) * N
                    return

            # 3) Pyramiding (add)
            if self.position and self.p.allow_pyramid:
                if self._units < int(self.p.max_units):
                    step = float(self.p.add_atr_step) * N
                    if (not math.isnan(self._last_add_price)) and close >= (self._last_add_price + step):
                        unit_shares = self._unit_size_shares()
                        if unit_shares > 0:
                            self._place_buy(unit_shares, f"PYRAMID: +{self.p.add_atr_step}N (close={close:.2f})")
                        return

    def _reset_state(self):
        self._units = 0
        self._last_add_price = float("nan")
        self._entry_price = float("nan")
        self._stop_price = float("nan")