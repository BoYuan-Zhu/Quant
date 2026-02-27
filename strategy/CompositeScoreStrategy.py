import backtrader as bt


class CompositeScoreSignal(bt.Indicator):
    """
    Output lines:
      - score: composite score in [-4,4]
      - buy_trig: 1 when score >= buy_th else 0
      - sell_trig: 1 when score <= sell_th else 0
    This indicator is plot-friendly.
    """
    lines = ('score', 'buy_trig', 'sell_trig')
    params = dict(
        # RSI
        rsi_period=14,
        rsi_low=30,
        rsi_high=70,

        # Bollinger
        bb_period=20,
        bb_dev=2.0,

        # Impulse MACD
        ema_period=13,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,

        # TD
        td_lookback=4,
        td_setup_len=9,
        td_shift_days=1,

        # thresholds
        buy_th=2,
        sell_th=-2,
    )

    plotinfo = dict(subplot=True)
    plotlines = dict(
        score=dict(_name='Score'),
        buy_trig=dict(_name='BuyTrig', ls='--'),
        sell_trig=dict(_name='SellTrig', ls='--'),
    )

    def __init__(self):
        # base indicators (bt built-ins)
        self.rsi = bt.ind.RSI(self.data.close, period=self.p.rsi_period)
        self.bb = bt.ind.BollingerBands(self.data.close, period=self.p.bb_period, devfactor=self.p.bb_dev)
        self.ema13 = bt.ind.EMA(self.data.close, period=self.p.ema_period)
        self.macd = bt.ind.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal,
        )

        # TD internal state
        self._td_buy_setup = 0
        self._td_sell_setup = 0
        self._td_buy_fire = []
        self._td_sell_fire = []

        super().__init__()

    # ---- component scores ----
    def _rsi_score(self):
        r = float(self.rsi[0])
        if r < self.p.rsi_low:
            return +1
        if r > self.p.rsi_high:
            return -1
        return 0

    def _bb_score(self):
        c = float(self.data.close[0])
        if c < float(self.bb.bot[0]):
            return +1
        if c > float(self.bb.top[0]):
            return -1
        return 0

    def _macd_hist(self, idx=0):
        # safest: hist = macd - signal (works for all bt versions)
        try:
            return float(self.macd.macd[idx] - self.macd.signal[idx])
        except Exception:
            # fallback by line index if needed
            return float(self.macd.lines[0][idx] - self.macd.lines[1][idx])

    def _impulse_score(self):
        if len(self) < 2:
            return 0
        ema_up = self.ema13[0] > self.ema13[-1]
        ema_dn = self.ema13[0] < self.ema13[-1]
        h0 = self._macd_hist(0)
        h1 = self._macd_hist(-1)
        hist_up = h0 > h1
        hist_dn = h0 < h1
        if ema_up and hist_up:
            return +1
        if ema_dn and hist_dn:
            return -1
        return 0

    def _td_score(self):
        if len(self) <= self.p.td_lookback:
            return 0

        now_bar = len(self) - 1
        c0 = float(self.data.close[0])
        cL = float(self.data.close[-self.p.td_lookback])

        buy_cond = c0 < cL
        sell_cond = c0 > cL

        self._td_buy_setup = min(self._td_buy_setup + 1, self.p.td_setup_len) if buy_cond else 0
        self._td_sell_setup = min(self._td_sell_setup + 1, self.p.td_setup_len) if sell_cond else 0

        if self._td_buy_setup == self.p.td_setup_len:
            self._td_buy_fire.append(now_bar + self.p.td_shift_days)
        if self._td_sell_setup == self.p.td_setup_len:
            self._td_sell_fire.append(now_bar + self.p.td_shift_days)

        score = 0
        if self._td_buy_fire and self._td_buy_fire[0] == now_bar:
            score += 1
            self._td_buy_fire.pop(0)
        if self._td_sell_fire and self._td_sell_fire[0] == now_bar:
            score -= 1
            self._td_sell_fire.pop(0)

        return score

    def next(self):
        s = self._rsi_score() + self._bb_score() + self._impulse_score() + self._td_score()
        self.lines.score[0] = s
        self.lines.buy_trig[0] = 1.0 if s >= self.p.buy_th else 0.0
        self.lines.sell_trig[0] = 1.0 if s <= self.p.sell_th else 0.0


class CompositeScoreStrategy(bt.Strategy):
    """
    Long-only step trading:
      - score >= 2 => +0.25 step (every buy op spaced >= 10 bars)
      - score <= -2 => -0.25 step (every sell op spaced >= 10 bars)
      - stop loss -5% => close all
    """

    params = dict(
        step=0.25,
        min_gap_same_side=10,
        stop_loss_pct=0.05,

        # pass-through to indicator
        rsi_period=14, rsi_low=30, rsi_high=70,
        bb_period=20, bb_dev=2.0,
        ema_period=13, macd_fast=12, macd_slow=26, macd_signal=9,
        td_lookback=4, td_setup_len=9, td_shift_days=1,
        buy_th=2, sell_th=-2,

        pct_eps=0.02,
        printlog=True,
    )

    def __init__(self):
        self.order = None

        # IMPORTANT: you have multiple datas; use data0 as the traded asset
        self.asset = self.datas[0]

        # plot-friendly signal indicator
        self.sig = CompositeScoreSignal(
            self.asset,
            rsi_period=self.p.rsi_period, rsi_low=self.p.rsi_low, rsi_high=self.p.rsi_high,
            bb_period=self.p.bb_period, bb_dev=self.p.bb_dev,
            ema_period=self.p.ema_period, macd_fast=self.p.macd_fast, macd_slow=self.p.macd_slow, macd_signal=self.p.macd_signal,
            td_lookback=self.p.td_lookback, td_setup_len=self.p.td_setup_len, td_shift_days=self.p.td_shift_days,
            buy_th=self.p.buy_th, sell_th=self.p.sell_th,
        )

        # cooldown
        self._last_buy_bar = -10**9
        self._last_sell_bar = -10**9

    def log(self, txt):
        if self.p.printlog:
            dt = self.asset.datetime.date(0).isoformat()
            print(f"{dt} | {txt}")

    def _cur_pct(self):
        value = self.broker.getvalue()
        if value <= 0:
            return 0.0
        pos_value = self.position.size * float(self.asset.close[0])
        return pos_value / value

    def _set_pct(self, target_pct):
        target_pct = max(0.0, min(1.0, target_pct))
        self.order = self.order_target_percent(data=self.asset, target=target_pct)

    def _stop_loss_check(self):
        if not self.position.size:
            return False
        entry = float(self.position.price)
        close = float(self.asset.close[0])
        if close <= entry * (1.0 - self.p.stop_loss_pct):
            self.log(f"STOP LOSS triggered: close={close:.2f} entry={entry:.2f} -> CLOSE ALL")
            self.order = self.close(data=self.asset)
            self._last_sell_bar = len(self) - 1
            return True
        return False

    def next(self):
        if self.order:
            return

        # stop loss first
        if self._stop_loss_check():
            return

        score = int(self.sig.score[0])
        now_bar = len(self) - 1
        cur_pct = self._cur_pct()

        # step buy
        if score >= self.p.buy_th:
            if now_bar - self._last_buy_bar >= self.p.min_gap_same_side:
                if cur_pct < 1.0 - self.p.pct_eps:
                    new_pct = cur_pct + self.p.step
                    self.log(f"BUY-STEP: score={score} pct {cur_pct:.2f}->{min(new_pct,1.0):.2f}")
                    self._set_pct(new_pct)
                    self._last_buy_bar = now_bar
            return

        # step sell
        if score <= self.p.sell_th:
            if now_bar - self._last_sell_bar >= self.p.min_gap_same_side:
                if cur_pct > 0.0 + self.p.pct_eps:
                    new_pct = cur_pct - self.p.step
                    self.log(f"SELL-STEP: score={score} pct {cur_pct:.2f}->{max(new_pct,0.0):.2f}")
                    self._set_pct(new_pct)
                    self._last_sell_bar = now_bar
            return

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            side = "BUY" if order.isbuy() else "SELL"
            self.log(f"ORDER {side} EXECUTED: price={order.executed.price:.2f} "
                     f"size={order.executed.size} value={order.executed.value:.2f} "
                     f"comm={order.executed.comm:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER FAILED: status={order.getstatusname()}")

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f"TRADE CLOSED: pnl={trade.pnl:.2f} pnlcomm={trade.pnlcomm:.2f}")