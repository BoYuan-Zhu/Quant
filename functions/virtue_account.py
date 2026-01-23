from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Literal
from datetime import datetime, date


Side = Literal["BUY", "SELL"]
Market = Literal["SH", "SZ"]  # Shanghai / Shenzhen


@dataclass
class Trade:
    time: str
    symbol: str
    market: Market
    side: Side
    price: float
    qty: int
    amount: float            # price * qty
    commission: float
    stamp_tax: float
    transfer_fee: float
    total_fee: float         # commission + stamp_tax + transfer_fee
    cash_change: float       # negative for buy, positive for sell
    realized_pnl: float      # only for SELL; 0 for BUY
    note: str = ""


@dataclass
class PositionLot:
    """A FIFO lot for cost basis."""
    qty: int
    cost_price: float        # per-share cost price (including buy-side fees allocated)


class AShareAccount:
    """
    Simulate an A-share stock account with:
    - cash / frozen cash (optional)
    - positions with FIFO lots for realized PnL
    - A-share fees: commission (min), stamp tax (sell only), transfer fee (config)
    - T+1 sell availability (optional)
    - trade ledger and summary metrics
    """

    def __init__(
        self,
        initial_cash: float,
        commission_rate: float = 0.0003,   # 0.03%
        min_commission: float = 5.0,
        stamp_tax_rate: float = 0.001,     # 0.1% (1‰) sell only; adjust if policy changes
        transfer_fee_rate_sh: float = 0.0,  # placeholder; widely varies, tune as needed
        transfer_fee_rate_sz: float = 0.0,
        t_plus_one: bool = True,
        rounding: int = 2,
    ):
        if initial_cash < 0:
            raise ValueError("initial_cash must be non-negative")

        self.cash: float = float(initial_cash)
        self.rounding = rounding

        # Fee config
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.transfer_fee_rate_sh = transfer_fee_rate_sh
        self.transfer_fee_rate_sz = transfer_fee_rate_sz

        # Trading rules
        self.t_plus_one = t_plus_one

        # Positions: symbol -> list of FIFO lots
        self._lots: Dict[str, List[PositionLot]] = {}
        # For T+1: record bought qty per day, to restrict selling same day
        self._today_buys: Dict[tuple[str, date], int] = {}

        # Ledger
        self.trades: List[Trade] = []

        # Realized PnL
        self.realized_pnl: float = 0.0

    # ---------- Public APIs ----------

    def buy(
        self,
        symbol: str,
        market: Market,
        price: float,
        qty: int,
        when: Optional[datetime] = None,
        note: str = "",
    ) -> Trade:
        self._validate_order(price, qty)
        when = when or datetime.now()

        amount = price * qty
        commission = self._calc_commission(amount)
        stamp_tax = 0.0  # buy side no stamp tax
        transfer_fee = self._calc_transfer_fee(market, amount)
        total_fee = commission + stamp_tax + transfer_fee

        cash_needed = amount + total_fee
        cash_needed = self._r(cash_needed)

        if self.cash < cash_needed - 1e-9:
            raise ValueError(f"Insufficient cash: need {cash_needed}, have {self.cash}")

        # Deduct cash
        self.cash = self._r(self.cash - cash_needed)

        # Allocate fees into cost basis (per-share cost)
        cost_price = (amount + total_fee) / qty
        self._lots.setdefault(symbol, []).append(PositionLot(qty=qty, cost_price=cost_price))

        # T+1 bookkeeping
        d = when.date()
        self._today_buys[(symbol, d)] = self._today_buys.get((symbol, d), 0) + qty

        trade = Trade(
            time=when.isoformat(timespec="seconds"),
            symbol=symbol,
            market=market,
            side="BUY",
            price=price,
            qty=qty,
            amount=self._r(amount),
            commission=self._r(commission),
            stamp_tax=self._r(stamp_tax),
            transfer_fee=self._r(transfer_fee),
            total_fee=self._r(total_fee),
            cash_change=self._r(-cash_needed),
            realized_pnl=0.0,
            note=note,
        )
        self.trades.append(trade)
        return trade

    def sell(
        self,
        symbol: str,
        market: Market,
        price: float,
        qty: int,
        when: Optional[datetime] = None,
        note: str = "",
    ) -> Trade:
        self._validate_order(price, qty)
        when = when or datetime.now()

        available = self.get_sellable_qty(symbol, when.date())
        if qty > available:
            raise ValueError(f"Not enough sellable shares: sell {qty}, sellable {available}")

        amount = price * qty
        commission = self._calc_commission(amount)
        stamp_tax = amount * self.stamp_tax_rate  # sell only
        transfer_fee = self._calc_transfer_fee(market, amount)
        total_fee = commission + stamp_tax + transfer_fee

        proceeds = amount - total_fee
        proceeds = self._r(proceeds)

        # Realized PnL via FIFO lots
        realized = self._consume_fifo(symbol, qty, sell_price=price)

        # Cash increases by net proceeds
        self.cash = self._r(self.cash + proceeds)

        trade = Trade(
            time=when.isoformat(timespec="seconds"),
            symbol=symbol,
            market=market,
            side="SELL",
            price=price,
            qty=qty,
            amount=self._r(amount),
            commission=self._r(commission),
            stamp_tax=self._r(stamp_tax),
            transfer_fee=self._r(transfer_fee),
            total_fee=self._r(total_fee),
            cash_change=self._r(proceeds),
            realized_pnl=self._r(realized),
            note=note,
        )
        self.trades.append(trade)
        self.realized_pnl = self._r(self.realized_pnl + realized)
        return trade

    def deposit(self, amount: float, note: str = "") -> None:
        if amount <= 0:
            raise ValueError("deposit amount must be positive")
        self.cash = self._r(self.cash + amount)
        # Optional: record as a trade-like entry if you want bookkeeping

    def withdraw(self, amount: float, note: str = "") -> None:
        if amount <= 0:
            raise ValueError("withdraw amount must be positive")
        if self.cash < amount - 1e-9:
            raise ValueError("insufficient cash")
        self.cash = self._r(self.cash - amount)

    def positions(self) -> Dict[str, int]:
        """Total share count per symbol."""
        out: Dict[str, int] = {}
        for sym, lots in self._lots.items():
            out[sym] = sum(l.qty for l in lots)
        return out

    def get_position_cost(self, symbol: str) -> float:
        """Total cost basis (including buy-side fees) of remaining position."""
        lots = self._lots.get(symbol, [])
        return self._r(sum(l.qty * l.cost_price for l in lots))

    def get_avg_cost(self, symbol: str) -> float:
        """Weighted average cost per share of remaining position."""
        lots = self._lots.get(symbol, [])
        total_qty = sum(l.qty for l in lots)
        if total_qty == 0:
            return 0.0
        total_cost = sum(l.qty * l.cost_price for l in lots)
        return self._r(total_cost / total_qty)

    def get_sellable_qty(self, symbol: str, on_date: Optional[date] = None) -> int:
        """
        Sellable quantity under T+1:
        - if t_plus_one: today's bought shares are not sellable today.
        - otherwise: all shares are sellable.
        """
        total = self.positions().get(symbol, 0)
        if not self.t_plus_one:
            return total

        on_date = on_date or datetime.now().date()
        today_buy = self._today_buys.get((symbol, on_date), 0)
        sellable = total - today_buy
        return max(0, sellable)

    def equity(self, prices: Dict[str, float]) -> float:
        """Cash + market value of positions using provided prices."""
        mv = 0.0
        for sym, qty in self.positions().items():
            p = prices.get(sym)
            if p is None:
                raise ValueError(f"Missing price for {sym}")
            mv += p * qty
        return self._r(self.cash + mv)

    def unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """(Market value - cost basis) of current holdings."""
        pnl = 0.0
        for sym, lots in self._lots.items():
            qty = sum(l.qty for l in lots)
            if qty == 0:
                continue
            p = prices.get(sym)
            if p is None:
                raise ValueError(f"Missing price for {sym}")
            mv = p * qty
            cost = sum(l.qty * l.cost_price for l in lots)
            pnl += (mv - cost)
        return self._r(pnl)

    def summary(self, prices: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Basic account summary.
        If prices provided, include equity & unrealized pnl.
        """
        s = {
            "cash": self._r(self.cash),
            "realized_pnl": self._r(self.realized_pnl),
        }
        if prices is not None:
            s["equity"] = self.equity(prices)
            s["unrealized_pnl"] = self.unrealized_pnl(prices)
        return s

    def trades_as_dicts(self) -> List[dict]:
        return [asdict(t) for t in self.trades]

    # ---------- Internals ----------

    def _validate_order(self, price: float, qty: int) -> None:
        if price <= 0:
            raise ValueError("price must be positive")
        if qty <= 0:
            raise ValueError("qty must be positive")
        if qty % 100 != 0:
            # A股买入通常一手=100股（卖出可不满一手取决于规则/碎股），这里保守限制
            raise ValueError("qty must be a multiple of 100 for A-share lots")

    def _calc_commission(self, amount: float) -> float:
        c = amount * self.commission_rate
        if c < self.min_commission:
            c = self.min_commission
        return c

    def _calc_transfer_fee(self, market: Market, amount: float) -> float:
        if market == "SH":
            return amount * self.transfer_fee_rate_sh
        if market == "SZ":
            return amount * self.transfer_fee_rate_sz
        raise ValueError("market must be 'SH' or 'SZ'")

    def _consume_fifo(self, symbol: str, sell_qty: int, sell_price: float) -> float:
        """
        Consume FIFO lots, return realized PnL.
        Realized PnL is computed using per-share cost_price (already includes buy-side fees).
        Note: sell-side fees are NOT included here; they already reduce proceeds.
        """
        lots = self._lots.get(symbol, [])
        if not lots:
            raise ValueError(f"No position to sell for {symbol}")

        remain = sell_qty
        realized = 0.0

        while remain > 0:
            if not lots:
                raise ValueError(f"Position lots insufficient for {symbol}")

            lot = lots[0]
            take = min(lot.qty, remain)

            realized += take * (sell_price - lot.cost_price)

            lot.qty -= take
            remain -= take

            if lot.qty == 0:
                lots.pop(0)

        # Clean empty
        if not lots:
            self._lots.pop(symbol, None)

        return realized

    def _r(self, x: float) -> float:
        return round(float(x), self.rounding)