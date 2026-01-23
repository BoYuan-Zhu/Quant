from __future__ import annotations
import time
from typing import List, Dict, Literal
from datetime import datetime,timedelta

from virtue_account import AShareAccount  # 改成你的文件名，例如 from my_account import AShareAccount


def run_multi_trade_test():
    initial_cash = 1_000_000.0

    acc = AShareAccount(
        initial_cash=initial_cash,
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate_sh=0.0,
        transfer_fee_rate_sz=0.0,
        t_plus_one=False,   # 测试时关闭 T+1
        rounding=2,
    )

    symbol = "600000"
    market = "SH"

    # ---------- 多次买入 ----------
    # B1: 1000 @ 10
    acc.buy(symbol, market, price=10.0, qty=1000, when=datetime(2026, 1, 1, 10), note="B1")

    # B2: 2000 @ 12
    acc.buy(symbol, market, price=12.0, qty=2000, when=datetime(2026, 1, 2, 10), note="B2")

    # ---------- 第一次卖出 ----------
    # S1: 1500 @ 11
    # FIFO:
    #   1000 from 10 → +1000
    #    500 from 12 → -500
    # realized = +500
    acc.sell(symbol, market, price=11.0, qty=1500, when=datetime(2026, 1, 3, 10), note="S1")

    # ---------- 再次买入 ----------
    # B3: 1000 @ 9
    acc.buy(symbol, market, price=9.0, qty=1000, when=datetime(2026, 1, 4, 10), note="B3")

    # ---------- 第二次卖出 ----------
    # S2: 1000 @ 13
    # FIFO: from remaining 12-lot
    # realized = +1000
    acc.sell(symbol, market, price=13.0, qty=1000, when=datetime(2026, 1, 5, 10), note="S2")

    # ---------- 检查当前持仓 ----------
    pos = acc.positions()
    assert pos[symbol] == 1500, f"Expected 1500 shares, got {pos[symbol]}"

    # 剩余仓位：
    #   500 @ 12
    #  1000 @ 9
    # avg cost = (500*12 + 1000*9) / 1500 = 10.0
    avg_cost = acc.get_avg_cost(symbol)
    assert abs(avg_cost - 10.0) < 1e-9, avg_cost

    # ---------- 检查已实现盈亏 ----------
    # S1: +500
    # S2: +1000
    assert acc.realized_pnl == 1500.0, acc.realized_pnl

    # ---------- 浮动盈亏 & 总资产 ----------
    prices = {symbol: 10.5}
    unrealized = acc.unrealized_pnl(prices)   # 1500 * (10.5 - 10.0) = 750
    assert unrealized == 750.0, unrealized

    equity = acc.equity(prices)
    total_pnl = equity - initial_cash
    assert total_pnl == acc.realized_pnl + unrealized

    # ---------- 输出结果 ----------
    print("✅ Multi-trade (1500-lot) test passed")
    print("Positions:", acc.positions())
    print("Avg cost:", acc.get_avg_cost(symbol))
    print("Realized PnL:", acc.realized_pnl)
    print("Unrealized PnL:", unrealized)
    print("Equity:", equity)

    print("\n--- Trades ---")
    for i, t in enumerate(acc.trades_as_dicts(), 1):
        print(f"\n=== Trade #{i} ===")
        for k, v in t.items():
            print(f"{k:>12}: {v}")

def bench(n=1000):
    acc = AShareAccount(
        initial_cash=10_000_000,
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate_sh=0.0,
        transfer_fee_rate_sz=0.0,
        t_plus_one=False,
    )

    symbol, market = "600000", "SH"
    t0 = datetime(2026, 1, 1, 10, 0, 0)

    # 先买足够的仓位，避免中途卖不掉
    acc.buy(symbol, market, price=10.0, qty=100000, when=t0)

    start = time.perf_counter()
    for i in range(n):
        when = t0 + timedelta(minutes=i)
        # 交替卖/买，让 lot 数量保持在可控范围
        acc.sell(symbol, market, price=10.0 + (i % 10) * 0.01, qty=100, when=when)
        acc.buy(symbol, market, price=10.0 + (i % 10) * 0.01, qty=100, when=when)
    end = time.perf_counter()

    print(f"{n*2} ops (sell+buy) took {end-start:.4f}s")
    print("cash:", acc.cash, "pos:", acc.positions()[symbol], "trades:", len(acc.trades))



if __name__ == "__main__":
    run_multi_trade_test()
    # bench(1000)