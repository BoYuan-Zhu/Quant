import backtrader as bt

class MACDGoldenCross(bt.Strategy):
    params = dict(comm=0.0005, buffer=0.995, lot=100)  # 手续费/预留/一手

    def __init__(self):
        self.cross = bt.ind.CrossOver(self.data.dif, self.data.dea)

    def _full_size(self, price):
        cash = self.broker.getcash() * self.p.buffer
        size = int(cash / (price * (1 + self.p.comm)))
        if self.p.lot > 1:
            size = (size // self.p.lot) * self.p.lot
        return max(size, 0)

    def next(self):
        if not self.position:
            if self.cross[0] > 0 and self.data.hist[0] > 0:
                size = self._full_size(float(self.data.close[0]))
                if size > 0:
                    self.buy(size=size)
        else:
            if self.cross[0] < 0:
                self.close()



class PandasMACDData(bt.feeds.PandasData):
    lines = ('dif', 'dea', 'hist',)

    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low',  'low'),
        ('close','close'),
        ('openinterest', -1),

        ('dif',  'MACD_DIF_12_26_close'),
        ('dea',  'MACD_DEA_9_12_26_close'),
        ('hist', 'MACD_HIST_12_26_9_close'),
    )
