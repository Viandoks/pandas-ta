import math
import numpy as np
import pandas as pd


class BotIndicators(object):
    """ A bunch of Technical Indicators to us with Pandas """
    def __init__(self):
         pass

    def averageTrueRange(self, trueRanges, window = 14, fillna=False):
        """ Returns Average True Range in a Pandas serie """
        return self.smoothedMovingAverage(trueRanges, window, fillna)
    def ATR(self, trueRanges=[], window=14, fillna=False):
        """ Short for averageTrueRange() """
        return self.averageTrueRange(trueRanges, window, fillna)

    def directionalMovementIndex(self, highs, lows, closes, window=14, adxWindow=14, fillna = False):
        DMI = pd.DataFrame(columns=['pDI', 'nDI', 'DX', 'ADX'])
        upMove = highs - highs.shift()
        dnMove = lows.shift() - lows
        pDM = closes*0
        nDM = closes*0
        pDM[(upMove>dnMove) & (upMove > 0)] = upMove
        nDM[(dnMove>upMove) & (dnMove > 0)] = dnMove
        TR = self.trueRange(highs, lows, closes)
        ATR = self.smoothedMovingAverage(TR, window, fillna)
        pDI = self.smoothedMovingAverage(pDM)/ATR*100
        nDI = self.smoothedMovingAverage(nDM)/ATR*100
        sum = pDI+nDI
        sum[pDI+nDI==0] = 1
        DX = abs(pDI-nDI)/sum*100
        ADX = self.smoothedMovingAverage(DX, adxWindow-1, fillna)
        DMI['pDI'] = pDI
        DMI['nDI'] = nDI
        DMI['DX'] = DX
        DMI['ADX'] = ADX
        return DMI
    def DMI(self, highs, lows, closes, window=14, adxWindow=14, fillna = False):
        return self.directionalMovementIndex(highs, lows, closes, window, adxWindow, fillna)


    def donchianChannels(self, closes, period=20):
        """ Returns Donchian Channels in two Pandas serie """
        return self.donchianUp(closes, period), self.donchianLow(closes, period)

    def donchianLow(self, closes, period=20):
        """ Returns Lower Donchian Channel in a Pandas serie """
        return closes.rolling(period, min_periods=0).min()

    def donchianUp(self, closes, period=20):
        """ Returns Upper Donchian Channel in a Pandas serie """
        return closes.rolling(period, min_periods=0).max()

    def engulfingPatternBull(self, opens, closes):
        e1 = opens.shift() > closes.shift()
        e2 = opens < closes.shift()
        e3 = closes > opens.shift()
        e4 = opens.shift(2) > closes.shift(2)
        e5 = opens.shift() < closes.shift()
        e6 = opens < closes
        e7 = opens.shift() <= closes.shift(2)
        e8 = opens.shift(2) < closes
        e9 = closes.shift() < closes
        return (e1 & e2 & e3) | (e4 & e5 & e6 & e7 & e8 & e9)

    def exponentialMovingAverage(self, series, window=14, fillna=False):
        """ Returns EMA in a Pandas serie """
        if fillna:
            return series.ewm(span=window, min_periods=0, adjust=False).mean()
        return series.ewm(span=window, min_periods=window, adjust=False).mean()
    def EMA(self, series, window=14, fillna=False):
        """ Short for averageTrueRange() """
        return self.exponentialMovingAverage(series, window, fillna)

    def heikinashi(self, opens, highs, lows, closes):
        """ Returns heiknashi in a Pandas Dataframe """
        heikinashi = pd.DataFrame(columns=['open', 'high', 'low', 'close'])
        heikinashi['open'] = (opens.shift()+closes.shift())/2
        heikinashi['open'].fillna((opens+closes)/2, inplace=True)
        heikinashi['close'] = (opens+highs+lows+closes)/4
        heikinashi['high'] = pd.concat([heikinashi['open'], heikinashi['close'], highs], axis=1).max(axis=1)
        heikinashi['low'] = pd.concat([heikinashi['open'], heikinashi['close'], lows], axis=1).min(axis=1)
        return heikinashi

    def hullMovingAverage(self, series, window=9, fillna=False):
        return self.weightedMovingAverage(2*self.weightedMovingAverage(series, window/2) - self.weightedMovingAverage(series, window), round(math.sqrt(window)))
    def HMA(self, series, window=14, fillna=False):
        return self.hullMovingAverage(series, window, fillna)

    #ichimoku default periods are 9, 26, 26, here default values are adapted to crypto market
    def ichimoku(self, highs, lows, closes, tenkanWindow=10, kijunWindow=30, senkouBWindow=60, displacement=30):
        """ Returns the Ichimoku Cloud in a Pandas Dataframe """
        ichimoku = pd.DataFrame(columns=['tenkan', 'kijun', 'senkouA', 'senkouB', 'displacement'])
        # tenkan
        high = highs.rolling(tenkanWindow, min_periods=0).max()
        low = lows.rolling(tenkanWindow, min_periods=0).max()
        ichimoku['tenkan'] = (high+low)/2
        # kijun
        high = highs.rolling(kijunWindow, min_periods=0).max()
        low = lows.rolling(kijunWindow, min_periods=0).max()
        ichimoku['kijun'] = (high+low)/2
        #senkou A
        ichimoku['senkouA'] = (ichimoku['tenkan']+ichimoku['kijun'])/2
        #senkou B
        high = highs.rolling(senkouBWindow, min_periods=0).max()
        low = lows.rolling(senkouBWindow, min_periods=0).max()
        ichimoku['senkouB'] = (high+low)/2
        # displacement
        ichimoku['displacement'] = closes.shift(displacement)
        return ichimoku

    def MACD(self, series, nslow=26, nfast=12, fillna=False):
        """ Returns MACD in a Pandas dataframe """
        macd = pd.DataFrame(columns=['macd_fast_line', 'macd_slow_line', 'macd'])
        macd['macd_fast_line'] = self.exponentialMovingAverage(series, nfast, fillna)
        macd['macd_slow_line'] = self.exponentialMovingAverage(series, nslow, fillna)
        macd['macd'] = macd['macd_fast_line']-macd['macd_slow_line']
        return macd

    def McGinleyDynamic(self, series, window=14):
        """ Returns McGinley Dynamic in a Pandas series """
        mg = series.copy()
        for i in range(1, len(series)):
            mg.iloc[i] = mg.iloc[i-1]+((series.iloc[i] - mg.iloc[i-1])/(window*((series.iloc[i]/mg.iloc[i-1])**4)))
        return mg

    def momentum(self, series, window=14):
        """ Returns momentum in a Pandas series """
        return series - series.shift(window)

    def RSI (self, series, window=14, fillna=False):
        """ Returns RSI in a Pandas series """
        #TODO: double check this

        diff = series.diff()
        up, down = diff, diff.copy()
        up[up<0] = 0
        down[down>0] = 0
        down = down.abs()

        emaup = up.ewm(span=window-1, min_periods=0).mean()
        emadn = down.ewm(span=window-1, min_periods=0).mean()
        # return 100-(100/(1+(emaup.shift()+up)/(emadn.shift()+down)))
        # print(100 * emaup / (emaup + emadn))
        return 100 * emaup / (emaup + emadn)

    def simpleMovingAverage(self, series, window=14, fillna=False):
        """ Returns SMA in a Pandas serie """
        if fillna:
            return series.rolling(window, min_periods=0).mean()
        else:
            return series.rolling(window, min_periods=window).mean()
    def SMA(self, series, window=14, fillna=False):
        """ Short for simpleMovingAverage() """
        return self.simpleMovingAverage(series, window)


    def smoothedMovingAverage(self, series, window=14, fillna=False):
        if fillna:
            return series.ewm(min_periods=0, alpha=1/window, adjust=False).mean()
        return series.ewm(min_periods=window, alpha=1/window, adjust=False).mean()
    def SMMA(self, series, window=14, fillna=False):
        return smoothedMovingAverage(series, window, fillna)

    def trueRange(self, highs, lows, closes):
        """ Returns True Range in a Pandas serie """
        atr1 = abs(highs - lows)
        atr2 = abs(highs - closes.shift())
        atr3 = abs(lows - closes.shift())
        return pd.concat([atr1, atr2, atr3], axis=1).max(axis=1)
    def TR(self, highs, lows, closes):
        """ Short for trueRange() """
        return self.trueRange(highs, lows, closes)

    def weightedMovingAverage(self, close, window=9, asc=True, fillna=False):
        """ Returns Weighted Moving Average in a Pandas Series """
        window = int(window)
        total_weight = 0.5 * window * (window + 1)
        weights_ = pd.Series(np.arange(1, window + 1))
        weights = weights_ if asc else weights_[::-1]
        def linear(w):
            def _compute(x):
                return (w * x).sum() / total_weight
            return _compute
        if fillna:
            close_ = close.rolling(window, min_periods=0)
        else:
            close_ = close.rolling(window, min_periods=window)
        wma = close_.apply(linear(weights), raw=True)
        return wma

    def williamsFractalBear(self, series, period=2):
        """ Returns Bearish William's Fractal in a Pandas serie """
        a = series.rolling(window=3).min()
        b = series.shift(-period).rolling(window=3).min()
        return a==b

    def williamsFractalBull(self, series, period=2):
        """ Returns Bearish William's Fractal in a Pandas serie """
        a = series.rolling(window=period+1).max()
        b = series.shift(-period).rolling(window=period+1).max()
        return a==b
