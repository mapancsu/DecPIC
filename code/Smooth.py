import numpy as np
from _resoltool import lwma_smooth

def lm_smooth(x, peaklist, sl=2): ## Linear Weighted Moving Average, sl:smoothingLevel
    x_s = lwma_smooth(peaklist, sl)
    # plot(peaklist)
    # plot(list(x_s))
    # show()
    return np.array(list(x_s))

def lwma_smooth1(x, peaklist, sl=2): ## Linear Weighted Moving Average, sl:smoothingLevel
    z = list()
    lwmaNormalizationValue = sl + 1
    for i in range(1, sl+1):
        lwmaNormalizationValue += i * 2
    for i in range(0, len(peaklist)):
        sum = 0
        for j in range(-sl, sl+1):
            if i + j < 0 or i + j > len(peaklist) - 1:
                sum += peaklist[i] * (sl - np.abs(j) + 1)
            else:
                sum += peaklist[i + j] * (sl - np.abs(j) + 1)
        smoothedPeakIntensity = sum / lwmaNormalizationValue
        z.append(smoothedPeakIntensity)
    return np.array(z)