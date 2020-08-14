
import numpy as np
from pylab import plot, show, xlabel, ylabel, legend, boxplot
from scipy.optimize import curve_fit
import scipy.special as sse
from CIC_extraction import Findpeaks, gethidepeakpoint
from scipy.optimize import differential_evolution

def simulate_data1(noiselevel=0.06):
    xdata = np.arange(100)

    A = [500, 1000, 350]
    xc = [18, 38, 65]
    w = [6, 5, 4]
    t0 = [8, 12, 6]
    ydatas = []
    for i in range(3):
        ydatas.append(emg(xdata, A[i], t0[i], w[i], xc[i]))
    noise = np.random.random(len(xdata))*np.max(ydatas[0]+ydatas[1]+ydatas[2])*noiselevel
    ydata2 = ydatas[0]+ydatas[1]+ydatas[2] + noise
    plot(xdata, ydatas[0], '--', label="c21")
    plot(xdata, ydatas[1], '--', label="c22")
    plot(xdata, ydatas[2], '--', label="c23")
    plot(xdata, noise, 'c--', label="Noise")
    plot(xdata, ydata2, 'k', label="Mixture")
    xlabel("Scan")
    ylabel("Intensity")
    legend()
    show()

    ydata1 = ydatas[0]+ydatas[1]+ydatas[2] + np.random.random(len(xdata))*np.max(ydatas[0]+ydatas[1]+ydatas[2])*noiselevel

    return ydata1, ydatas[0],ydatas[1],ydatas[2]

def simulate_data2(noiselevel=2.0):
    xdata = np.arange(100)

    A = [500, 1000, 350]
    xc = [22, 38, 55]
    w = [6, 5, 4]
    t0 = [8, 12, 6]
    ydatas = []
    for i in range(3):
        ydatas.append(emg(xdata, A[i], t0[i], w[i], xc[i]))
    
    noise = np.random.random(len(xdata))*np.max(ydatas[0]+ydatas[1]+ydatas[2])*noiselevel
    ydata2 = ydatas[0]+ydatas[1]+ydatas[2] + noise

    # plot(xdata, ydatas[0], 'r--', label="c21")
    # plot(xdata, ydatas[1], 'g--', label="c22")
    # plot(xdata, ydatas[2], 'b--', label="c23")
    plot(xdata, ydatas[0], '--', label="c21")
    plot(xdata, ydatas[1], '--', label="c22")
    plot(xdata, ydatas[2], '--', label="c23")
    plot(xdata, noise, 'c--', label="Noise")
    plot(xdata, ydata2, 'k', label="Mixture")
    xlabel("Scan")
    ylabel("Intensity")
    legend()
    show()

    
    # plot(ydata2)
    # show()
    return ydata2, ydatas[0],ydatas[1],ydatas[2]

def emg(x, A, t0, w, xc):  # exponential gaussian
    return (A / t0) * np.exp(0.5 * (w / t0) ** 2 - (x - xc) / t0) * (0.5 + 0.5 * sse.erf(((x - xc) / w - w / t0) / 2 ** 0.5))

def gauss(x, A, xc, w):
    return A*np.exp(-0.5*(x-xc)**2/w**2)

def emgfit(x, *param):
    group = int(len(param)/4)
    fy = 0
    for i in range(group):
        fy += emg(x, param[4*i+0], param[4*i+1], param[4*i+2], param[4*i+3])
    return fy

def gaussfit(x, *param):
    group = int(len(param)/3)
    fy = 0
    for i in range(group):
        fy += gauss(x, param[3*i+0], param[3*i+1], param[3*i+2])
    return fy

def EMGModel(ydata, p0):
    xdata = np.arange(len(ydata))
    popt, pcov = curve_fit(emgfit, xdata, ydata, p0=p0)
    fity = []
    for i in range(0, int(len(popt)/4)):
        y = emg(xdata, popt[4*i+0], popt[4*i+1], popt[4*i+2], popt[4*i+3])
        fity.append(y)
    print(popt)
    return fity

def GuassModel(ydata, p0):
    xdata = np.arange(len(ydata))
    popt, pcov = curve_fit(gaussfit, xdata, ydata, p0=p0)
    fity = []
    for i in range(0, int(len(popt)/3)):
        y = gauss(xdata, popt[3*i+0], popt[3*i+1], popt[3*i+2])
        fity.append(y)
    print(popt)
    return fity

def deemgfit(param, *data):
    group = int(len(param) / 4)
    x, y = data
    fy = 0
    for i in range(group):
        yy = emg(x, param[4 * i + 0], param[4 * i + 1], param[4 * i + 2], param[4 * i + 3])
        yy[np.where(np.isnan(yy))[0]] = 0.0
        fy += yy
    return np.linalg.norm(fy-y) #np.sum((fy-y)**2)#

def deemgfit1(param, *data):
    group = int(param[0])
    p = param[1:]
    x, y = data
    fy = 0
    for i in range(group):
        fy += emg(x, p[4 * i + 0], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3])
    return np.linalg.norm(fy-y)

def DEEMGModel(ydata, p0):
    xdata = np.arange(len(ydata))

    area = np.sum(ydata)
    width = len(ydata)

    N = 3
    bounds = []
    for i in range(N):
        bounds.append([0.01, area])
        bounds.append([1, width-1])
        bounds.append([1, 10])
        bounds.append([1, width-1])

    #res = differential_evolution(deemgfit, bounds, args=(xdata, ydata))
    res = differential_evolution(deemgfit, bounds, args=(xdata, ydata), tol=0.00001, init="latinhypercube")
    #res = differential_evolution(deemgfit, bounds, args=(xdata, ydata), tol=0.5,strategy='randtobest1exp') ##, init="random"
    
    # y1 = emg(xdata, res.x[0], res.x[1], res.x[2], res.x[3])
    # y2 = emg(xdata, res.x[4], res.x[5], res.x[6], res.x[7])
    # y3 = emg(xdata, res.x[8], res.x[9], res.x[10], res.x[11])
    # y4 = emg(xdata, res.x[12], res.x[13], res.x[14], res.x[15])
    
    # plot(xdata, ydata, label="Origin data")
    # plot(xdata, y1, label="Com1")
    # plot(xdata, y2, "r", label="Com2")
    # plot(xdata, y3, "y", label="Com3")
    # # plot(xdata, y4, "c", label="Com4")
    # # plot(xdata, y1+y2+y3+y4, "g--", linewidth=2, label="Fitting data")
    # plot(xdata, y1+y2+y3, "g--", linewidth=2, label="Fitting data")
    # xlabel("Rt (s)")
    # ylabel("Intensity")
    # legend()
    
    # plot(rt, ydata,label="Origin data")
    # plot(rt, y1, label="Com1")
    # plot(rt, y2,"r", label="Com2")
    # plot(rt, y1+y2,"g--", linewidth=2,label="Fitting data")
    # xlabel("Rt (s)")
    # ylabel("Intensity")
    # legend()
    
    fity = []
    for i in range(0, int(len(res.x)/4)):
        y = emg(xdata, res.x[4*i+0], res.x[4*i+1], res.x[4*i+2], res.x[4*i+3])
        fity.append(y)

    return fity

def test():

    R2_1 = []
    coefs1 = []
    for i in range(100):
        ydata2, y1, y2, y3 = simulate_data2(0.06)
        p0 = [ydata2[25], 25, 0.5, ydata2[38], 38, 0.5, ydata2[55], 55, 0.5]
        fity1 = GuassModel(ydata2, p0)
        cors = []
        for i, y in enumerate(fity1):
            cors.append(np.corrcoef(ydata2, y)[0, 1])
        coefs1.append(cors)
        R2_1.append(np.linalg.norm(np.sum(np.array(fity1), axis=0))/np.linalg.norm(ydata2))
    coefs_arr1 = np.array(coefs1)


    R2_2 = []
    coefs2 = []
    for i  in range(100):
        ydata2, y1, y2, y3 = simulate_data2(0.06)
        
        
        fity1 = EMGModel(ydata2, p0)
        cors = []
        for i, y in enumerate(fity1):
            cors.append(np.corrcoef(ydata2, y)[0, 1])
        coefs2.append(cors)
        R2_2.append(np.linalg.norm(np.sum(np.array(fity1), axis=0))/np.linalg.norm(ydata2))
    coefs_arr2 = np.array(coefs2)

    boxplot([coefs_arr1[2, :], coefs_arr2[2, :]])
    show()
    boxplot([R2_1, R2_2])
    show()
    return

def emgfit1(x, *param):
    group = int((len(param)-1)/4)
    fy = 0
    for i in range(group):
        fy += emg(x, param[4*i+0], param[4*i+1], param[4*i+2], param[4*i+3])
    return fy

if __name__ == "__main__":

    ### simulated data1
    C = 1
    ydata, y1_o, y2_o, y3_o = simulate_data1(0.01)

    ### guass model
    p1 = [ydata[18], 22, 0.5, ydata[38], 38, 0.5, ydata[65], 55, 0.5]
    fity2 = GuassModel(ydata, p1)
    sum_y = np.sum(np.array(fity2),axis=0)
    R2 = np.linalg.norm(sum_y)/np.linalg.norm(ydata)
    print(R2)
    for i in range(len(fity2)):
        plot(fity2[i], label="Fitting c"+str(C)+str(i))
    plot(ydata, color='k', label="Origin data")
    plot(sum_y, 'g--', label="Fitting data")
    legend()
    xlabel("Scan")
    ylabel("Intensity")
    show()

    ### DEEMG model
    p0 = [ydata[18], 0.5, 0.5, 18, ydata[38], 0.5, 0.5, 38, ydata[65], 0.5, 0.5, 65]
    fity3 = DEEMGModel(ydata, p0)
    sum_y = np.sum(np.array(fity3),axis=0)
    R2 = np.linalg.norm(sum_y)/np.linalg.norm(ydata)
    print(R2)
    for i in range(len(fity3)):
        plot(fity3[i], label="Fitting c"+str(C)+str(i))
    plot(ydata, color='k', label="Origin data")
    plot(sum_y, 'g--', label="Fitting data")
    legend()
    xlabel("Scan")
    ylabel("Intensity")
    show()
