from _ICExtract import DDAMS, DDA_PIC
import sys, math, csv, os
from sko.GA import GA
import scipy.special as sse
from scipy.optimize import curve_fit
from scipy.signal import medfilt
from scipy.optimize import differential_evolution
import numpy as np
import pylab
from pylab import plot, legend, figure, scatter, xlabel, ylabel, show, vlines, FormatStrFormatter, subplot, subplots_adjust
from Smooth import lm_smooth
from Baseline import airPLS_method
from MSPlot import Scatter_p, plot_region

def getNoiseByMedfilt(xb, n_peakpoint, peakwidth, Kp):
    
    if len(n_peakpoint) > 1:
        for k in range(len(n_peakpoint)-1, 0, -1):
            if n_peakpoint[k, 0] == n_peakpoint[k-1, 2]:
                n_peakpoint[k-1, 2] = n_peakpoint[k, 2]
                n_peakpoint = np.delete(n_peakpoint, k, axis=0)
    
    int_array = np.copy(xb)
    for k, p in enumerate(n_peakpoint):
        ttt = np.arange(p[0], p[2]+1)
       
        if k == 0:
            if p[0] > 0:
                peak_prev = np.arange(0, p[0])
            else:
                peak_prev = np.array([p[0]])
        else:
            if n_peakpoint[k-1][2] < p[0]:
                peak_prev = np.arange(n_peakpoint[k-1][2]+1, p[0]+1)
            else:
                peak_prev = np.array([p[0]])
        
        if k == len(n_peakpoint)-1:
            if p[2] < len(xb)-1:
                peak_after = np.arange(p[2]+1, len(xb))
            else:
                peak_after = np.array([p[2]])
        else:
            if n_peakpoint[k+1][0] > p[2]:
                peak_after = np.arange(p[2]+1, n_peakpoint[k+1][0]+1)
            else:
                peak_after = np.array([p[2]])
                
        if len(peak_prev) == 1 and len(peak_after) >= 1:
            if int_array[peak_prev] > 2*int_array[peak_after][0]:
                int_array[peak_prev] = min(int_array[peak_after])
        if len(peak_after) == 1 and len(peak_prev) >= 1:
            if int_array[peak_after] > 2*int_array[peak_prev[-1]]:
                int_array[peak_after] = min(int_array[peak_prev])
                
        traindata = np.hstack((peak_prev, peak_after))
        a = np.polyfit(traindata, int_array[traindata], 1)
        b = np.poly1d(a)
        int_array[ttt] = b(ttt)
    noise = medfilt(int_array, int(np.fix(peakwidth*Kp)+1))
    
    return noise
    
def Findpeaks(ic, min_lens, N, minpeakheight):

    xb = airPLS_method(ic, thr=35)
    xb[xb<0.0] = 0.0
    xs = lm_smooth(np.arange(len(xb)), xb, sl=3)
    
    ic_diff = np.diff(xs)
    temp_ic_diff = np.zeros(len(xs))
    temp_ic_diff[np.where(ic_diff > 0)[0]] = 1
    temp_peakpoint = np.diff(temp_ic_diff)
    temp = np.where(temp_peakpoint==-1)[0] + 1
    temp0 = np.where(temp_peakpoint == 1)[0] + 1

    if len(temp) >= 1:
        peakpoint = np.zeros((len(temp), 3), dtype="int64")

        if temp[0] < temp0[0]:
            peakpoint[0, 0] = 0
            peakpoint[:, 1] = temp

            if len(temp) > len(temp0):
                peakpoint[1:, 0] = temp0
                peakpoint[0:len(temp)-1, 2] = temp0
                peakpoint[len(temp)-1, 2] = len(xs)-1
            else:
                peakpoint[1:, 0] = temp0[0:len(temp0)-1]
                peakpoint[:, 2] = temp0
        else:
            peakpoint[:, 0] = temp0
            peakpoint[:, 1] = temp

            if len(temp) > len(temp0)-1:
                peakpoint[0:len(temp)-1, 2] = temp0[1:]
                peakpoint[len(temp)-1, 2] = len(xs)-1
            else:
                peakpoint[:, 2] = temp0[1:]
    else:
        peakpoint = np.array([])

    ## update peakpoint to make sure having more than 15 points in each peak
    if len(peakpoint) >= 1:
        temp = peakpoint[:, 2] - peakpoint[:, 0]
        peakpoint = peakpoint[np.where(temp >= min_lens)[0], :]
    if len(peakpoint) == 0: return xb, xs, []

    
    total_peakpoint = []
    tn = 1
    for i, p in enumerate(peakpoint):
        if len(temp_ic_diff[p[0]:p[1]])>=N and len(temp_ic_diff[p[1]:p[2]])>=N:
                if np.all(temp_ic_diff[p[1]-N:p[1]]==1) and np.all(temp_ic_diff[p[1]:p[1]+N]==0):
                    total_peakpoint.append(p)
        elif len(temp_ic_diff[p[0]:p[1]])>=N and len(temp_ic_diff[p[1]:p[2]])>=tn:
                if np.all(temp_ic_diff[p[1]-N:p[1]]==1) and np.all(temp_ic_diff[p[1]:p[1]+tn]==0):
                    total_peakpoint.append(p)
        elif len(temp_ic_diff[p[0]:p[1]])>=tn and len(temp_ic_diff[p[1]:p[2]])>=N:
                if np.all(temp_ic_diff[p[1]-tn:p[1]]==1) and np.all(temp_ic_diff[p[1]:p[1]+N]==0):
                    total_peakpoint.append(p)
    if len(total_peakpoint) == 0: return xb, xs, []
        

    for i in range(len(total_peakpoint)-1, -1, -1):
        p = total_peakpoint[i]
        temp_r = np.sum(np.abs(xs[p[0]:p[2]+1]-xb[p[0]:p[2]+1]))/np.sum(xs[p[0]:p[2]+1])
        if temp_r > 0.30:
            total_peakpoint.pop(i)
    if len(total_peakpoint) == 0: return xb, xs, []
    
    Kp = 2
    n_peakpoint = np.array(total_peakpoint)
    peakwidth = max(n_peakpoint[:, 2]-n_peakpoint[:, 0]+1)
    noise = getNoiseByMedfilt(xb, n_peakpoint, peakwidth, Kp)
    
    for i in range(len(total_peakpoint)-1, -1, -1):
        p = total_peakpoint[i]
        if xs[p[1]] <= minpeakheight or xs[p[1]]/noise[p[1]] <= 3:
            total_peakpoint.pop(i)
    if len(total_peakpoint) == 0: return xb, xs, []    
                   
    return xb, xs, noise, total_peakpoint
          
def getpeakgroup(total_p, xb, thre=0.1):
    
    group_index = []
    if len(total_p) > 1:
        for i in range(len(total_p)-1):
            aa = total_p[i]
            bb = total_p[i+1]
            if aa[2] == bb[0]:
                if xb[aa[2]] >= thre * min([xb[aa[1]], xb[bb[1]]]):
                    group_index.append([i, i+1])
                else:
                    group_index.append([i, i])
            else:
                group_index.append([i, i])
                group_index.append([i+1, i+1])
    else:
        group_index.append([0, 0])
        
    for i in range(len(group_index)-1, 0, -1):
        if group_index[i][0] == group_index[i-1][1]:
            group_index[i-1][1] = group_index[i][1]
            group_index.pop(i)
    
    total_p_arr = np.array(total_p)
    infor_group = []
    for i in range(len(group_index)):
        infor_group.append(total_p_arr[group_index[i][0]:group_index[i][1]+1, :])
        
    return infor_group

def gethidepeakpoint(xs):

    diffy = np.diff(xs)
    diffy2 = np.diff(diffy)

    p = []
    for i in range(len(diffy2)-1):
        if diffy2[i] >= 0 and diffy2[i+1] < 0:
            state = 1
            if i-2 >= 0: 
                for k in range(i-2, i+1):
                    if diffy[k] > diffy[k+1]:
                        state = 0
            else:
                state = 0
            if i+4 < len(diffy)-1:
                for k in range(i+1, i+4):
                    if diffy[k] < diffy[k+1]:
                        state = 0
            else:
                state = 0
            if state == 1:
                if diffy[i] < 0:
                    p.append(i+2)
        elif diffy2[i] < 0 and diffy2[i+1] >= 0:
            state = 1
            if i-2 >= 0: 
                for k in range(i-2, i+1):
                    if diffy[k] < diffy[k+1]:
                        state = 0
            else:
                state = 0
            if i+4 < len(diffy)-1:
                for k in range(i+1, i+4):
                    if diffy[k] > diffy[k+1]:
                        state = 0
            else:
                state = 0
            if state == 1:
                if diffy[i] > 0:
                    p.append(i+2)
    hidep = p
    return np.array(hidep)

def DEEMGfit(xdata, ydata):
     
    bounds = [(1, 10000000), (0, 10000000), (0, 10000000), (0, len(xdata)),
              (1, 10000000), (0, 10000000), (0, 10000000), (0, len(xdata))]
    result = differential_evolution(DEEMG, bounds, args=(xdata, ydata))
    
    return result

def DEEMG(param, *data):
    group = int(len(param)/4)
    x, y = data
    fy = 0
    for i in range(group):
        fy += emg(x, param[4*i+0], param[4*i+1], param[4*i+2], param[4*i+3])
    return 1/np.square(y - fy).sum()

def emgfit(x, *param):
    group = int(len(param)/4)
    fy = 0
    for i in range(group):
        fy += emg(x, param[4*i+0], param[4*i+1], param[4*i+2], param[4*i+3])
    return fy

def emg(x, A, t0, w, xc):  # exponential gaussian
    return (A / t0) * np.exp(0.5 * (w / t0) ** 2 - (x - xc) / t0) * (0.5 + 0.5 * sse.erf(((x - xc) / w - w / t0) / 2 ** 0.5))
   
def Get_CIC(file, EER1, min_int1, MaxMissedScan, min_lens):

    mzfile = file.encode(sys.getfilesystemencoding())
    ddams = DDAMS()
    ddams.loadfile(mzfile)

    # print(rts)
    # TIC = ddams.getTIC(0)
    # plot(TIC[:, 0], TIC[:, 1], color='r', label="Standards:12 ng/mL")
    # plot(TIC[:, 0], TIC[:, 1], color='g', label="Standards with tea:12 ng/mL")
    # xlabel("Retetion time (s)")
    # ylabel("Intensity")
    # legend()
    # show()

    ###  detect ion chromatograms
    CICs = DDA_PIC(ddams, EER1, min_int1, MaxMissedScan, min_lens)

    return CICs

def Gettic(file):
    mzfile = file.encode(sys.getfilesystemencoding())
    ddams = DDAMS()
    ddams.loadfile(mzfile)
    tic = ddams.getTIC(0)
    return tic

if __name__ == "__main__":
    ### get peak feature from mzML or mzXML file
    file = "E:/decPIC/Standards_tea_extraction/mzxml/1-12ng.mzXML"
    CICs = Get_CIC(file, 0.02, 300, 5, 10)

    ### find peaks of ion chromatogram
    ind = 1500
    IC = CICs[ind]
    xb, xs, noise, total_peakpoint = Findpeaks(IC[:, 2], 5, 2, 300)
    # peak group for complex ion chromatogram
    infor_group = getpeakgroup(total_peakpoint, xb, thre=0.1)

    ### DEEMG model for complex ion chromatogram
    ind_cic = 2
    pg = infor_group[ind_cic]
    xdata = np.arange(len(xb))
    ydata = xb
    result = DEEMGfit(xdata, ydata)
