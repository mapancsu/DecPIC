import pylab
import numpy as np
from pylab import figure, scatter, xlabel, ylabel, show, FormatStrFormatter

def get_region(f, diams, mapi, width, mz_tol):
    (rt, mz, val) = f
    region = diams.getRegion(mapi, rt - width, rt + width, mz - mz_tol, mz + mz_tol)
    return np.array(region).reshape((len(region), region[0].shape[0]))

def plot_region(rg, rt = None, mz = None):
    figure()
    scatter(rg[:, 0], rg[:, 1], c=np.log(rg[:, 2]), s=5)
    if rt != None and mz != None:
        scatter(rt, mz, c='r', marker='x', s=2)
    xlabel('Retention time(s)')
    ylabel('m/z')
    ax = pylab.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    show()

def Scatter_f(f, diams, mapi, width, mz_tol):
    rg = get_region(f, diams, mapi, width, mz_tol)
    plot_region(rg)

def Scatter_p(f, diams, mapi, rt_range, mz_tol):
    (rt, mz, val) = f
    region = diams.getRegion(mapi, rt_range[0], rt_range[1], mz - mz_tol, mz + mz_tol)
    rg = np.array(region).reshape((len(region), region[0].shape[0]))
    return rg