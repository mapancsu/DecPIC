from _spcio import airPLS_PP

def airPLS_method(x, thr=25):
    xs = x - airPLS_PP(x, thr)[:, 0]
    return xs

if __name__=="__main__":
    import numpy as np
    from scipy.stats import norm
    import matplotlib.pyplot as pl
    x=np.arange(0,1000,1)
    g1=norm(loc = 100, scale = 1.0) # generate three gaussian as a signal
    g2=norm(loc = 300, scale = 3.0)
    g3=norm(loc = 750, scale = 5.0)
    signal=g1.pdf(x)+g2.pdf(x)+g3.pdf(x)
    baseline1=5e-4*x+0.2 # linear baseline
    baseline2=0.2*np.sin(np.pi*x/x.max()) # sinusoidal baseline
    noise=np.random.random(x.shape[0])/500
    print ('Generating simulated experiment')
    y1=signal+baseline1+noise
    y2=signal+baseline2+noise

    c1 = airPLS_method(y1)
    #c3 = airPLS_method(y1)
    c2 = airPLS_method(y2)
    #c2 = airPLS_method(y2)

    fig,ax=pl.subplots(nrows=2,ncols=1)
    ax[0].plot(x,y1,'-k')
    ax[0].plot(x,c1,'-r')
    ax[0].set_title('Linear baseline')
    ax[1].plot(x,y2,'-k')
    ax[1].plot(x,c2,'-r')
    ax[1].set_title('Sinusoidal baseline')
    pl.show()