import matplotlib.pyplot as plt
import numpy as np

def runningMoments(data):
    x      = np.cumsum(data)
    x2     = np.cumsum(data**2)
    n      = np.arange(1, len(data)+1)
    mean   = x / n 
    std    = np.sqrt((x2*n - x*x) / (n * (n-1)))
    return mean, std

def walkers(data, labels, npts=250, fontmin=11):
    # Modified from: http://conference.scipy.org/jhepc2013/2013/entry5/index.html
    nwalkers, nstep, ndim = data.shape
    subsample = nstep / npts

    red       = "#cf6775"
    blue      = "#67a9cf"
    fontcolor = "#dddddd"
    fig = plt.figure(facecolor='#222222', figsize=(8.5, 11))
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    for i in range(ndim):
        spA = plt.subplot2grid((ndim,3), (i,0), colspan=2)
        spB = plt.subplot2grid((ndim,3), (i,2))
        spA.set_axis_bgcolor("#333333")
        spB.set_axis_bgcolor("#333333")
        spA.tick_params(direction="out", colors="w")
        spB.tick_params(axis="x", bottom="off", top="off")
        spB.tick_params(axis="y", left="off", right="off")
        spA.tick_params(axis="y", right="off")
        if i != ndim-1: spA.tick_params(axis="x", bottom="off", top="off")
        else: spA.tick_params(axis="x", top="off")

        cmap  = np.log(np.std(data[:,:,i], axis=1))
        cmap -= np.min(cmap)
        cmap /= np.max(cmap)
        for j in range(nwalkers):
            wdata = data[j,:,i]
            rmean, rstd = runningMoments(wdata)

            wdata = wdata[::subsample][1:] # [1:] since std[0] = nan
            rmean = rmean[::subsample][1:]
            rstd  = rstd[::subsample][1:]
            nsub  = np.arange(nstep)[::subsample][1:]
            cmap  = np.abs(wdata-rmean)/rstd

            #spA.plot(nsub, wdata, drawstyle="steps", color="w", alpha=0.15)    
            spA.plot(nsub, wdata, drawstyle="steps", color=plt.cm.bone_r(cmap[i]), alpha=0.15)    
            spA.plot(nsub, rmean, color=red, linestyle="-")
            spA.fill_between(nsub, rmean-rstd, rmean+rstd, facecolor=blue, alpha=0.15)

        spB.hist(np.ravel(data[:,:,i]), orientation='horizontal', facecolor=red, bins=50, edgecolor="none")
        spB.set_ylabel(labels[i], rotation='horizontal', fontsize=fontmin+3, labelpad=15, weight="bold", color=fontcolor)
        spB.set_ylim(spA.get_ylim())
        spB.xaxis.set_visible(False)
        spB.yaxis.tick_right()
        spB.yaxis.set_label_position("right")
        plt.setp(spB.get_yticklabels(), visible=False)
        
        spA.locator_params(nbins=7, axis="y")
        spA.set_yticks(spA.get_yticks()[1:-1])        
        spA.set_xlim(0, nstep)
        if i != ndim-1:
            plt.setp(spA.get_xticklabels(), visible=False)
        else:
            spA.set_xlabel("Step", fontsize=fontmin+3, labelpad=8, weight="bold", color=fontcolor)
            plt.setp(spA.get_xticklabels(), fontsize=fontmin, weight="bold", color=fontcolor)
            
        plt.setp(spA.get_yticklabels(), fontsize=fontmin, weight="bold", color=fontcolor)
