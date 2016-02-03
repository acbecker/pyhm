import matplotlib.pyplot as plt
import numpy as np

def percentiles(data, percent=(68.27, 95.45, 99.73)):
    return np.array([(np.percentile(data, 50-p/2), np.percentile(data, 50+p/2)) for p in percent])

def triangle(params, titles, subfigsize=2.0, whitespace=0.1, fontmin=11):
    nsample, nparam = params.shape
    if len(titles) != nparam:
        print "ERROR: require %d titles" % (nparam)
        return
    
    figsize = nparam * subfigsize + (nparam - 1) * whitespace # debug
    fig, subplots = plt.subplots(nparam, nparam, sharex="col")

    # Core data plotting
    for i in range(nparam):
        sp = subplots[i,i]
        zorder = 1
        for pmin, pmax in percentiles(params[:,i]):
            sp.axvspan(pmin, pmax, alpha=0.2, zorder=zorder); zorder += 1
        sp.hist(params[:,i], zorder=zorder, bins=50); zorder += 1
        sp.axvline(np.percentile(params[:,i], 50), color="r", lw=2, zorder=zorder); zorder += 1
        
        for j in range(i):
            sp = subplots[i,j]
            sp.hexbin(params[:,j], params[:,i], gridsize=35, cmap=plt.cm.Greys)

        for j in range(i+1, nparam):
            plt.setp(subplots[i,j], visible=False)

    # Now fix up all the figures
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i in range(nparam):
        sp = subplots[i,i]
        plt.setp(sp.get_yticklabels(), visible=False) # Just the number of samples; no need to plot as they can be large
        sp.set_xlim(percentiles(params[:,i], (99.99,))[0])
        
        if i == nparam-1:
            sp.set_xlabel(titles[i], weight="bold", labelpad=15, fontsize=fontmin+3)
            sp.xaxis.set_label_coords(0.5, -0.2)
            plt.setp(sp.get_xticklabels(), weight="bold", fontsize=fontmin, rotation=30) 
        else:
            plt.setp(sp.get_xticklabels()+sp.get_yticklabels(), visible=False) 
            
        for j in range(i):
            sp = subplots[i,j]
            if j == 0:
                sp.set_ylabel(titles[i], weight="bold", labelpad=15, fontsize=fontmin+3)
                sp.yaxis.set_label_coords(-0.2, 0.5)
                plt.setp(sp.get_yticklabels(), weight="bold", fontsize=fontmin, rotation=30)
                if i != nparam-1:
                    plt.setp(sp.get_xticklabels(), visible=False)
            elif i == nparam-1:
                sp.set_xlabel(titles[j], weight="bold", fontsize=fontmin+3)
                sp.xaxis.set_label_coords(0.5, -0.2)
                plt.setp(sp.get_xticklabels(), weight="bold", fontsize=fontmin, rotation=30)
                plt.setp(sp.get_yticklabels(), visible=False)
            else:
                plt.setp(sp.get_yticklabels(), visible=False) 
                    
