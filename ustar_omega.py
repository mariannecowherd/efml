#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:31:02 2020

@author: marianne, gegan

@title: ustaromega
"""
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy
import vectrinofuncs as vfs

params = {
   'axes.labelsize': 28,
   'font.size': 28,
   'legend.fontsize': 18,
   'xtick.labelsize': 28,
   'ytick.labelsize': 28,
   'text.usetex': True,
   'font.family': 'serif',
   'axes.grid' : False,
   'image.cmap': 'plasma'
   }


#iswavy = np.empty(384,dtype=bool)
#check if the burst is sufficiently wavy for wave decomposition
#for N in range(384): iswavy[N] = vfs.iswaves(N,False)

#haswaves  = np.arange(384)[iswavy]
haswaves=np.load('haswaves.npy')
iswavy = np.load('iswavy.npy')

plt.rcParams.update(params)
plt.close('all')



blparams = np.load('/Users/Marianne/Documents/GitHub/efml/blparams.npy',allow_pickle=True).item()
delta = blparams['delta']
phasebins = blparams['phasebins']
ustarwc_sg17 = blparams['ustarwc_sg17']
ustarwc_gm = blparams['ustarwc_gm']
ustar_adv5 = blparams['ustar_adv5']
ustar_logfit = blparams[ 'ustar_logfit']
ustarwc_meas = blparams['ustarwc_meas']
ustarc_meas = blparams[ 'ustarc_meas']
ubadv = blparams['ubadv']
ubvec = blparams['ubvec']
omega = blparams['omega']
dirspread = blparams['dirspread']

nu = 1e-6

Re_d = ubvec * np.sqrt(2*nu/omega)/nu


fig,ax = plt.subplots()
sources = ['gm','meas']
xs = [ustarwc_gm,ustarwc_meas]/omega
colors = ['gray','black']

for i in range(len(sources)):
    y=(2*delta[1,:]+2*delta[5,:])/212
    x = xs[i]
    x,y = x[iswavy],y[iswavy]
    x,y = vfs.nanrm2(x,y)
    x,y = vfs.remove_outliers(x,y,'pca')

    
    ct=7
    ymean, edges, bnum = scipy.stats.binned_statistic(vfs.naninterp(x),vfs.naninterp(y),'mean',bins = ct)
    ystd, e, bn  = scipy.stats.binned_statistic(vfs.naninterp(x),vfs.naninterp(y),'std',bins = ct)
    mids = (edges[:-1]+edges[1:])/2
    
    
    ci = np.zeros_like(mids)
    for ii in range(len(ci)):
        ci[ii] = 1.96*ystd[ii]/np.sqrt(np.sum(bnum == (ii+1)))
    
    model=LinearRegression().fit(x.reshape((-1,1)),y.reshape((-1,1)))
    yfit = (model.intercept_ + model.coef_ * x).flatten(order='F')
    
    print(model.coef_)
    ax.errorbar(mids*100,ymean*100,yerr = ci*100,fmt = 'o', color=colors[i],
                capsize = 2,label=(sources[i]+r', $C_1$='+str(round(model.coef_[0][0],4))))
    ax.plot(x*100,yfit*100,':',color = colors[i])#, label = 'm='+str(round(model.coef_[0][0],4)))
    
    #ax.text(0.005,0.004,'slope = ' + str(round(model.coef_[0][0],5)))
    #ax.set_yscale('log')

handles, labels = ax.get_legend_handles_labels()
order = [3,2,1,0]
ax.legend(handles, labels, frameon=False)
#ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],ncol=2,frameon=False)

ax.set_ylabel(r'$\langle\delta\rangle$ (cmab)')
    #ax.set_title(str(source))
ax.set_xlabel(r'$u_*\omega^{-1}$ (cm)')

fig.savefig('plots/all_delta.pdf',dpi=500)

