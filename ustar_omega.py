#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:31:02 2020

@author: marianne, gegan

@title: ustaromega
"""
import numpy as np
from sklearn.linear_model import LinearRegression
import scikits.bootstrap as boot
import matplotlib.pyplot as plt
import scipy
import vectrinofuncs as vfs

params = {
   'axes.labelsize': 28,
   'font.size': 28,
   'legend.fontsize': 18,
   'xtick.labelsize': 28,
   'ytick.labelsize': 28,
   'text.usetex': False,
   'font.family': 'serif',
   'axes.grid': False,
   'image.cmap': 'plasma'
   }

# initialize

# uncomment and run this to define wavy vs non-wavy bursts
# otherwise, load wavy vs non-wavy burst identification from files
# iswavy = np.empty(384,dtype=bool)
# check if the burst is sufficiently wavy for wave decomposition
# for N in range(384): iswavy[N] = vfs.iswaves(N,False)
# haswaves  = np.arange(384)[iswavy]

# load wavy vs non-wavy busrts from data files
haswaves = np.load('data/haswaves.npy')
iswavy = np.load('data/iswavy.npy')

plt.rcParams.update(params)
plt.close('all')

bl = np.load('data/blparams.npy',allow_pickle=True).item()
delta = 2*bl['delta']
phasebins = bl['phasebins']
ustarwc_sg17 = bl['ustarwc_sg17']
ustarwc_gm = bl['ustarwc_gm']
ustar_adv5 = bl['ustar_adv5']
ustar_logfit = bl['ustar_logfit']
ustarwc_meas = bl['ustarwc_meas']
ustarc_meas = bl[ 'ustarc_meas']
ubadv = bl['ubadv']
ubvec = bl['ubvec']
omega = bl['omega']
dirspread = bl['dirspread']

# nu_fit_gm = np.load('data/nu_fits_phase.npy', allow_pickle=True).item()


nu = 1e-6

# wave reynolds number
Re_d = ubvec * np.sqrt(2*nu/omega)/nu

# For confidence bounds
def returnlinear(x, y):
    model = LinearRegression(fit_intercept=True)
    result = model.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    return result.coef_[0]


fig, ax = plt.subplots()
sources = ['GM', 'Meas']
xs = [ustarwc_gm, ustarwc_meas]/omega
colors = ['gray', 'black']

for i in range(len(sources)):
    y = (delta[1, :]+delta[5, :])/2
    x = xs[i]
    x, y = x[iswavy], y[iswavy]
    x, y = vfs.nanrm2(x, y)
    x, y = vfs.remove_outliers(x, y, 'pca')

    ct = 7
    ymean, edges, bnum = scipy.stats.binned_statistic(vfs.naninterp(x), vfs.naninterp(y), 'mean', bins=ct)
    ystd, e, bn = scipy.stats.binned_statistic(vfs.naninterp(x), vfs.naninterp(y), 'std', bins=ct)
    mids = (edges[:-1]+edges[1:])/2

    ci = np.zeros_like(mids)
    for ii in range(len(ci)):
        ci[ii] = 1.96*ystd[ii]/np.sqrt(np.sum(bnum == (ii+1)))

    model=LinearRegression().fit(x.reshape((-1,1)),y.reshape((-1,1)))
    yfit = (model.intercept_ + model.coef_ * x).flatten(order='F')
    
    ci_fit = boot.ci((x.reshape((-1,1)),y.reshape((-1,1))),returnlinear)[:,0]
    c1error = np.nanmax(np.abs(model.coef_[0] - ci_fit))

    print(model.coef_)
    ax.errorbar(mids*100, ymean*100, yerr=ci*100,fmt = 'o', color=colors[i],
                capsize=2, label=(sources[i]+r', $C_1$ = '+ str(round(model.coef_[0][0], 2)) + r' $\pm$ ' + str(round(c1error, 2))))
    ax.plot(x*100, yfit*100, ':', color=colors[i])  # , label = 'm='+str(round(model.coef_[0][0],4)))


handles, labels = ax.get_legend_handles_labels()
order = [3, 2, 1, 0]
ax.legend(handles, labels, frameon=False)
ax.set_ylabel(r'$\langle\delta\rangle$ (cm)')
ax.set_xlabel(r'$u_*\omega^{-1}$ (cm)')
fig.set_size_inches(8, 6)
fig.tight_layout(pad=0.5)
plt.show()

fig.savefig('plots/ustar_omega.pdf', dpi=500)
