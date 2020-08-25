#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:53:59 2020

@author: gegan
"""

import sys
sys.path.append('/Users/gegan/Documents/Python/Research/General')

import matplotlib.pyplot as plt
from mylib import naninterp

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic,linregress

params = {
   'axes.labelsize': 28,
   'font.size': 28,
   'legend.fontsize': 18,
   'xtick.labelsize': 28,
   'ytick.labelsize': 28,
   'text.usetex': True,
   'font.family': 'serif'
   }


bl = np.load('blparams.npy', allow_pickle = True).item()

delta = naninterp(np.nanmean(bl['delta'][[1,5],:], axis = 0))

ab = naninterp(bl['ubvec']/bl['omega'])
Rew = naninterp((ab**2)*bl['omega']/1e-6)

dab = delta/ab

dk = delta/0.01
ak = ab/0.01

ustar = naninterp(bl['ustarwc_meas'])
wc_scale = naninterp(0.41*ustar/bl['ubvec'])
#%% Laminar and smooth turbulent case
rbins = np.logspace(1.6, 4.3, 10)


rplot = (rbins[:-1] + rbins[1:])/2
dmean, _ , bnum = binned_statistic(Rew, dab, statistic = 'mean', bins = rbins)
dstd, _ , _  = binned_statistic(Rew, dab,'std',bins = rbins)

ci95d = np.zeros_like(dmean)
for ii in range(len(ci95d)):
    ci95d[ii] = 1*dstd[ii]/np.sqrt(np.sum(bnum == (ii+1)))
    
    
#%% Plotting laminar case

Relin = np.linspace(25, np.nanmax(Rew), 1000)

fig, (ax1, ax2, ax3) = plt.subplots(1,3)

ax1.errorbar(rplot, dmean, yerr = dstd, fmt = 'o', capsize = 2, color = '0.0')
ax1.plot(Relin,np.pi/np.sqrt(2*Relin),':', color = '0.3', label = 'laminar')
ax1.plot(Relin,0.0465*(Relin**(-0.1)), ':', color = '0.7', label = 'smooth turbulent')
ax1.set_xscale('log')
ax1.set_xticks([1e2,1e3,1e4])
ax1.set_yscale('log')

ax1.set_xlabel(r'$Re_w = \frac{a_b^2 \omega}{\nu}$')
ax1.set_ylabel(r'$\delta a_b^{-1}$')
ax1.legend()

#%%  rough turbulent case
abins = np.linspace(0, 10, 10)
alin = np.linspace(0,10,1000)

aplot = (abins[:-1] + abins[1:])/2
dmean, _ , bnum = binned_statistic(ak, dk, statistic = 'mean', bins = abins)
dstd, _ , _  = binned_statistic(ak, dk,'std',bins = abins)

ci95d = np.zeros_like(dmean)
for ii in range(len(ci95d)):
    ci95d[ii] = 1*dstd[ii]/np.sqrt(np.sum(bnum == (ii+1)))
    
ax2.errorbar(aplot, dmean, yerr = dstd, fmt = 'o', capsize = 2, color = '0.0')
ax2.plot(alin,0.072*(alin**0.75),':', color = '0.3', label = 'J1980')
ax2.plot(alin,0.27*(alin**0.67),':', color = '0.7', label = 'S1987')

ax2.set_xlabel(r'$a_b k_s^{-1}$')
ax2.set_ylabel(r'$\delta k_s^{-1}$')
ax2.legend()

#%% Plotting combined wave-current case

wbins = np.linspace(0,0.14,10)
wlin = np.linspace(0,0.14,1000)

wplot = (wbins[:-1] + wbins[1:])/2
dmean, _ , bnum = binned_statistic(wc_scale, dab, statistic = 'mean', bins = wbins)
dstd, _ , _  = binned_statistic(wc_scale, dab,'std',bins = wbins)

ci95d = np.zeros_like(dmean)
for ii in range(len(ci95d)):
    ci95d[ii] = 1*dstd[ii]/np.sqrt(np.sum(bnum == (ii+1)))
    
#least squares fit
m, b, _, _, _ = linregress(wplot,dmean)

ax3.errorbar(wplot, dmean, yerr = dstd, fmt = 'o', capsize = 2, color = '0.0')
ax3.plot(wlin,m*wlin + b, '--', color = '0.0', label = 'LS')
ax3.plot(wlin, 2*wlin + b, ':', color = '0.3', label = 'GM1979')
ax3.plot(wlin, 0.367*wlin + b, ':', color = '0.7', label = 'CJ1985')
ax3.legend()

ax3.set_ylabel(r'$\delta a_b^{-1}$')
ax3.set_xlabel(r'$\kappa u_{*wc} u_b^{-1}$')

plt.rcParams.update(params)
fig.set_size_inches(15,5)
fig.tight_layout(pad = 0.5)
plt.savefig('/Users/gegan/Desktop/OliverMeetings/8-20/boundary_layer_scaling_2.pdf')