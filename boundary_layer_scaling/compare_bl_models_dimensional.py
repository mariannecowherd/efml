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

bl = np.load('../blparams.npy', allow_pickle = True).item()

delta = naninterp(np.nanmean(bl['delta'][[1,5],:], axis = 0))

laminar = np.sqrt(2*1e-6/bl['omega'])

turb = bl['ustarwc_meas']/bl['omega']

ab = naninterp(bl['ubvec']/bl['omega'])

# dab = delta/ab
 
# dk = delta/0.01
# ak = ab/0.01

# ustar = naninterp(bl['ustarwc_meas'])
# wc_scale = naninterp(0.41*ustar/bl['ubvec'])
#%% Laminar scaling 
lbins = np.linspace(6e-4, 1.2e-3, 10)


lplot = (lbins[:-1] + lbins[1:])/2
dmean, _ , bnum = binned_statistic(laminar, delta, statistic = 'mean', bins = lbins)
dstd, _ , _  = binned_statistic(laminar, delta,'std',bins = lbins)

ci95d = np.zeros_like(dmean)
for ii in range(len(ci95d)):
    ci95d[ii] = 1*dstd[ii]/np.sqrt(np.sum(bnum == (ii+1)))
    
    
#%% plotting 


fig, (ax1, ax2, ax3) = plt.subplots(1,3)

ax1.errorbar(lplot, dmean, yerr = dstd, fmt = 'o', capsize = 2, color = '0.0')
# ax1.plot(Relin,np.ones_like(Relin),':', color = '0.3', label = 'laminar')
# ax1.plot(Relin,0.0465*(Relin**(-0.1)), ':', color = '0.7', label = 'smooth turbulent')
# ax1.set_xscale('log')
# ax1.set_yscale('log')

ax1.set_xlabel(r'$\left(\frac{2 \nu}{\omega}\right)^{1/2}$')
ax1.set_ylabel(r'$\delta$')
ax1.ticklabel_format(axis = 'x',style = 'sci', scilimits = (0,0))
# ax1.legend()

#%%  turbulent scaling
ubins = np.linspace(0,0.015,10)

uplot = (ubins[:-1] + ubins[1:])/2
dmean, _ , bnum = binned_statistic(turb, delta, statistic = 'mean', bins = ubins)
dstd, _ , _  = binned_statistic(turb, delta,'std',bins = ubins)

ci95d = np.zeros_like(dmean)
for ii in range(len(ci95d)):
    ci95d[ii] = 1*dstd[ii]/np.sqrt(np.sum(bnum == (ii+1)))
    
ax2.errorbar(uplot, dmean, yerr = dstd, fmt = 'o', capsize = 2, color = '0.0')

ax2.set_xlabel(r'$\frac{u_{*wc}}{\omega}$')
ax2.set_ylabel(r'$\delta$')
ax2.ticklabel_format(axis = 'x',style = 'sci', scilimits = (0,0))
# ax2.legend()

#%% Plotting combined wave-current case

abins = np.logspace(-2.5,-1,10)


aplot = (abins[:-1] + abins[1:])/2
dmean, _ , bnum = binned_statistic(ab, delta, statistic = 'mean', bins = abins)
dstd, _ , _  = binned_statistic(ab, delta,'std',bins = abins)

ci95d = np.zeros_like(dmean)
for ii in range(len(ci95d)):
    ci95d[ii] = 1*dstd[ii]/np.sqrt(np.sum(bnum == (ii+1)))
    

ax3.errorbar(aplot, dmean, yerr = dstd, fmt = 'o', capsize = 2, color = '0.0')
# ax3.plot(wlin,m*wlin + b, '--', color = '0.0', label = 'LS')
# ax3.plot(wlin, 2*wlin + b, ':', color = '0.3', label = 'GM1979')
# ax3.plot(wlin, 0.367*wlin + b, ':', color = '0.7', label = 'CJ1985')
# ax3.legend()

ax3.set_ylabel(r'$\delta$')
ax3.set_xlabel(r'$a_b$')
ax3.set_xscale('log')
ax3.set_xticks([1e-2,1e-1])

fig.set_size_inches(15,5)
fig.tight_layout(pad = 0.5)
# plt.savefig('/Users/gegan/Desktop/OliverMeetings/8-20/boundary_layer_dimensional.pdf')