#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:08:43 2020

@author: marianne

@title: delta_theories
"""

'''
This code calculates displacement thickness delta scalings and compares them
to observed thickness from the field campaign
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy
from vectrinofuncs import nanrm2

#location of stress profiles
filepath = '/Users/Marianne/Documents/GitHub/efml/'
stress = np.load(filepath + 'phase_stress_alt.npy', allow_pickle = True).item()
blparams = np.load(filepath + 'blparams_alt.npy',allow_pickle=True).item()

delta = blparams['delta']
omega = blparams['omega']
ustarwc_sg17 = blparams['ustarwc_sg17']
ustarwc_gm = blparams['ustarwc_gm']
ustar_adv5 = blparams['ustar_adv5']
ustar_logfit = blparams[ 'ustar_logfit']
ustarwc_meas = blparams['ustarwc_meas']
ustarc_meas = blparams[ 'ustarc_meas']
ubvec = blparams['ubvec']


params = {
   'axes.labelsize': 28,
   'font.size': 28,
   'legend.fontsize': 12,
   'xtick.labelsize': 28,
   'ytick.labelsize': 28,
   'text.usetex': True,
   'font.family': 'serif',
   'axes.grid' : False,
   'image.cmap': 'plasma'
   }



plt.rcParams.update(params)
plt.close('all')

nu=1e-6
Re_d = ubvec * np.sqrt(2*nu/omega)/nu

ct=50

d1 = ((delta[0]+delta[4])/2 * np.sqrt(nu))/omega
d2 = (ustarwc_gm**2) / (omega*nu)

d1,d2=nanrm2(d1,d2)

Cdmean, edges, bnum = scipy.stats.binned_statistic(d1,d2,'mean',bins = ct)
Cdstd, e, bn  = scipy.stats.binned_statistic(d1,d2,'std',bins = ct)
mids = (edges[:-1]+edges[1:])/2
ci = np.zeros_like(mids)
for ii in range(len(ci)):
    ci[ii] = Cdstd[ii]/np.sqrt(np.sum(bnum == (ii+1)))

fig,ax = plt.subplots()
ax.errorbar(mids,Cdmean,yerr = ci,fmt = 'ko', capsize = 2)
ax.set_yscale('log')
ax.set_xlabel(r'$\frac{\delta\sqrt{\nu}}{\omega}$')
ax.set_ylabel(r'$\frac{u_*^2}{\omega\nu}$')

plt.save("plots/d1d2.pdf")

d3 = -24/Re_d
d4 = -np.sqrt(ustarwc_gm)/ubvec

fig,ax=plt.subplots()
plt.plot(d3,d4)

d5 = nu/ustarwc_gm
d6 = np.sqrt(ustarwc_gm)/ubvec
