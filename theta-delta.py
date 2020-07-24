#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:54:47 2020

@author: marianne
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore')

import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling.models import BlackBody1D
from astropy import units as u
from scipy.interpolate import interp1d
from scipy import interpolate

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

sns.set_style('ticks')
sns.set_style('whitegrid')
sns.set_context("talk", font_scale=0.9, rc={"lines.linewidth": 1.5})
sns.set_context(rc = {'patch.linewidth': 0.0})

#%%

#bin by ustar/omega

ct=15

d_binned = np.zeros((ct,8))
x = ustarwc_gm/omega
y = (delta[1,:]+delta[5,:])/2
d = np.zeros((delta.shape))
#y = np.asarray(range(384))


for ii in range(8):
    a,temp = nanrm2(x,delta[ii,:])
    b,c = remove_outliers(a,temp,'pca')
    d[ii,0:len(c)] = c
#y = np.asarray(range(384))

x,y = nanrm2(x,y)
x,y = remove_outliers(x,y,'pca')

ymean, edges, bnum = scipy.stats.binned_statistic(naninterp(x),naninterp(y),'mean',bins = ct)
ystd, e, bn  = scipy.stats.binned_statistic(naninterp(x),naninterp(y),'std',bins = ct)

#ymean2=[]
for ii in range(ct):
    idx = ((x>=edges[ii]) & (x<edges[ii+1]))
    d_binned[ii,:] = np.nanmean(d[:,0:len(idx)][:,idx],axis=1)

ymean2=np.sort(ymean)


sns.set_style('ticks')
sns.set_context("talk", font_scale=0.9, rc={"lines.linewidth": 1.5})
sns.set_context(rc = {'patch.linewidth': 0.0})
fig, ax = plt.subplots(1, figsize=(8, 5), gridspec_kw={'hspace': 0.20, 'wspace': 0.15})

# choose a colormap
c_m = matplotlib.cm.plasma

#d_means = (usbins[:,0]+usbins[:,4])/2
norm = matplotlib.colors.Normalize( vmin=np.nanmin(ymean2), vmax=np.nanmax(ymean2))


# create a ScalarMappable and initialize a data structure
s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
s_m.set_array([])

for i in range(len(usbins)):
    deltas = d_binned[i]
    ax.plot(phasebins,  deltas, 'o:',markersize=5, color=s_m.to_rgba(ymean2[i]))

plt.colorbar(s_m, label=r'$u_*/\omega$')

phaselabels = [r'$-\pi$',r'$-\frac{3\pi}{4}$',r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$', 
               r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$' ]
plt.xticks(ticks = phasebins, labels = phaselabels)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\Delta~[m]$')

plt.savefig('theta-delta.pdf', dpi=500)
