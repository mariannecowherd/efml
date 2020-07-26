#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:54:47 2020

@author: marianne, gegan

@title: theta-delta
"""

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import scipy

import vectrinofuncs as vfs

import warnings
warnings.simplefilter('ignore')

#plot styles
sns.set_style('ticks')
sns.set_context("talk", font_scale=0.9, rc={"lines.linewidth": 1.5})
sns.set_context(rc = {'patch.linewidth': 0.0})


params = {
   'axes.labelsize': 14,
   'font.size': 14,
   'legend.fontsize': 12,
   'xtick.labelsize': 14,
   'ytick.labelsize': 14,
   'text.usetex': True,
   'font.family': 'serif',
   'axes.grid' : False,
   'image.cmap': 'plasma'
   }


plt.rcParams.update(params)
plt.close('all')


#bin by ustar/omega

#data
blparams = np.load('/Users/Marianne/Documents/GitHub/efml/blparams.npy',allow_pickle=True).item()
delta = blparams['delta']
phasebins = blparams['phasebins']
ustarwc_gm = blparams['ustarwc_gm']
omega = blparams['omega']

#how many bins
ct=7

d_binned = np.zeros((ct,8))
x = ustarwc_gm/omega
y = (delta[1,:]+delta[5,:])/2
d = np.zeros((delta.shape))

#keeps the delta vector the same size as x with the same values removed
for ii in range(8):
    a,temp = vfs.nanrm2(x,delta[ii,:])
    b,c = vfs.remove_outliers(a,temp,'pca')
    d[ii,0:len(c)] = c

x,y = vfs.nanrm2(x,y)
x,y = vfs.remove_outliers(x,y,'pca')

ymean, edges, bnum = scipy.stats.binned_statistic(vfs.naninterp(x),vfs.naninterp(y),'mean',bins = ct)
ystd, e, bn  = scipy.stats.binned_statistic(vfs.naninterp(x),vfs.naninterp(y),'std',bins = ct)

for ii in range(ct):
    idx = ((x>=edges[ii]) & (x<edges[ii+1]))
    d_binned[ii,:] = np.nanmean(d[:,0:len(idx)][:,idx],axis=1)
    
#
emean = [(edges[i] + edges[i+1])/2 for i in range(len(edges)-1)]


fig, ax = plt.subplots(1, figsize=(8, 5), gridspec_kw={'hspace': 0.20, 'wspace': 0.15})

# choose a colormap
c_m = matplotlib.cm.plasma

#norm is a class of values based on the observed range
norm = matplotlib.colors.Normalize(vmin=np.nanmin(emean), vmax=np.nanmax(emean))

# create a ScalarMappable and initialize a data structure
s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
s_m.set_array([])

for i in range(len(d_binned)):
    deltas = d_binned[i]
    ax.plot(phasebins,  deltas, 'o:',markersize=5, color=s_m.to_rgba(emean[i]))

plt.colorbar(s_m, label=r'$u_*/\omega$')

phaselabels = [r'$-\pi$',r'$-\frac{3\pi}{4}$',r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$', 
               r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$' ]
plt.xticks(ticks = phasebins, labels = phaselabels)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\Delta~[m]$')

plt.savefig('theta-delta.pdf', dpi=500)
