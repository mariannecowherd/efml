#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 14:43:26 2020

@author: gegan
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

waveturb = np.load('waveturb.npy', allow_pickle = True).item()
bl = np.load('blparams.npy', allow_pickle = True).item() 

idx = list(waveturb.keys())
del idx[292]
del idx[203]

allsed = np.load('/Users/gegan/Documents/Python/Research/Erosion_SD6/allsedP1_sd6.npy', 
                 allow_pickle = True).item()
ubar = allsed['ubar'][:,idx]
z = allsed['z'][:,idx]
dubardz = np.zeros_like(ubar)

m,n = ubar.shape
for i in range(n):
    idxgood = (z[:,i] > 0)
    tck = interpolate.splrep(np.flipud(z[idxgood,i]),np.flipud(ubar[idxgood,i]))
    znew = np.linspace(np.nanmin(z[idxgood,i]),np.nanmax(z[idxgood,i]),200)
    dubardz[:,i] =  np.interp(z[:,i],znew,interpolate.splev(znew,tck, der = 1))

data = np.load('phase_stress.npy', allow_pickle = True).item()

dphi = np.pi/4 #Discretizing the phase
phasebins = np.arange(-np.pi,np.pi,dphi)
delta = np.nanmean(bl['delta'][:,idx],axis = 1)

phaselabels = [r'$-\frac{3\pi}{4}$',r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$', 
               r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',
               r'$\pi$']

params = {
   'axes.labelsize': 28,
   'font.size': 28,
   'legend.fontsize': 18,
   'xtick.labelsize': 28,
   'ytick.labelsize': 28,
   'text.usetex': True,
   'font.family': 'serif'
   }


#%% Equation 3.2c
intmask = ((data['z'][:,idx] < 0.0105) & (data['z'][:,idx] > 0.0005)).astype(int)

data['epsilon'][np.isnan(data['epsilon'])] = 0
data['tke'][np.isnan(data['tke'])] = 0
data['tke_wave'][np.isnan(data['tke_wave'])] = 0
data['uw'][np.isnan(data['uw'])] = 0
data['uw_wave'][np.isnan(data['uw_wave'])] = 0
data['dudz'][np.isnan(data['dudz'])] = 0
dubardz[np.isnan(dubardz)] = 0


#%% nu_t from k-epsilon model vs delta
scale = (1/(np.sum(intmask,axis = 0)*0.001))

epsilon = np.array([np.nanmean( scale*np.trapz(data['epsilon'][:,idx,i]*intmask, 
                                         np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ])

k = np.array([np.nanmean(scale*np.trapz(data['tke'][:,idx,i]*intmask,
                                  np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )


nut = 0.09*(k**2)/epsilon

fig, ax1 = plt.subplots()

color = 'C0'
ax1.plot(phasebins, delta, color = color)
ax1.set_ylabel(r'$\delta$ (m)', color = color)
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
ax1.tick_params(axis='y', labelcolor=color)

color = 'C1'
ax2 = ax1.twinx()
ax2.plot(phasebins, nut, color = color)
ax2.set_ylabel(r'$\nu_T$ (m$^2$ s$^{-1}$)', color = color)
ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
ax2.tick_params(axis='y', labelcolor=color)

plt.xticks(ticks = phasebins, labels = phaselabels)

fig.set_size_inches(8,5)
fig.tight_layout(pad = 0.5)
plt.rcParams.update(params)

#%% TKE terms
burstnums = list(range(384))
eps = np.zeros((30,len(burstnums)))

for ii in burstnums:
    diss = np.load('/Users/gegan/Documents/npyrevised/dissdAll_' + str(ii) + '.npy',allow_pickle = True).item()
    eps[:,ii] = diss['epsF']

tke = np.load('tke_balance.npy', allow_pickle = True).item()

tke['Ps'][np.isnan(tke['Ps'])] = 0
tke['Pw'][np.isnan(tke['Pw'])] = 0
tke['wpk'][np.isnan(tke['wpk'])] = 0
tke['wtk'][np.isnan(tke['wtk'])] = 0
    
scale = (1/(np.sum(intmask,axis = 0)*0.001))

Ps = scale*np.trapz(tke['Ps'][:,idx]*intmask, np.flipud(data['z'][:,idx]), axis = 0)

Pw = scale*np.trapz(tke['Pw'][:,idx]*intmask, np.flipud(data['z'][:,idx]), axis = 0)

epsilon = scale*np.trapz(eps[:,idx]*intmask, np.flipud(data['z'][:,idx]), axis = 0)


wpk = scale*np.trapz(tke['wpk'][:,idx]*intmask, np.flipud(data['z'][:,idx]), axis = 0)

wtk = scale*np.trapz(tke['wtk'][:,idx]*intmask, np.flipud(data['z'][:,idx]), axis = 0)

dkdt = scale*np.trapz(tke['dkdt'][:,idx]*intmask, np.flipud(data['z'][:,idx]), axis = 0)

# fig, ax = plt.subplots()

# ax.plot(Ps + epsilon, label = r'-$\overline{u^{\prime} w^{\prime} } \frac{\partial \overline{u}}{\partial z} - \epsilon$')
# ax.plot(-epsilon, label = r'$\epsilon$')
# ax.plot(-Pw , label = r'$- \overline{ \langle u^{\prime} w^{\prime} \rangle \frac{\partial \tilde{u}}{\partial z} }$' )
# ax.plot(-wpk - wtk, label = r'$- \frac{\partial}{\partial z}\left( \overline{w^{\prime} k } \right) $')
# ax.plot(-wtk, label = r'- $\overline{ \tilde{w} \frac{ \partial \langle k \rangle }{\partial z} } $')
# ax.plot(-dkdt, label = r'$\frac{ \partial k }{ \partial t}$')

# ax.legend()
#%%

fig, ax = plt.subplots()

# ax.plot(phasebins, Ps, label = r'$\overline{u^{\prime} w^{\prime} } \frac{\partial \overline{u}}{\partial z}$')
ax.plot(phasebins, Ps - epsilon, label = r'$-\overline{u^{\prime} w^{\prime} } \frac{\partial \overline{u}}{\partial z} - \epsilon$')
ax.plot(phasebins, Pw, label = r'$-\overline{\tilde{u} \tilde{w} } \frac{\partial \tilde{u}}{\partial z}$')
ax.plot(phasebins,np.zeros_like(phasebins),'-', color = '0.8')
ax.plot(phasebins, wpk, label = r'$- \frac{\partial}{\partial z}\left( \overline{w^{\prime} k } \right) $')
ax.plot(phasebins, wtk, label = r'- $\overline{ \tilde{w} \frac{ \partial k}{\partial z} } $')
# ax.plot(phasebins, epsilon, label = r'$\epsilon$')
# ax.plot(phasebins, diffusion, label = r'$\frac{\partial kw^{\prime}}{\partial z}$')
ax.set_ylabel('Terms in TKE balance')
ax.legend()
plt.xticks(ticks = phasebins, labels = phaselabels)

fig.set_size_inches(8,5)
fig.tight_layout(pad = 0.5)
plt.rcParams.update(params)