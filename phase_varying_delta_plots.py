#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:45:18 2020

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

nu_scale = np.nanmean(0.41*bl['delta'][:,idx]*bl['ustarwc_sg17'][idx], axis = 1)
nu_scale = np.nanmean(10*0.41*bl['omega'][idx]*(bl['delta'][:,idx])**2, axis = 1)

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
ax1.plot(phasebins, nu_scale, color = color)
ax1.set_ylabel(r'$10 \kappa \delta^2 \omega$ (m$^2$ s$^{-1}$)', color = color)
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
ax1.tick_params(axis='y', labelcolor=color)

color = 'C1'
ax2 = ax1.twinx()
ax2.plot(phasebins, nut, color = color)
ax2.set_ylabel(r'$\nu_T = C_\mu k^2 \epsilon^{-1}$ (m$^2$ s$^{-1}$)', color = color)
ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
ax2.tick_params(axis='y', labelcolor=color)

plt.xticks(ticks = phasebins, labels = phaselabels)

fig.set_size_inches(8,5)
fig.tight_layout(pad = 0.5)
plt.rcParams.update(params)

#%% Wave TKE-based viscosity vs delta
kwave = np.array([np.nanmean(scale*np.trapz(data['tke_wave'][:,idx,i]*intmask,
                                  np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )

nut = 0.09*kwave**2/epsilon
fig, ax1 = plt.subplots()

color = 'C0'
ax1.plot(phasebins, nu_scale, color = color)
ax1.set_ylabel(r'$\kappa u_* \delta$ (m$^2$ s$^{-1}$)', color = color)
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
ax1.tick_params(axis='y', labelcolor=color)

color = 'C1'
ax2 = ax1.twinx()
ax2.plot(phasebins, nut, color = color)
ax2.set_ylabel(r'$\tilde{\nu_T}$ (m$^2$ s$^{-2}$)', color = color)
ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
ax2.tick_params(axis='y', labelcolor=color)

plt.xticks(ticks = phasebins, labels = phaselabels)

fig.set_size_inches(8,5)
fig.tight_layout(pad = 0.5)
plt.rcParams.update(params)


#%% Traditional normal old eddy viscosity

nut_temp = data['uw'][:,idx,:]/dubardz.reshape(*dubardz.shape,-1)

nut = np.array([np.nanmean(scale*np.trapz(nut_temp[:,:,i]*intmask,
                                  np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )


#Negative values...not so good...

#%% Now with vertical wave shear

nut = data['uw'][:,idx,:]/data['dudz'][:,idx,:]

nut = np.array([np.nanmean(scale*np.trapz(nut[:,:,i]*intmask,
                                  np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )


#%% and wave stress

nut = data['uw_wave'][:,idx,:]/dubardz.reshape(*dubardz.shape,-1)

nut = np.array([np.nanmean(scale*np.trapz(nut[:,:,i]*intmask,
                                  np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )

#%% and wave stress with vertical wave shear

nut = data['uw_wave'][:,idx,:]/data['dudz'][:,idx,:]

nut = np.array([np.nanmean(scale*np.trapz(nut[:,:,i]*intmask,
                                  np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )

