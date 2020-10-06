#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:45:18 2020

@author: gegan
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import scipy.signal as sig

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

# nu_scale = np.nanmean(0.41*bl['delta'][:,idx]*bl['ustarwc_sg17'][idx], axis = 1)
nu_scale = np.nanmean(14.32*0.41*bl['omega'][idx]*(bl['delta'][:,idx])**2, axis = 1)
nu_scale_ci = 1.96*np.nanstd(14.32*0.41*bl['omega'][idx]*(bl['delta'][:,idx])**2, axis = 1)/np.sqrt(len(idx))

phaselabels = [r'$-\pi$', r'$-\frac{3\pi}{4}$',r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$', 
               r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$'
               ]

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

# timescale = data['tke']/data['epsilon']
# timescale[np.isnan(timescale)] = 0

# nut = 0.09*data['tke']**2/data['epsilon']
# nut[np.isnan(nut)] = 0
#%% nu_t from k-epsilon model vs delta
scale = (1/(np.sum(intmask,axis = 0)*0.001))

epsilon = np.array([np.nanmean( scale*np.trapz(data['epsilon'][:,idx,i]*intmask, 
                                         np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ])

epsilon_ci = np.array([np.nanstd( scale*np.trapz(data['epsilon'][:,idx,i]*intmask, 
                                         np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ])/np.sqrt(len(idx))

k = np.array([np.nanmean(scale*np.trapz(data['tke'][:,idx,i]*intmask,
                                  np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )

k_ci = np.array([np.nanstd(scale*np.trapz(data['tke'][:,idx,i]*intmask,
                                  np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )/np.sqrt(len(idx))
k2_ci = 2*k*k_ci

nut = 0.09*(k**2)/epsilon
nut_ci = nut*np.sqrt((epsilon_ci/epsilon)**2 + (k2_ci/(k**2))**2)

fig, ax1 = plt.subplots()

color = 'C0'
ax1.errorbar(phasebins, nu_scale, yerr = nu_scale_ci, fmt = 'o-', color = color, 
             capsize = 2, label = r'$C_{u_*} \kappa \delta^2 \omega$')
ax1.set_ylabel(r'$\nu_T$ (m$^2$ s$^{-1}$)')
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))

color = 'C1'
# ax2 = ax1.twinx()
ax1.errorbar(phasebins, nut, yerr = nut_ci, fmt = 'o-', color = color, 
         capsize = 2, label = r'$C_\mu k^2 \epsilon^{-1}$')
plt.xticks(ticks = phasebins, labels = phaselabels)
ax1.legend()

fig.set_size_inches(8,5)
fig.tight_layout(pad = 0.5)
plt.rcParams.update(params)
# plt.savefig('/Users/gegan/Desktop/OliverMeetings/8-27/k_epsilon_comparison.pdf')
#%% finding optimal phase shift

#spline interpolation
tck = interpolate.splrep(phasebins,nu_scale)
phasenew = np.linspace(-np.pi,3*np.pi/4,60)
nu_scale_int = interpolate.splev(phasenew,tck)

tck = interpolate.splrep(phasebins,nut)
nut_int = interpolate.splev(phasenew,tck)

corr = np.correlate(sig.detrend(nu_scale_int),sig.detrend(nut_int), mode = 'full')[len(nu_scale_int)-1:]
# corr = np.correlate(sig.detrend(nu_scale_int)/np.std(nu_scale_int),sig.detrend(nut_int)/np.std(nut_int), mode = 'full')[len(nu_scale_int)-1:]
lag_idx = corr.argmax() 
tlag = lag_idx*(phasenew[1] - phasenew[0])*(1/.36)/(2*np.pi)

#compared to estimate from d^2/nu_t = delta/kappa*ustar

t_scale = np.nanmean(delta/(np.nanmean(bl['ustarwc_meas'])))
t_turb = 0.09*np.nanmean(k/epsilon)

#Plotting figure with spline fits
fig, ax1 = plt.subplots()

color = 'C0'
ax1.errorbar(phasebins, nu_scale, yerr = nu_scale_ci, fmt = 'o', color = color, 
             capsize = 2, label = r'$C_{u_*} \kappa \delta^2 \omega$')
ax1.plot(phasenew,nu_scale_int,'-', color = color)
ax1.set_ylabel(r'$\nu_T$ (m$^2$ s$^{-1}$)')
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))

color = 'C1'
# ax2 = ax1.twinx()
ax1.errorbar(phasebins, nut, yerr = nut_ci, fmt = 'o', color = color, 
          capsize = 2, label = r'$C_\mu k^2 \epsilon^{-1}$')
ax1.plot(phasenew,nut_int,'-', color = color)
plt.xticks(ticks = phasebins, labels = phaselabels)


ax1.set_ylim(1e-4,3.3e-4)
ax1.legend(loc = 'upper right')


# ax1.annotate(s = '', xy = (-0.45,2.5e-4), xytext = (-0.45 + lag_idx*(phasenew[1] - phasenew[0]),2.5e-4),
#               arrowprops=dict(arrowstyle='<->'))
ax1.annotate(s = 'optimal lag = {:.2f} s'.format(tlag), xy = (-2,3.05e-4), 
              fontsize = 16)

ax1.annotate(s = r'$C_\mu k \epsilon^{-1} = $' + ' {:.2f} s'.format(t_turb), xy = (-2,2.8e-4), 
              fontsize = 16)

ax1.annotate(s = r'$\delta u_*^{-1} = $' + ' {:.2f} s'.format(t_scale), xy = (-2,2.6e-4), 
              fontsize = 16)


fig.set_size_inches(8,5)
fig.tight_layout(pad = 0.5)
plt.rcParams.update(params)
plt.savefig('/Users/gegan/Desktop/OliverMeetings/9-3/time_lags.pdf')