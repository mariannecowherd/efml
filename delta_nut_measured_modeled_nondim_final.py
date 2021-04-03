#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 08:41:15 2020

@author: gegan
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import interpolate
import scipy.signal as sig
import netCDF4 as nc


import sys
sys.path.append('/Users/gegan/Documents/Python/Research/General')

from mylib import naninterp

params = {
   'axes.labelsize': 28,
   'font.size': 28,
   'legend.fontsize': 16,
   'xtick.labelsize': 28,
   'ytick.labelsize': 28,
   'text.usetex': True,
   'font.family': 'serif'
   }

waveturb = np.load('data/waveturb.npy', allow_pickle = True).item()
bl = np.load('data/blparams_alt.npy', allow_pickle = True).item() 
phase_data = np.load('data/phase_stress_alt.npy', allow_pickle = True).item()
gotm_data = nc.Dataset('data/combined_wbbl_01.nc', mode = 'r')

idx = list(waveturb.keys())
del idx[292]
del idx[203]

dphi = np.pi/4 #Discretizing the phase
phasebins = np.arange(-np.pi,np.pi,dphi)
phaselabels = [r'$-\pi$', r'$-\frac{3\pi}{4}$',r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$', 
               r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$'
               ]


def displacement_thickness(uprof,z):
    """Calculates a modified displacement thickness for phase resolved boundary layer
    velocity profiles. 
    
    uprof is an n x p array where n is the number of vertical
    measurement bins, and p is the number of phases. 
    
    z is the vertical coordinate and is an n x 1 array. 
    
    
    """
    n,p = uprof.shape
    delta = np.zeros((p,))
    for i in range(p):
        int_range = ((z > 0) & (z < 1e-2))
        z_int = z[int_range]
        
        uprof_int = uprof[int_range,i]
        max_range = ((z > 0) & (z < 1e-3))
        if np.nanmean(uprof[max_range,i]) < 0:
            umax = np.nanmin(uprof_int)
            idxmax = np.nanargmin(uprof_int)
        else:
            umax = np.nanmax(uprof_int)
            idxmax = np.nanargmax(uprof_int)
        
        delta[i] = np.trapz(1 - uprof_int[:idxmax]/umax,z_int[:idxmax])
    
    
    return delta 

#%% Modeled data
tidx = list(range(1000,150000))

nut = (gotm_data.variables['num'][tidx,1:,0,0] + gotm_data.variables['num'][tidx,:-1,0,0])/2
nuttemp = nut.T
utemp = gotm_data.variables['u'][tidx,:,0,0].T
ztemp = gotm_data.variables['z'][tidx,:,0,0].T + 2.5

eps = (gotm_data.variables['eps'][tidx,:-1,0,0] + gotm_data.variables['eps'][tidx,1:,0,0])/2
eps_model = eps.T

tke = (gotm_data.variables['tke'][tidx,:-1,0,0] + gotm_data.variables['tke'][tidx,1:,0,0])/2
tke_model = tke.T 


time = gotm_data.variables['time'][tidx]

z = np.linspace(1e-4,1.5e-2,30)

u = np.zeros((len(z),utemp.shape[1]))
nut_model = np.zeros((len(z),nuttemp.shape[1]))

for i in range(u.shape[1]):
    u[:,i] = np.interp(z,ztemp[:,i],utemp[:,i])
    nut_model[:,i] = np.interp(z, ztemp[:,i], nuttemp[:,i])

#nondimensionalizing
ustar = 0.01
nut_model = nut_model/(0.41*ustar*z.reshape(-1,1))

#Filtering at wave frequencies
f_low = 0.32
f_high = 0.345
fs = 100

b,a = sig.butter(2,[f_low/(fs/2),f_high/(fs/2)], btype = 'bandpass')
ufilt = sig.filtfilt(b,a,u, axis = 1) 


up = u - np.nanmean(u, axis = 1, keepdims = True)
ubar = np.nanmean(ufilt[-10:,:], axis = 0)

#%%
nut_bar = nut_model[0,:]

#calculate analytic signal based on de-meaned and low pass filtered velocity
hu = sig.hilbert(ubar - np.nanmean(ubar)) 

#Phase based on analytic signal
p = np.arctan2(hu.imag,hu.real) 

delta_full = displacement_thickness(up,z)

uprof_model = np.zeros((z.shape[0],len(phasebins)))
nutprof_model = np.zeros((z.shape[0],len(phasebins)))
nut_phase_model = np.zeros((len(phasebins),))
nut_ci_model = np.zeros((len(phasebins),))
nu_scale_model = np.zeros((len(phasebins),))
nu_scale_ci_model = np.zeros((len(phasebins),))
tke_phase_model = np.zeros((len(phasebins),))
eps_phase_model = np.zeros((len(phasebins),))
delta_model = np.zeros((len(phasebins),))

nut_const_model =  np.zeros((len(phasebins),))
nut_const_ci_model = np.zeros((len(phasebins),))

kappa = 0.41

for jj in range(len(phasebins)):
       
    if jj == 0:
        #For -pi
        idx1 = ( (p >= phasebins[-1] + (dphi/2)) | (p <= phasebins[0] + (dphi/2))) #Measured
    else:
        #For phases in the middle
        idx1 = ((p >= phasebins[jj]-(dphi/2)) & (p <= phasebins[jj]+(dphi/2))) #measured
   
    uprof_model[:,jj] = np.mean(up[:,idx1], axis = 1) #Averaging over the indices for this phase bin for measurement
    nut_phase_model[jj] = np.mean(nut_bar[idx1])
    nut_ci_model[jj] = 1.96*np.std(nut_bar[idx1])/np.sqrt(np.sum(idx1))
    nu_scale_model[jj] = np.mean(kappa * (2*np.pi/3) * delta_full[idx1]**2)//(0.41*ustar*z[0])
    nu_scale_ci_model[jj] = 1.96*np.std(kappa * (2*np.pi/3) * delta_full[idx1]**2)/np.sqrt(np.sum(idx1))
    tke_phase_model[jj] = np.mean(tke_model[:20,idx1])
    eps_phase_model[jj] = np.mean(eps_model[:20,idx1])
    delta_model[jj] = np.mean(delta_full[idx1])
    nut_const_model[jj] = (0.09*np.mean(tke_model[0,idx1])**2/np.nanmean(eps_model[0,:]))/(0.41*ustar*z[0])

#%% Measured boundary layer thickness scaling
delta = np.nanmean(2*bl['delta'][:,idx],axis = 1)
c1_ustar = .14

#Scaled eddy viscosity, nondimensionalized by vertically-averaged ustar-based eddy viscosity
nu_scale = np.nanmean((1/c1_ustar)*0.41*bl['omega'][idx]*(2*bl['delta'][:,idx])**2/(0.41*bl['ustarwc_meas'][idx]*.0075), axis = 1)
nu_scale_ci = 1.96*np.nanstd((1/c1_ustar)*0.41*bl['omega'][idx]*(2*bl['delta'][:,idx])**2/(0.41*bl['ustarwc_meas'][idx]*.0075), axis = 1)/np.sqrt(len(idx))

# masking k and epsilon
intmask = ((phase_data['z'][:,idx] < 0.0125) & (phase_data['z'][:,idx] > 0.0035)).astype(int)

phase_data['epsilon'][np.isnan(phase_data['epsilon'])] = 0
phase_data['tke'][np.isnan(phase_data['tke'])] = 0

# nu_t from k-epsilon model 
scale = (1/(np.sum(intmask,axis = 0)*0.001))

#Using constant epsilon
epsilon_raw = np.load('data/epsilon.npy', allow_pickle = True).item()['epsilon']

epsilon = np.array([np.nanmean( scale*np.trapz(epsilon_raw[:,idx]*intmask, 
                                          np.flipud(phase_data['z'][:,idx]), axis = 0)) for i in range(8) ])

#And phase-varying TKE
k = np.array([np.nanmean(scale*np.trapz(phase_data['tke'][:,idx,i]*intmask,
                                  np.flipud(phase_data['z'][:,idx]), axis = 0)) for i in range(8) ] )

#Calculating eddy viscosity
nut_temp = np.zeros((len(idx),8))

for i in range(8):
    for n, j in enumerate(idx):
        zidx = ((phase_data['z'][:,j] > 0.0035) & (phase_data['z'][:,j] < 0.0125))
        nut_temp[n,i] = np.nanmean((0.09*phase_data['tke'][zidx,j,i]**2/epsilon_raw[zidx,j])/(0.41*bl['ustarwc_meas'][j]*phase_data['z'][zidx,j]))

nut_final = np.nanmean(nut_temp, axis = 0)
nut_ci = np.nanstd(nut_temp, axis = 0)/np.sqrt(len(idx))

# finding optimal phase shift via interpolation
tck = interpolate.splrep(phasebins,nu_scale)
phasenew = np.linspace(-np.pi,3*np.pi/4,8)
nu_scale_int = interpolate.splev(phasenew,tck)

tck = interpolate.splrep(phasebins,nut_final)
nut_int = interpolate.splev(phasenew,tck)

#ustar/kz and confidence interval from GM fit
us_fit = np.array([1999.41906942, 1462.57352136,  435.11395888, 2724.29070851, 2217.7746008,
 1484.30191751,  461.5108289,  2742.90226235])
us_ci = np.array([345.94037492, 270.73286049,  63.1326758,  555.12422574, 414.29621952,
 251.94720063,  72.77431675, 561.9341212, ])

nu_fit = 0.41*us_fit*(np.nanmean(np.arange(0.0035,.0125,.001)))
nu_fit_ci = 0.41*us_ci*(np.nanmean(np.arange(0.0035,.0125,.001)))

tck = interpolate.splrep(phasebins,nu_fit)
nu_fit_int = interpolate.splev(phasenew,tck)

corr = np.correlate(sig.detrend(nu_scale_int),sig.detrend(nut_int), mode = 'full')[len(nu_scale_int)-1:]
lag_idx = corr.argmax() 
tlag = lag_idx*(phasenew[1] - phasenew[0])*(1/.36)/(2*np.pi)

#Turbulence timescale 
t_turb = 0.09*np.nanmean(k/epsilon)

phase_lag = t_turb/(1/.36)

#%%Plotting figure 
fig, (ax1, ax2) = plt.subplots(1,2)

color = '0.0'


ax1.errorbar(phasebins, nu_fit, yerr = nu_fit_ci, fmt = 'o', color = '0.3',
             capsize = 2)
l1 = ax1.plot(phasenew, nu_fit_int, '--', color = '0.3', label = r'$\nu_*$')

ax1.set_xticks(phasebins)
ax1.set_xticklabels(phaselabels)

ax1.errorbar(phasebins, nu_scale, yerr = nu_scale_ci, fmt = 'o', color = color, 
             capsize = 2)
l2 = ax1.plot(phasenew,nu_scale_int,'-', color = color, label = r'$\nu_\delta$')
ax1.set_ylabel(r'$\nu_T (\kappa u_{*m} z)^{-1}$ ')

color = '0.7'
ax1.errorbar(phasebins, nut_final, yerr = nut_ci, fmt = 'o', color = color, 
          capsize = 2)
l3 = ax1.plot(phasenew,nut_int,'-', color = color, label = r'$\nu_{k \langle \varepsilon \rangle}$')

ax1.set_title('(a)')
ax1.annotate(s = r'optimal lag = $\frac{\pi}{4}$', 
              xy = (0.3,0.925), xycoords = 'axes fraction',fontsize = 16)
ax1.annotate(s = r'$\langle C_\mu k \langle \varepsilon \rangle^{-1} \rangle = \frac{3 \pi}{14}$', 
              xy = (0.3,0.825), xycoords = 'axes fraction', fontsize = 16)

ax1.set_ylim(-2, 40)
lns = l1+l2+l3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left')

#Now for model - interpolation (just linear for now)
tck = interpolate.splrep(phasebins,nu_scale_model)
phasenew = np.linspace(-np.pi,3*np.pi/4,8)
nu_scale_int = interpolate.splev(phasenew,tck)

tck = interpolate.splrep(phasebins,nut_phase_model)
nut_int = interpolate.splev(phasenew,tck)

tck = interpolate.splrep(phasebins, nut_const_model)
nut_const_int = interpolate.splev(phasenew, tck)

corr = np.correlate(sig.detrend(nu_scale_int),sig.detrend(nut_int), mode = 'full')[len(nu_scale_int)-1:]
lag_idx = corr.argmax() 
tlag = lag_idx*(phasenew[1] - phasenew[0])*(1/.333)/(2*np.pi)

#compared to estimate from d^2/nu_t = delta/kappa*ustar
t_turb = 0.09*np.nanmean(tke_model[5:15,:]/eps_model[5:15,:])
t_turb = 0.09*np.nanmean(tke_model[0,:]/eps_model[0,:])

color = '0.0'
ax2.errorbar(phasebins, nu_scale_model, yerr = nu_scale_ci_model, fmt = 'o', color = color, 
             capsize = 2)
ax2.plot(phasenew, nu_scale_int,'-', color = color, label = r'$\kappa \delta^2 \omega$')
ax2.set_ylabel(r'$\nu_T (\kappa u_{*m} z)^{-1}$')

color = '0.7'
ax2.errorbar(phasebins, nut_phase_model, yerr = nut_ci_model, fmt = 'o', color = color, 
          capsize = 2)
ax2.plot(phasenew,nut_int,':', color = color, label = r'$\nu_{k \varepsilon}$')

ax2.plot(phasebins, nut_const_model, 'o', color = color)

ax2.plot(phasenew, nut_const_int, '-', color = color,
         label = r'$\nu_{k \langle \varepsilon \rangle }$')

ax2.set_xticks(phasebins)
ax2.set_xticklabels(phaselabels)
ax2.legend(loc = 'upper left')
ax2.set_title('(b)')
ax2.set_ylim(-2, 40)

ax2.annotate(s = r'optimal lag = $\frac{\pi}{4}$',  xy = (0.3,0.925), 
             xycoords = 'axes fraction', fontsize = 16)

ax2.annotate(s = r'$\langle C_\mu k \langle \varepsilon \rangle^{-1} \rangle \approx \frac{\pi}{500}$',
             xy = (0.3,0.85), xycoords = 'axes fraction', fontsize = 16)


fig.set_size_inches(13,6)
fig.tight_layout(pad = 0.5)
plt.rcParams.update(params)