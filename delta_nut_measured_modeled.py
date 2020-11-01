#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:29:22 2020

@author: gegan
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import scipy.signal as sig
import netCDF4 as nc

import sys
sys.path.append('/Users/gegan/Documents/Python/Research/General')

from mylib import naninterp

waveturb = np.load('data/waveturb.npy', allow_pickle = True).item()
bl = np.load('data/blparams.npy', allow_pickle = True).item() 
phase_data = np.load('data/phase_stress.npy', allow_pickle = True).item()
gotm_data = nc.Dataset('data/combined_wbbl_01.nc', mode = 'r')

# phase_data = dict()
# phase_data['epsilon'] = ke['eps']
# phase_data['tke'] = ke['tke']
# phase_data['z'] = phase_data_temp['z']

idx = list(waveturb.keys())
del idx[292]
del idx[203]

dphi = np.pi/4 #Discretizing the phase
phasebins = np.arange(-np.pi,np.pi,dphi)
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
        # umax = np.nanmax(uprof_int)
        # idxmax = np.nanargmax(uprof_int)
        
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

z = np.linspace(0,1.5e-2,30)

u = np.zeros((len(z),utemp.shape[1]))
nut_model = np.zeros((len(z),nuttemp.shape[1]))

for i in range(u.shape[1]):
    u[:,i] = np.interp(z,ztemp[:,i],utemp[:,i])
    nut_model[:,i] = np.interp(z, ztemp[:,i], nuttemp[:,i])


f_low = 0.32
f_high = 0.345
fs = 100

b,a = sig.butter(2,[f_low/(fs/2),f_high/(fs/2)], btype = 'bandpass')
ufilt = sig.filtfilt(b,a,u, axis = 1) 


up = u - np.nanmean(u, axis = 1, keepdims = True)
ubar = np.nanmean(ufilt, axis = 0)

del utemp, ztemp, nuttemp

#%%
nut_bar = nut_model[0,:]
# nut_bar = (nut_model[6,:] + nut_model[7,:])/2
# nut_bar = np.nanmean(nut_model[5:15,:], axis  = 0)

#Filtering in spectral space 

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
    nu_scale_model[jj] = np.mean(kappa * (2*np.pi/3) * delta_full[idx1]**2)
    nu_scale_ci_model[jj] = 1.96*np.std(kappa * (2*np.pi/3) * delta_full[idx1]**2)/np.sqrt(np.sum(idx1))
    tke_phase_model[jj] = np.mean(tke_model[:20,idx1])
    eps_phase_model[jj] = np.mean(eps_model[:20,idx1])
    delta_model[jj] = np.mean(delta_full[idx1])
    nut_const_model[jj] = 0.09*np.mean(tke_model[0,idx1])**2/np.nanmean(eps_model[0,:])
# delta_model = displacement_thickness(uprof_model,z)

#%% Measured boundary layer thickness scaling

delta = np.nanmean(2*bl['delta'][:,idx],axis = 1)
# c1_ustar = 1
# c1_ustar = 0.0971
c1_ustar = .2
nu_scale = np.nanmean((1/c1_ustar)*0.41*bl['omega'][idx]*(2*bl['delta'][:,idx])**2, axis = 1)
nu_scale_ci = 1.96*np.nanstd((1/c1_ustar)*0.41*bl['omega'][idx]*(2*bl['delta'][:,idx])**2, axis = 1)/np.sqrt(len(idx))

# masking k and epsilon
intmask = ((phase_data['z'][:,idx] < 0.0105) & (phase_data['z'][:,idx] > 0.0025)).astype(int)

phase_data['epsilon'][np.isnan(phase_data['epsilon'])] = 0
phase_data['tke'][np.isnan(phase_data['tke'])] = 0

# nu_t from k-epsilon model 
scale = (1/(np.sum(intmask,axis = 0)*0.001))

#Old method with bad epsilon
# epsilon = np.array([np.nanmean( scale*np.trapz(phase_data['epsilon'][:,idx,i]*intmask, 
#                                          np.flipud(phase_data['z'][:,idx]), axis = 0)) for i in range(8) ])



# epsilon_std = np.array([np.nanstd( scale*np.trapz(phase_data['epsilon'][:,idx,i]*intmask, 
#                                          np.flipud(phase_data['z'][:,idx]), axis = 0)) for i in range(8) ])

#New method with constant epsilon
epsilon_raw = np.load('data/epsilon.npy', allow_pickle = True).item()['epsilon']

epsilon = np.array([np.nanmean( scale*np.trapz(epsilon_raw[:,idx]*intmask, 
                                          np.flipud(phase_data['z'][:,idx]), axis = 0)) for i in range(8) ])

epsilon_std = np.array([np.nanstd( scale*np.trapz(epsilon_raw[:,idx]*intmask, 
                                          np.flipud(phase_data['z'][:,idx]), axis = 0)) for i in range(8) ])

# #New method with phase varying structure function epsilon
# epsilon_raw = np.load('data/eps_sf.npy', allow_pickle = True)

# epsilon = np.nanmean(epsilon_raw[idx,:], axis = 0)

# epsilon_std =  np.nanstd(epsilon_raw[idx,:], axis = 0)


k = np.array([np.nanmean(scale*np.trapz(phase_data['tke'][:,idx,i]*intmask,
                                  np.flipud(phase_data['z'][:,idx]), axis = 0)) for i in range(8) ] )

k_std = np.array([np.nanstd(scale*np.trapz(phase_data['tke'][:,idx,i]*intmask,
                                  np.flipud(phase_data['z'][:,idx]), axis = 0)) for i in range(8) ] )
k2_std = 2*k*k_std

cov_k_eps = np.array([np.cov(naninterp(scale*np.trapz((phase_data['tke'][:,idx,i]**2)*intmask,
                                  np.flipud(phase_data['z'][:,idx]), axis = 0)), 
                                        naninterp(scale*np.trapz(phase_data['epsilon'][:,idx,i]*intmask, 
                                         np.flipud(phase_data['z'][:,idx]), axis = 0)))[0,1] for i in range(8) ] )
nut = 0.09*(k**2)/epsilon
nut_std = nut*np.sqrt((epsilon_std/epsilon)**2 + (k2_std/(k**2))**2 )
nut_ci = 1.96*nut_std/np.sqrt(len(idx))

# finding optimal phase shift

#spline interpolation
tck = interpolate.splrep(phasebins,nu_scale)
phasenew = np.linspace(-np.pi,3*np.pi/4,200)
nu_scale_int = interpolate.splev(phasenew,tck)

tck = interpolate.splrep(phasebins,nut)
nut_int = interpolate.splev(phasenew,tck)

corr = np.correlate(sig.detrend(nu_scale_int),sig.detrend(nut_int), mode = 'full')[len(nu_scale_int)-1:]
# corr = np.correlate(sig.detrend(nu_scale_int)/np.std(nu_scale_int),sig.detrend(nut_int)/np.std(nut_int), mode = 'full')[len(nu_scale_int)-1:]
lag_idx = corr.argmax() 
tlag = lag_idx*(phasenew[1] - phasenew[0])*(1/.36)/(2*np.pi)

#compared to estimate from d^2/nu_t = delta/kappa*ustar
t_turb = 0.09*np.nanmean(k/epsilon)

#%%Plotting figure with spline fits
fig, (ax1, ax2) = plt.subplots(1,2)

color = '0.0'
ax1.errorbar(phasebins, nu_scale, yerr = nu_scale_ci, fmt = 'o', color = color, 
             capsize = 2, label = r'$\nu_\delta = \frac{\kappa}{C_1} \delta^2 \omega$')
ax1.plot(phasenew,nu_scale_int,'-', color = color)
ax1.set_ylabel(r'$\nu_T$ (m$^2$ s$^{-1}$)')
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))

color = '0.7'
# ax2 = ax1.twinx()
ax1.errorbar(phasebins, nut, yerr = nut_ci, fmt = 'o', color = color, 
          capsize = 2, label = r'$\nu_{k \varepsilon} = C_\mu k^2 \langle\varepsilon\rangle^{-1}$')
ax1.plot(phasenew,nut_int,'-', color = color)

ax1.set_ylim(1.4e-4,4.8e-4)
ax1.legend(loc = 'upper right')
ax1.set_title('(a)')


# ax1.annotate(s = '', xy = (-0.45,2.5e-4), xytext = (-0.45 + lag_idx*(phasenew[1] - phasenew[0]),2.5e-4),
#               arrowprops=dict(arrowstyle='<->'))
ax1.annotate(s = 'optimal lag = {:.2f} s'.format(tlag), 
              xy = (0.1,0.925), xycoords = 'axes fraction',fontsize = 16)

ax1.annotate(s = r'$C_\mu k \epsilon^{-1} = $' + ' {:.2f} s'.format(t_turb), 
              xy = (0.1,0.85), xycoords = 'axes fraction', fontsize = 16)

ax1.set_xticks(phasebins)
ax1.set_xticklabels(phaselabels)
# ax1.annotate(s = r'$\delta u_*^{-1} = $' + ' {:.2f} s'.format(t_scale), xy = (-2,2.6e-4), 
#               fontsize = 16)

#Now for model
#spline interpolation
tck = interpolate.splrep(phasebins,nu_scale_model)
phasenew = np.linspace(-np.pi,3*np.pi/4,200)
nu_scale_int = interpolate.splev(phasenew,tck)

tck = interpolate.splrep(phasebins,nut_phase_model)
nut_int = interpolate.splev(phasenew,tck)

tck = interpolate.splrep(phasebins, nut_const_model)
nut_const_int = interpolate.splev(phasenew, tck)

corr = np.correlate(sig.detrend(nu_scale_int),sig.detrend(nut_int), mode = 'full')[len(nu_scale_int)-1:]
# corr = np.correlate(sig.detrend(nu_scale_int)/np.std(nu_scale_int),sig.detrend(nut_int)/np.std(nut_int), mode = 'full')[len(nu_scale_int)-1:]
lag_idx = corr.argmax() 
tlag = lag_idx*(phasenew[1] - phasenew[0])*(1/.333)/(2*np.pi)

#compared to estimate from d^2/nu_t = delta/kappa*ustar
# t_turb = 0.09*np.nanmean(tke_model[5:15,:]/eps_model[5:15,:])
t_turb = 0.09*np.nanmean(tke_model[0,:]/eps_model[0,:])

#Finding optimal scale factor
alpha = np.linspace(1,10,1000)
errors = np.zeros_like(alpha)

for i,j in enumerate(alpha):
    errors[i] = np.linalg.norm(np.roll(nut_int,lag_idx) - j*nu_scale_int)
alpha_opt = alpha[np.argmin(errors)]

alpha_opt = 1
color = '0.0'
ax2.errorbar(phasebins, alpha_opt*nu_scale_model, yerr = nu_scale_ci_model, fmt = 'o', color = color, 
             capsize = 2, label = r'$\kappa \delta^2 \omega$')
ax2.plot(phasenew,alpha_opt*nu_scale_int,'-', color = color)
ax2.set_ylabel(r'$\nu_T$ (m$^2$ s$^{-1}$)')
ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))

color = '0.7'
# ax2 = ax1.twinx()
ax2.errorbar(phasebins, nut_phase_model, yerr = nut_ci_model, fmt = 'o', color = color, 
          capsize = 2, label = r'$\nu_{k \varepsilon} = C_\mu k^2 \varepsilon^{-1}$')

ax2.plot(phasebins, nut_const_model, 'o', color = color)
ax2.plot(phasenew, nut_const_int, ':', color = color,
         label = r'$\nu_{k \varepsilon} = C_\mu k^2 \langle \varepsilon \rangle^{-1}$')
ax2.plot(phasenew,nut_int,'-', color = color)

ax2.set_xticks(phasebins)
ax2.set_xticklabels(phaselabels)
ax2.legend(loc = 'upper right')
ax2.set_title('(b)')
ax2.set_ylim(-.2e-5,1.75e-5)


# ax1.annotate(s = '', xy = (-0.45,2.5e-4), xytext = (-0.45 + lag_idx*(phasenew[1] - phasenew[0]),2.5e-4),
#               arrowprops=dict(arrowstyle='<->'))
ax2.annotate(s = 'optimal lag = {:.2f} s'.format(tlag), xy = (0.1,0.925), 
             xycoords = 'axes fraction', fontsize = 16)

ax2.annotate(s = r'$C_\mu k \epsilon^{-1} = $' + ' {:.3f} s'.format(t_turb),
             xy = (0.1,0.85), xycoords = 'axes fraction', fontsize = 16)


fig.set_size_inches(13,6)
fig.tight_layout(pad = 0.5)
plt.rcParams.update(params)
plt.savefig('plots/delta_nut_comparison.pdf')