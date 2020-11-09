#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:40:58 2020

@author: gegan
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from scipy.optimize import curve_fit
from scipy import interpolate
import scipy.special as sc
import vectrinofuncs as vfs

import warnings
warnings.filterwarnings("ignore")

profs = np.load('data/phaseprofiles_alt.npy')
stress = np.load('data/phase_stress_alt.npy', allow_pickle=True).item()
bl = np.load('data/blparams_alt.npy', allow_pickle = True).item()
z = stress['z']
phasebins = bl['phasebins']
omega = bl['omega']
ub = bl['ubvec']

#%% ensemble velocity profiles
vel_interp = np.zeros([384,15,8]) #initialize output
velidx=[]
znew = np.linspace(0.001, 0.015, 15)

burstnums = list(range(384))
#ensemble average, normalize by ubvec
for n in burstnums:
    for i in range(8):
        try:
            vel_old = profs[:,n,i]/ub[n]
            zold = z[:,n].flatten()
            zold,vel_old = vfs.nanrm2(zold,vel_old)
            f_vel = interpolate.interp1d(zold,vfs.naninterp(vel_old),kind='cubic')
            vel_interp[n,:,i] = (f_vel(znew))
            velidx.append(n)
        except ValueError:
            continue

velidx=np.unique(velidx)

vel_ens = np.sqrt(2)*np.nanmean(vel_interp[velidx,:,:],axis=0)


#%% defining gm function
def make_gm_offset(omega, kb, u0, offset):
    def gm(z, ustar):
        kappa = 0.41
        l = kappa*ustar/omega
        zeta = (z - offset)/l
        zeta0 = kb/(30*l)
        
        uw  = u0*(1 -((sc.ker(2*np.sqrt(zeta)) + 1j*sc.kei(2*np.sqrt(zeta)))/
                      (sc.ker(2*np.sqrt(zeta0)) + 1j*sc.kei(2*np.sqrt(zeta0)))))
        
        return uw.real
    
    return gm

#%% fitting gm function
kb = 0.01
omega = np.nanmean(omega)
tvec = np.arange(-np.pi, np.pi, np.pi/4)/omega

uinf = 0.86*np.exp(1j*tvec*omega)
offset = 0.0025

ustar = np.zeros((8,))
u0 = np.zeros((8,))
r2 = np.zeros((8,))

for i in range(8):
    
    popt, pcov = curve_fit(make_gm_offset(omega,kb,uinf[i],offset),znew[3:-3],vel_ens[3:-3,i],
                                          p0 = 1e-2, bounds = (1e-4, 1e-1))
    
    ustar[i] = popt[0]

#%%  Plotting
fig, ax = plt.subplots()

for i in range(8):
    
    ax.plot(vel_ens[:,i], znew[:], '-', color = 'C' + str(i))
    
    zint = np.linspace(0.001, 0.015, 100)
    ax.plot(make_gm_offset(omega,kb,uinf[i],offset)(zint,ustar[i])[14:], zint[14:], '--', 
            color = 'C' + str(i))

fig.set_size_inches(8,6)
fig.tight_layout(pad = 0.5)
