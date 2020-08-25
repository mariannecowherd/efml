#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:30:51 2020

@author: marianne

@title: tkeplot
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

#Loading stress
stress = np.load('/Users/Marianne/Documents/GitHub/efml/phase_stress.npy', allow_pickle = True).item()
blparams = np.load('/Users/Marianne/Documents/GitHub/efml/blparams.npy',allow_pickle=True).item()
delta = blparams['delta']
tke = stress['tke']
tke_wave = stress['tke_wave']


z = stress['z']
ubvec = blparams['ubvec']

#there are a couple weird ones
z[z > 10] = np.nan
tke_wave = tke_wave[~np.isnan(z)]
tke = tke[~np.isnan(z)]
z = z[~np.isnan(z)]

P = tke_wave.shape[1]

z = np.reshape(z,(30,382))
tke_wave = np.reshape(tke_wave,(30,382,8))
tke = np.reshape(tke,(30,382,8))

#interpolation
zinterp = np.linspace(0.001,0.015,15)
tke_wave_interp = np.zeros((len(zinterp),P))
tke_interp = np.zeros((len(zinterp),P))

for i in range(zinterp.size):
    
    idx = np.nanargmin(np.abs(z - zinterp[i]), axis = 0)    
    uz = np.zeros((len(idx),P))
    vz = np.zeros((len(idx),P))
    for j in range(len(idx)):
        u0 = ubvec[j]
        uz[j,:] = tke_wave[idx[j],j,:]/(u0**2)
        vz[j,:] = tke[idx[j],j,:]/(u0**2)
    
    tke_wave_interp[i,:] = np.nanmean(uz, axis = 0)
    tke_interp[i,:] = np.nanmean(vz, axis = 0)

znew = np.linspace(0.001, 0.015, 15)*100
dphi = np.pi/4 #Discretizing the phase
phasebins = np.arange(-3*np.pi/4, np.pi + dphi,dphi)

tkemesh = np.roll(tke_wave_interp, 1)
z_mesh, y_mesh = np.meshgrid(znew,phasebins)
levtke1 = np.linspace(np.nanmin(tkemesh),np.nanmax(tkemesh),20);
plt.figure(figsize=(15,10))
cf = plt.contourf(y_mesh, z_mesh, tkemesh.T, levtke1, extend='both')
cbar = plt.colorbar(cf, label=r'$\tilde{k}/ u_b^2$')
cbar.set_ticks(np.arange(0,np.nanmax(levtke1),5e-3))

phaselabels = [r'$-\frac{3\pi}{4}$',r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$', 
               r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',
               r'$\pi$']

plt.ylim(np.nanmin(znew),np.nanmax(znew))

delta_mean = np.nanmean(2*delta, axis = 1)
delta_ci = 1.96*np.nanstd(2*delta, axis = 1)/np.sqrt(delta.shape[1])

plt.errorbar(phasebins,np.roll(100*delta_mean,1), yerr = np.roll(delta_ci*100,1),
             fmt = 'o:',color= "white", capsize = 2)
plt.xticks(ticks = phasebins, labels = phaselabels)


plt.xlabel('wave phase', fontsize=16)
plt.ylabel('z (cmab)', fontsize=16)
plt.tight_layout()
plt.show()

plt.savefig('plots/wavetke.pdf',dpi=500)



tkemesh = np.roll(tke_interp, 1)
z_mesh, y_mesh = np.meshgrid(znew,phasebins)
levtke1 = np.linspace(np.nanmin(tkemesh),np.nanmax(tkemesh),20);
plt.figure(figsize=(15,10))
cf = plt.contourf(y_mesh, z_mesh, tkemesh.T, levtke1, extend='both')
cbar = plt.colorbar(cf, label=r'$\tilde{k}/ u_b^2$')
cbar.set_ticks(np.arange(0,np.nanmax(levtke1),5e-2))

delta_mean = np.nanmean(2*delta, axis = 1)
delta_ci = 1.96*np.nanstd(2*delta, axis = 1)/np.sqrt(delta.shape[1])

plt.errorbar(phasebins,np.roll(100*delta_mean,1), yerr = np.roll(delta_ci*100,1),
             fmt = 'o:',color= "white", capsize = 2)
plt.xticks(ticks = phasebins, labels = phaselabels)

plt.xlabel('wave phase', fontsize=16)
plt.ylabel('z (cmab)', fontsize=16)
plt.tight_layout()
plt.show()

plt.savefig('plots/tke.pdf',dpi=500)


