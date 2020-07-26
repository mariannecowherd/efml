#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 14:56:51 2020

@author: marianne, gegan

@title: waveturbplot
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
uw_wave = stress['uw_wave']
upwp = stress['uw']


z = stress['z']
ubvec = blparams['ubvec']

#there are a couple weird ones
z[z > 10] = np.nan
uw_wave = uw_wave[~np.isnan(z)]
z = z[~np.isnan(z)]

P = uw_wave.shape[1]

z = np.reshape(z,(30,382))
uw_wave = np.reshape(uw_wave,(30,382,8))

#interpolation
zinterp = np.linspace(0.001,0.015,15)
uw_wave_interp = np.zeros((len(zinterp),P))

for i in range(zinterp.size):
    
    idx = np.nanargmin(np.abs(z - zinterp[i]), axis = 0)    
    uz = np.zeros((len(idx),P))
    for j in range(len(idx)):
        u0 = ubvec[j]
        uz[j,:] = uw_wave[idx[j],j,:]/(u0**2)
    
    uw_wave_interp[i,:] = np.nanmean(uz, axis = 0)

znew = np.linspace(0.001, 0.015, 15)
dphi = np.pi/4 #Discretizing the phase
phasebins = np.arange(-np.pi,np.pi,dphi)

uwmesh = -uw_wave_interp 
z_mesh, y_mesh = np.meshgrid(znew,phasebins)
levuw1 = np.linspace(np.nanmin(uwmesh),np.nanmax(uwmesh),20);
plt.figure(figsize=(15,10))
cf = plt.contourf(y_mesh, z_mesh, uwmesh.T, levuw1, extend='both')
plt.colorbar(cf, label=r'$-\overline{\tilde{u}\tilde{w}}/ u_b^2$')


phaselabels = [r'$-\pi$',r'$-\frac{3\pi}{4}$',r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$', 
               r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$' ]

plt.ylim(np.nanmin(znew),np.nanmax(znew))

plt.plot(phasebins,np.nanmean(delta,axis=1),'o:',color="white")
plt.xticks(ticks = phasebins, labels = phaselabels)


plt.xlabel('wave phase', fontsize=16)
plt.ylabel('z', fontsize=16)
plt.tight_layout()
plt.show()

plt.savefig('waveturb.pdf',dpi=500)
