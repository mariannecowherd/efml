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
dudz = stress['dudz']

z = stress['z']
ubvec = blparams['ubvec']

#there are a couple weird ones
z[z > 10] = np.nan
uw_wave = uw_wave[~np.isnan(z)]
upwp = upwp[~np.isnan(z)]
dudz = dudz[~np.isnan(z)]
z = z[~np.isnan(z)]

P = uw_wave.shape[1]

z = np.reshape(z,(30,382))
uw_wave = np.reshape(uw_wave,(30,382,8))
upwp = np.reshape(upwp,(30,382,8))
dudz = np.reshape(dudz,(30,382,8))

#interpolation
zinterp = np.linspace(0.001,0.015,15)
uw_wave_interp = np.zeros((len(zinterp),P))
upwp_interp = np.zeros((len(zinterp),P))
dudz_interp = np.zeros((len(zinterp),P))

for i in range(zinterp.size):
    
    idx = np.nanargmin(np.abs(z - zinterp[i]), axis = 0)    
    uz = np.zeros((len(idx),P))
    up = np.zeros((len(idx),P))
    du = np.zeros((len(idx),P))
    for j in range(len(idx)):
        u0 = ubvec[j]
        uz[j,:] = uw_wave[idx[j],j,:]/(u0**2)
        up[j,:] = upwp[idx[j],j,:]/(u0**2)
        du[j,:] = dudz[idx[j],j,:]/u0
    
    uw_wave_interp[i,:] = np.nanmean(uz, axis = 0)
    upwp_interp[i,:] = np.nanmean(up, axis = 0)
    dudz_interp[i,:] = np.nanmean(du, axis = 0)

znew = np.linspace(0.001, 0.015, 15)*100
dphi = np.pi/4 #Discretizing the phase
phasebins = np.arange(-3*np.pi/4, np.pi + dphi,dphi)

uwmesh = -np.roll(uw_wave_interp, 1)
z_mesh, y_mesh = np.meshgrid(znew,phasebins)
levuw1 = np.linspace(np.nanmin(uwmesh),np.nanmax(uwmesh),20);
plt.figure(figsize=(15,10))
cf = plt.contourf(y_mesh, z_mesh, uwmesh.T, levuw1, extend='both')
cbar = plt.colorbar(cf, label=r'$-\overline{\tilde{u}\tilde{w}}/ u_b^2$')
cbar.set_ticks(np.arange(0,0.003,5e-4))

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

plt.savefig('plots/waveturb.pdf',dpi=500)

fix,ax=plt.subplots()
#calculate upwp/dudz for each theta
updumesh = np.abs(np.roll(upwp_interp/dudz_interp[5,:], 1))
z_mesh, y_mesh = np.meshgrid(znew,phasebins)
levud1 = np.linspace(np.nanmin(updumesh),np.nanmax(updumesh),20);
plt.figure(figsize=(15,10))
cf = plt.contourf(y_mesh, z_mesh, updumesh.T, levud1, extend='both')
cbar = plt.colorbar(cf, label=r"$-\overline{u'w'}/ du/dz$")
cbar.set_ticks(np.arange(0,0.003,5e-4))

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

nu_int = np.zeros(8)
for i in range(8):
    d = np.roll(delta_mean,1)
    zidx = zinterp<d[i]
    nu_int[i]=np.trapz(-uw_wave_interp[zidx,i],zinterp[zidx])
    
    
    




