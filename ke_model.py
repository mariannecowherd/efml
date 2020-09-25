#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:41:58 2020

@author: marianne

@title: k-epsilon model
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vectrinofuncs import contour_interp

#plot style
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

stress = np.load('/Users/Marianne/Documents/GitHub/efml/phase_stress.npy', allow_pickle = True).item()
blparams = np.load('/Users/Marianne/Documents/GitHub/efml/blparams.npy',allow_pickle=True).item()

dudz = stress['dudz'] #(30, 382, 8)
upwp = stress['uw'] #(30, 382, 8)
z = stress['z'] #"the stress" but in a french accent
delta = blparams['delta']
ubvec = blparams['ubvec']
k = stress['tke_wave']
epsilon = stress['epsilon']
dudz = stress['dudz']


Cmu = 0.1 #central michigan university

nu_ke = Cmu * k**2 / epsilon * dudz**2 #(30,384,2) cukes not nukes

norm = ubvec
nuke_interp = contour_interp(nu_ke,z,norm)a

#prepare the contour plot
znew = np.linspace(0.001, 0.015, 15)*100
dphi = np.pi/4 #Discretizing the phase
phasebins = np.arange(-3*np.pi/4, np.pi + dphi,dphi)

mesh = np.roll(nuke_interp, 1) #just chang this line
z_mesh, y_mesh = np.meshgrid(znew,phasebins)
lev1 = np.linspace(np.nanmin(mesh),np.nanmax(mesh),20);
plt.figure(figsize=(15,10))
cf = plt.contourf(y_mesh, z_mesh, mesh.T, lev1, extend='both')
cbar = plt.colorbar(cf, label=r'$kemodel \nu_T/u_b$')
cbar.set_ticks(np.arange(0,np.nanmax(lev1),5e-2)) #and this line

phaselabels = [r'$-\frac{3\pi}{4}$',r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$', 
               r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',
               r'$\pi$']

#add in delta
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

plt.savefig('plots/kemodel.pdf',dpi=500) #save plot to send to neighbors
