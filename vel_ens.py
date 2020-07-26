#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 14:39:46 2020

@author: marianne

@title: vel-ens
"""

import numpy as np
import scipy.signal as sig
from scipy import interpolate
import vectrinofuncs as vfs
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


#data
blparams = np.load('/Users/Marianne/Documents/GitHub/efml/blparams.npy',allow_pickle=True).item()
profiles = np.load('phaseprofiles.npy',allow_pickle=True)
stress = np.load('phase_stress.npy',allow_pickle=True).item()
delta = blparams['delta']
phasebins = blparams['phasebins']
ustarwc_gm = blparams['ustarwc_gm']
omega = blparams['omega']
ubvec = blparams['ubvec']
zs = stress['z']
u0s = stress['freestream']


burstnums = list(range(384))
'''
construct stokes solution given omega, phi, and u0
omega is a range around the average wave peak for the entire deployment
u0 is the average bottom wave orbital velocity
phi is a random phase offset
''' 

dphi = np.pi/4 #Discretizing the phase
phasebins = np.arange(-np.pi,np.pi,dphi)
nu = 1e-6 #kinematic viscosity

ub_bar = np.nanmean(ubvec)
u0 = np.nanmean(np.abs(u0s))
omega_bar = np.nanmean(omega)
omega_std = np.nanstd(omega)
z = np.linspace(0.00, 0.015, 100) #height vector
t = np.linspace(-np.pi/omega_bar,np.pi/omega_bar,100) #time vector 
nm = 1000 #how many values of omega
oms = np.linspace(omega_bar-omega_std,omega_bar+omega_std,nm) #omega vector

omstokes = np.zeros((len(z),len(phasebins),nm)) #initialized output 

for k in range(len(oms)):
    om = oms[k]
    uwave = np.zeros((len(z),len(t))) #temporary array for given frequency
    phi = np.random.rand() * 2*np.pi #random value for phi
    
    #stokes solution
    for jj in range(len(z)):
        uwave[jj,:] = u0*(np.cos(om*t-phi) - 
                    np.exp(-np.sqrt(om/(2*nu))*z[jj])*np.cos(
                        (om*t-phi) - np.sqrt(om/(2*nu))*z[jj]))
    huwave = sig.hilbert(np.nanmean(uwave,axis = 0))  #hilbert transform
    pw = np.arctan2(huwave.imag,huwave.real) #phase
    
    ustokes = np.zeros((len(z),len(phasebins)))
    
    #allocate into phase bins
    for ii in range(len(phasebins)):
       
            if ii == 0:
                #For -pi
                idx2 =  ( (pw >= phasebins[-1] + (dphi/2)) | (pw <= phasebins[0] + (dphi/2))) #analytical
            else:
                #For phases in the middle
                idx2 = ((pw >= phasebins[ii]-(dphi/2)) & (pw <= phasebins[ii]+(dphi/2))) #analytical
           
            ustokes[:,ii] = np.nanmean(uwave[:,idx2],axis = 1)  #Averaging over the phase bin
    
            omstokes[:,ii,k] = ustokes[:,ii]
            
omsum = np.zeros((len(z),len(phasebins)))

#average bursts for the same phase bin for all values of omega
for k in range(len(z)):
    for i in range(len(phasebins)):
        omsum[k,i] = np.nanmean(omstokes[k,i,:])
        
'''
ensemble average measured velocity profiles at each phase bin
interpolate onto standard z scale
normalize by bottom wave orbital velocity
'''        
vel_interp = np.zeros([384,15,8]) #initialize output
velidx=[]
znew = np.linspace(0.001, 0.015, 15)

#ensemble average, normalize by ubvec
for n in burstnums:
    for i in range(8):
        try:
            vel_old = profiles[:,n,i]/ubvec[n]
            zold = stress['z'][:,n]
            zold = zold.flatten()
            zold,vel_old = vfs.nanrm2(zold,vel_old)
            f_vel = interpolate.interp1d(zold,vfs.naninterp(vel_old),kind='cubic')
            vel_interp[n,:,i] = (f_vel(znew))
            velidx.append(n)    
        except ValueError:
            continue
            
velidx=np.unique(velidx)

vel_ens = np.nanmean(vel_interp[velidx,:,:],axis=0)

phasebins2 = ['$-\\pi$', '$-3\\pi/4$', '$-\\pi/2$','$-\\pi/4$', '$0$',
              '$\\pi/4$', '$\\pi/2$', '$3\\pi/4$']
fig,ax = plt.subplots()
for i in range(8):
    ax.plot(1.9*vel_ens[:,i],znew) #1.9 is a fudge factor
    colorstr = 'C' + str(i)
    ax.plot(omsum[:,i]/ub_bar,z+0.001,':',color = colorstr,label = r'$\theta = $' + phasebins2[i])
    
ax.set_ylim(0,0.015)
plt.savefig('vel_ens.pdf')
