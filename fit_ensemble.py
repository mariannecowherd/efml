#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:09:45 2020

@author: gegan

@title: fit_ensemble
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from scipy.optimize import curve_fit
from scipy import interpolate
import vectrinofuncs as vfs

profs = np.load('phaseprofiles.npy')
stress = np.load('phase_stress.npy', allow_pickle=True).item()
bl = np.load('blparams.npy', allow_pickle = True).item()
nparams = np.load('nparams.npy', allow_pickle = True).item()
z = stress['z']
phasebins = bl['phasebins']
omega = bl['omega']
ub = bl['ubvec']
idx = (ub > 0.07)

#%% Ensemble averaging to find mean profile
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

vel_ens = 2*np.nanmean(vel_interp[velidx,:,:],axis=0)

#%% Optimization function

def make_stokes_offset(phasebins, omega, u0, colnum, dphi = np.pi/4):
    def stokes(z,nu,offset):


        tf = 1 #how many wave periods to average over

        t = np.linspace(-tf*np.pi/omega,tf*np.pi/omega,tf*100) #Time vector
        uwave = np.zeros((len(z),len(t))) #temporary array for given frequency

        #Stokes solution at each height
        for jj in range(len(z)):
                uwave[jj,:] = u0*(np.cos(omega*t) -
                  np.exp(-np.sqrt(omega/(2*nu))*(z[jj]-offset))*np.cos(
                          (omega*t) - np.sqrt(omega/(2*nu))*(z[jj]-offset)))

        huwave = sig.hilbert(np.nanmean(uwave,axis = 0))
        pw = np.arctan2(huwave.imag,huwave.real)

        ustokes = np.zeros((len(z),len(phasebins))) #Analytical profile

        for ii in range(len(phasebins)):

            if ii == 0:
                #For -pi
                idx2 =  ( (pw >= phasebins[-1] + (dphi/2)) | (pw <= phasebins[0] + (dphi/2))) #Analytical
            else:
                #For phases in the middle
                idx2 = ((pw >= phasebins[ii]-(dphi/2)) & (pw <= phasebins[ii]+(dphi/2))) #analytical

            ustokes[:,ii] = np.nanmean(uwave[:,idx2],axis = 1)  #Averaging over the indices for this phase bin for analytical solution

        return ustokes[:,colnum]
    return stokes

def make_stokes(phasebins, omega, u0, offset, colnum, dphi = np.pi/4):
    def stokes(z,nu):


        tf = 1 #how many wave periods to average over

        t = np.linspace(-tf*np.pi/omega,tf*np.pi/omega,tf*100) #Time vector
        uwave = np.zeros((len(z),len(t))) #temporary array for given frequency

        #Stokes solution at each height
        for jj in range(len(z)):
                uwave[jj,:] = u0*(np.cos(omega*t) -
                  np.exp(-np.sqrt(omega/(2*nu))*(z[jj]-offset))*np.cos(
                          (omega*t) - np.sqrt(omega/(2*nu))*(z[jj]-offset)))

        huwave = sig.hilbert(np.nanmean(uwave,axis = 0))
        pw = np.arctan2(huwave.imag,huwave.real)

        ustokes = np.zeros((len(z),len(phasebins))) #Analytical profile

        for ii in range(len(phasebins)):

            if ii == 0:
                #For -pi
                idx2 =  ( (pw >= phasebins[-1] + (dphi/2)) | (pw <= phasebins[0] + (dphi/2))) #Analytical
            else:
                #For phases in the middle
                idx2 = ((pw >= phasebins[ii]-(dphi/2)) & (pw <= phasebins[ii]+(dphi/2))) #analytical

            ustokes[:,ii] = np.nanmean(uwave[:,idx2],axis = 1)  #Averaging over the indices for this phase bin for analytical solution

        return ustokes[:,colnum]
    return stokes

#%%
m,n,p = profs.shape

nu_fit = np.zeros((p,))*np.nan

plt.figure(1)

for j in range(p):

    idxgood = (znew < 0.01)
    uprof = vel_ens[idxgood,j]
    zpos = znew[idxgood] - 0.001

    # popt, pcov = curve_fit(make_stokes_offset(phasebins,np.nanmean(omega),1,j),zpos,uprof,
    #                                       p0 = (5e-6, 0.004), bounds = ([1e-6,1e-4],[1e-4,6e-3]))
    popt, pcov = curve_fit(make_stokes(phasebins,np.nanmean(omega),1,0,j),zpos,uprof,
                                          p0 = 2e-6, bounds = (1e-6,1e-4))

    plt.plot(vel_ens[idxgood,j],zpos,'-', color = 'C' + str(j))
    plt.plot(make_stokes(phasebins,np.nanmean(omega),1,0,j)(zpos,popt[0]),zpos, ':', color = 'C' + str(j))
    nu_fit[j] = popt[0]




#%%
