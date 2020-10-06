#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 11:49:42 2020

@author: gegan
"""

import numpy as np
import scipy.signal as sig
from scipy.optimize import curve_fit

profs = np.load('phaseprofiles.npy')
stress = np.load('phase_stress.npy', allow_pickle=True).item()
bl = np.load('blparams.npy', allow_pickle = True).item()
nparams = np.load('nparams.npy', allow_pickle = True).item()
z = stress['z']
phasebins = bl['phasebins']
omega = bl['omega']
ub = bl['ubvec']
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

nu_fit = np.zeros((n,p))*np.nan
offset_fit = np.zeros((n,p))*np.nan

for i in range(n):
    for j in range(p):
        
        
        idxgood = ((z[:,i] > 0) & (z[:,i] < 0.0105))
        uprof = profs[idxgood,i,j]
        zpos = z[idxgood,i]
        
        try:
            popt, pcov = curve_fit(make_stokes_offset(phasebins,omega[i],ub[i],j),zpos,uprof, 
                                                  p0 = (5e-6, 0.004), bounds = ([1e-6,1e-4],[1e-4,6e-3]))
            # popt, pcov = curve_fit(make_stokes(phasebins,omega[i],ub[i],offset_const,j),zpos,uprof, 
            #                                       p0 = 5e-6, bounds = (1e-6,1e-4))
            
            nu_fit[i,j] = popt[0]
            offset_fit[i,j] = popt[1]
        except ValueError:
            continue
        except RuntimeError: 
            continue

    print(i)
    np.save('nu_fits_phase.npy', nu_fit)
    
    
#%%

idx = (ub > 0.07)

