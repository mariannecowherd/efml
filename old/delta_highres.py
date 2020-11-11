#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:45:08 2020

@author: marianne

@title: highres delta
"""

import numpy as np
import scipy.io as sio
import scipy.signal as sig
from scipy import interpolate
import sys

import vectrinofuncs as vfs


filepath = '/Volumes/Seagate Backup Plus Drive/VectrinoSummer/'
sys.path.append(filepath)


blparams = np.load('blparams.npy',allow_pickle=True).item()
ubvec = blparams['ubvec']/np.sqrt(2)
stress = np.load('phase_stress.npy',allow_pickle=True).item()

wavedir = np.load('wavedir.npy',allow_pickle = True)

burstnums = list(range(384))
fs = 64
thetamaj_summer =  -28.4547
dphi = np.pi/12 #Discretizing the phase
phasebins = np.arange(-np.pi,np.pi,dphi)

profiles = np.zeros((30,len(burstnums),len(phasebins)))
#%%
for n in burstnums:
    try:
        vec = sio.loadmat(filepath + 'vecfiles/vectrino_' + str(n) + '.mat')
        
        #direction
        veleast,velnorth = vfs.pa_rotation(vec['velmaj'],vec['velmin'],-thetamaj_summer)
        velmajwave,velminwave = vfs.pa_rotation(veleast,velnorth,wavedir[n])
        u = np.nanmean(velmajwave, axis = 0) #vertical average
        v = np.nanmean(velminwave, axis=0) #vertical average
        
        #Calculating fluctuating velocity 
        p,m = np.shape(velmajwave)
        up = np.zeros((p,m))
        vp = np.zeros((p,m))
        w1p = np.zeros((p,m))
        ubar = np.nanmean(velmajwave,axis = 1)
        vbar = np.nanmean(velminwave,axis = 1)
        w1bar = np.nanmean(vec['velz1'],axis = 1)
        for i in range(m):
            up[:,i] = velmajwave[:,i] - ubar
            vp[:,i] = velmajwave[:,i] - vbar
            w1p[:,i] = vec['velz1'][:,i] - w1bar

        
        #spectrum to find wave peak
        fu,pu = sig.welch(u,fs = fs, window = 'hamming', nperseg = len(u)//10,detrend = 'linear') 
        fmax = fu[np.nanargmax(pu)]
        
        #Filtering in spectral space 
        ufilt = vfs.wave_vel_decomp(u,fs = fs, component = 'u')
        vfilt = vfs.wave_vel_decomp(v,fs = fs, component = 'v')
        
        #calculate analytic signal based on de-meaned and low pass filtered velocity
        hu = sig.hilbert(ufilt - np.nanmean(ufilt)) 
        
        #Phase based on analytic signal
        p = np.arctan2(hu.imag,hu.real)

        for j in range(len(phasebins)):
           
            if j == 0:
                #For -pi
                idx1 = ( (p >= phasebins[-1] + (dphi/2)) | (p <= phasebins[0] + (dphi/2))) #Measured
            else:
                #For phases in the middle
                idx1 = ((p >= phasebins[j]-(dphi/2)) & (p <= phasebins[j]+(dphi/2))) #measured
            
 
    
            
            uproftemp = np.nanmean(up[:,idx1],axis = 1) 
            profiles[:,n,j] =  uproftemp

    
        
        print(n)
        
    except ValueError:
        continue
    except TypeError:
        continue
    
np.save('hrdelta.npy', profiles)


#%%
vel_interp = np.zeros([384,15,100]) #initialize output
velidx=[]
znew = np.linspace(0.001, 0.015, 15)

#ensemble average, normalize by ubvec
for n in burstnums:
    for i in range(100):
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

vel_temp = np.nanmean(vel_interp[velidx,:,:],axis=0)

delta_highres = 2*vfs.displacement_thickness_interp(vel_temp,znew)


#%%
#highres simulation
nu = 1e-6 #kinematic viscosity

ub_bar = np.nanmean(ubvec)
u0 = np.nanmean(np.abs(u0s))
omega_bar = np.nanmean(omega)
omega_std = np.nanstd(omega)
z = np.linspace(0.00, 0.015, 100) #height vector
t = np.linspace(-np.pi/omega_bar,np.pi/omega_bar,100) #time vector 
nm = 1000 #how many values of omega
oms_hr = np.linspace(omega_bar-omega_std,omega_bar+omega_std,nm) #omega vector

omstokes_highres = np.zeros((len(z),len(phasebins),nm)) #initialized output 

for k in range(len(oms_hr)):
    om = oms_hr[k]
    uwave = np.zeros((len(z),len(t))) #temporary array for given frequency
    phi = np.random.rand() * 2*np.pi #random value for phi
    
    #stokes solution
    for jj in range(len(z)):
        uwave[jj,:] = (np.cos(om*t-phi) - 
                    np.exp(-np.sqrt(om/(2*nu))*z[jj])*np.cos(
                        (om*t-phi) - np.sqrt(om/(2*nu))*z[jj]))
    huwave = sig.hilbert(np.nanmean(uwave,axis = 0))  #hilbert transform
    pw = np.arctan2(huwave.imag,huwave.real) #phase
    
    ustokes_highres = np.zeros((len(z),len(phasebins)))
    
    #allocate into phase bins
    for ii in range(len(phasebins)):
       
            if ii == 0:
                #For -pi
                idx2 =  ( (pw >= phasebins[-1] + (dphi/2)) | (pw <= phasebins[0] + (dphi/2))) #analytical
            else:
                #For phases in the middle
                idx2 = ((pw >= phasebins[ii]-(dphi/2)) & (pw <= phasebins[ii]+(dphi/2))) #analytical
           
            ustokes_highres[:,ii] = np.nanmean(uwave[:,idx2],axis = 1)  #Averaging over the phase bin
    
            omstokes_highres[:,ii,k] = ustokes_highres[:,ii]
            
omsum_highres = np.zeros((len(z),len(phasebins)))

#average bursts for the same phase bin for all values of omega
for k in range(len(z)):
    for i in range(len(phasebins)):
        omsum_highres[k,i] = np.nanmean(omstokes_highres[k,i,:])
        
#%%
        
#znew2 = np.linspace(0.001,0.015,100)
dom_highres = 2*vfs.displacement_thickness_interp(omsum_highres,z)
theory = omsum_highres
om_theory = vfs.displacement_thickness_interp(theory,z2)
for i in range(24):
    z2 = z+0.001
    omidx = np.nanargmin(np.abs(om_theory[i]-z))
    ax[0].plot(theory[omidx,i],100*(z2[omidx]),'*', color = 'y')
    
            
#%%
omidx=np.unique(omidx)

om_temp = np.nanmean(om_interp[omidx,:,:],axis=0)


