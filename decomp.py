#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 15:25:18 2020

@author: marianne, gegan

@title: decomp
"""

import numpy as np
import scipy.io as sio
import scipy.signal as sig
import sys
sys.path.append('/Users/marianne/Documents/GitHub/efml')

import vectrinofuncs as vfs


#phase decomposition
#using bricker-monismith method 

filepath = '/Users/marianne/Desktop/VectrinoSummer/vecfiles/' #location of vectrino data

wavedir = np.load('wavedir.npy',allow_pickle = True)

burstnums = list(range(384))
fs = 64
thetamaj_summer =  -28.4547
dphi = np.pi/4 #Discretizing the phase
phasebins = np.arange(-np.pi,np.pi,dphi)

profiles = np.zeros((30,len(burstnums),len(phasebins)))
uw_wave = np.zeros((30,len(burstnums),len(phasebins)))
upwp = np.zeros((30,len(burstnums),len(phasebins)))
z = np.zeros((30,len(burstnums)))

for n in burstnums:
    try:
        vec = sio.loadmat(filepath + 'vectrino_' + str(n) + '.mat')
        
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
        for ii in range(m):
            up[:,ii] = velmajwave[:,ii] - ubar
            vp[:,ii] = velmajwave[:,ii] - vbar
            w1p[:,ii] = vec['velz1'][:,ii] - w1bar

        
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
            
            uphase = up[:,idx1]
            vphase = vp[:,idx1]
        
            tempvec = {'velmaj': velmajwave[:,idx1], 'velmin': velminwave[:,idx1],
                           'w1': vec['velz1'][:,idx1], 'w2':vec['velz2'], 'z': vec['z']}
            
            waveturb = vfs.get_turb_waves(tempvec, fs,'phase')
            
            uw_wave[:,n,j] = waveturb['uw1_wave']
            upwp[:,n,j] = waveturb['uw1']
            z[:,n] = vec['z'].flatten()
    
            
            uproftemp = np.nanmean(up[:,idx1],axis = 1) 
            profiles[:,n,j] =  uproftemp

        
        np.save('phase_stress.npy', {'uw_wave': uw_wave, 'uw': upwp, 'z' : z,'freestream': ubar})
    
        
        print(n)
        
    except ValueError:
        continue
    
np.save('phaseprofiles.npy', profiles)
        


