#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:09:30 2020

@author: gegan
"""

import numpy as np
import scipy.io as sio
import scipy.signal as sig
from scipy import interpolate
from scipy.stats import linregress
import sys
sys.path.append('/Users/gegan/Documents/Python/Research/General')

from mylib import naninterp
import vecfuncs as vfs
import traceback

#this takes a long time -- need to update to make it more efficient

#phase decomposition
#using bricker-monismith method 

filepath = '/Volumes/Seagate Backup Plus Drive/FullDataRepo/VectrinoSummer/' #location of vectrino data

wavedir = np.load('wavedir.npy',allow_pickle = True)

burstnums = list(range(384))
fs = 64
thetamaj_summer =  -28.4547
dphi = np.pi/4 #Discretizing the phase
phasebins = np.arange(-np.pi,np.pi,dphi)

#Terms in tke balance
wtkt = np.zeros((30,len(burstnums)))
Pws = np.zeros((30,len(burstnums)))
Pw = np.zeros((30,len(burstnums)))
Pw_temp = np.zeros((30,8))
wtrs = np.zeros((30,len(burstnums)))
wtrs_temp = np.zeros((30,8))
dkdt = np.zeros((30,len(burstnums)))

for n in burstnums:
    try:
        vec = sio.loadmat(filepath + 'vectrino_' + str(n) + '.mat')
        
        #direction
        veleast,velnorth = vfs.pa_rotation(vec['velmaj'],vec['velmin'],-thetamaj_summer)
        velmajwave,velminwave = vfs.pa_rotation(veleast,velnorth,wavedir[n])
        U = np.nanmean(velmajwave, axis = 0) #vertical average
        V = np.nanmean(velminwave, axis=0) #vertical average
        W = np.nanmean(vec['velz1'], axis = 0)
        
        #Calculating mean velocity 
        ubar = np.nanmean(velmajwave,axis = 1, keepdims = True)
        vbar = np.nanmean(velminwave,axis = 1, keepdims = True)
        w1bar = np.nanmean(vec['velz1'],axis = 1, keepdims = True)
        
        #Calculating wave velocity at each height
        utilde = np.zeros_like(velmajwave)
        vtilde = np.zeros_like(velmajwave)
        wtilde = np.zeros_like(velmajwave)
        
        z = vec['z'].squeeze()
        idxgood = (z > 0)
        ngood = np.sum(idxgood)
        
        for idx in range(ngood):
            utilde[idx,:] =  vfs.wave_vel_decomp(velmajwave[idx,:],fs = fs, component = 'u')
            vtilde[idx,:] = vfs.wave_vel_decomp(velminwave[idx,:],fs = fs, component = 'u')
            wtilde[idx,:] = vfs.wave_vel_decomp(vec['velz1'][idx,:],fs = fs, component = 'w')
        
        up = velmajwave - utilde - ubar
        vp = velminwave - vtilde - vbar
        wp = vec['velz1'] - wtilde - w1bar
        
        #wave TKE
        kt = 0.5*(utilde**2 + vtilde**2 + wtilde**2)
        t = vec['time'][:,0]*86400
        
        #Unsteady term
        for zz in range(ngood):
            dkdt[zz,n] = linregress(naninterp(t),kt[zz,:])[0]
        
        # wave transport of wave tke term
        wtkt_temp = np.nanmean(wtilde*kt, axis = 1)
        tck = interpolate.splrep(np.flipud(z[idxgood]),np.flipud(wtkt_temp[idxgood]), s = 1e-8)
        znew = np.linspace(np.nanmin(z[idxgood]),np.nanmax(z[idxgood]),200)
        dwtktdz = interpolate.splev(znew,tck, der = 1)
        wtkt[:,n] = np.interp(z,znew,dwtktdz)
        
        #wave stress - mean shear production term
        tck = interpolate.splrep(np.flipud(z[idxgood]),np.flipud(ubar[idxgood,0]), s = 1e-8)
        znew = np.linspace(np.nanmin(z[idxgood]),np.nanmax(z[idxgood]),200)
        dudz_temp = interpolate.splev(znew,tck, der = 1)
        dudz = np.interp(z,znew,dudz_temp)         
        Pws[:,n] = np.nanmean(utilde*wtilde, axis = 1)*dudz
        
        
        # #spectrum to find wave peak
        # fu,pu = sig.welch(U,fs = fs, window = 'hamming', nperseg = len(U)//10,detrend = 'linear') 
        # fmax = fu[np.nanargmax(pu)]
        
        #Filtering in spectral space 
        ufilt = vfs.wave_vel_decomp(U,fs = fs, component = 'u')
        wfilt = vfs.wave_vel_decomp(W,fs = fs, component = 'w')
        
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
            
            up_phase = up[:,idx1]
            vp_phase = vp[:,idx1]
            wp_phase = wp[:,idx1]
            utilde_phase = utilde[:,idx1]
            vtilde_phase = vtilde[:,idx1]
            wtilde_phase = wtilde[:,idx1]
            
            kt = 0.5*(utilde_phase**2 + vtilde_phase**2 + wtilde_phase**2)
            
            z = vec['z'].flatten()
            idxgood = (z > 0)
            
            
            utilde_phase_bar = np.nanmean(utilde_phase, axis = 1)
            tck = interpolate.splrep(np.flipud(z[idxgood]),np.flipud(utilde_phase_bar[idxgood]))
            znew = np.linspace(np.nanmin(z[idxgood]),np.nanmax(z[idxgood]),200)
            dudz_temp = interpolate.splev(znew,tck, der = 1)
            dudz_wave = np.interp(z,znew,dudz_temp) 
            

            #wave production term
            Pw_temp[:,j] = np.nanmean(dudz_wave.reshape(-1,1)*up_phase*wp_phase, axis = 1)
            
            #Wave transport of phase-averaged stress term
            wtilders = np.nanmean(wtilde_phase*up_phase*wp_phase, axis = 1)
            tck = interpolate.splrep(np.flipud(z[idxgood]),np.flipud(wtilders[idxgood]), s = 1e-8)
            znew = np.linspace(np.nanmin(z[idxgood]),np.nanmax(z[idxgood]),200)
            dwtdz_temp = interpolate.splev(znew,tck, der = 1)
            wtrs_temp[:,j] = np.interp(z,znew,dwtdz_temp)
            
        Pw[:,n] = np.nanmean(Pw_temp, axis = 1)
        wtrs[:,n] = np.nanmean(wtrs_temp, axis = 1)
        export = {'wtrs': wtrs, 'Pw': Pw, 'Pws': Pws, 'wtkt': wtkt,'dkdt': dkdt}
            
        
        np.save('tke_balance_wave.npy', export)
        print(n)
        
    except ValueError:
        print(traceback.format_exc())
        # continue
    except TypeError :
        print(traceback.format_exc())
        # continue