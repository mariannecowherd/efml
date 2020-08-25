#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:00:24 2020

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
wpk = np.zeros((30,len(burstnums)))
Ps = np.zeros((30,len(burstnums)))
Pw = np.zeros((30,len(burstnums)))
Pw_temp = np.zeros((30,8))
wtk = np.zeros((30,len(burstnums)))
wtk_temp = np.zeros((30,8))
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
        
        k = 0.5*(up**2 + vp**2 + wp**2)
        t = vec['time'][:,0]*86400
        
        #Unsteady term
        for zz in range(ngood):
            dkdt[zz,n] = linregress(naninterp(t),k[zz,:])[0]
        
        # turbulent transport term
        wpk_temp = np.nanmean(wp*k, axis = 1)
        tck = interpolate.splrep(np.flipud(z[idxgood]),np.flipud(wpk_temp[idxgood]), s = 1e-8)
        znew = np.linspace(np.nanmin(z[idxgood]),np.nanmax(z[idxgood]),200)
        dwpkdz = interpolate.splev(znew,tck, der = 1)
        wpk[:,n] = np.interp(z,znew,dwpkdz)
        
        #Shear production term
        tck = interpolate.splrep(np.flipud(z[idxgood]),np.flipud(ubar[idxgood,0]), s = 1e-8)
        znew = np.linspace(np.nanmin(z[idxgood]),np.nanmax(z[idxgood]),200)
        dudz_temp = interpolate.splev(znew,tck, der = 1)
        dudz = np.interp(z,znew,dudz_temp)         
        Ps[:,n] = np.nanmean(up*wp, axis = 1)*dudz
        
        
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
            wtilde_phase = wtilde[:,idx1]
            
            k = 0.5*(up_phase**2 + vp_phase**2 + wp_phase**2)
            
            z = vec['z'].flatten()
            idxgood = (z > 0)
            #getting dkdt
            
            
            utilde_phase_bar = np.nanmean(utilde_phase, axis = 1)
            tck = interpolate.splrep(np.flipud(z[idxgood]),np.flipud(utilde_phase_bar[idxgood]))
            znew = np.linspace(np.nanmin(z[idxgood]),np.nanmax(z[idxgood]),200)
            dudz_temp = interpolate.splev(znew,tck, der = 1)
            dudz_wave = np.interp(z,znew,dudz_temp) 
            

            #wave production term
            Pw_temp[:,j] = np.nanmean(dudz_wave.reshape(-1,1)*up_phase*wp_phase, axis = 1)
            
            #Wave transport term
            kbar = np.nanmean(k, axis = 1)
            tck = interpolate.splrep(np.flipud(z[idxgood]),np.flipud(kbar[idxgood]), s = 1e-8)
            znew = np.linspace(np.nanmin(z[idxgood]),np.nanmax(z[idxgood]),200)
            dkdz_temp = interpolate.splev(znew,tck, der = 1)
            dkdz = np.interp(z,znew,dkdz_temp).reshape(-1,1)
            wtk_temp[:,j] = np.nanmean(wtilde_phase*dkdz, axis = 1)
            
        Pw[:,n] = np.nanmean(Pw_temp, axis = 1)
        wtk[:,n] = np.nanmean(wtk_temp, axis = 1)
        export = {'wpk': wpk, 'Pw': Pw, 'Ps': Ps, 'wtk': wtk,'dkdt': dkdt}
            
        
        np.save('tke_balance.npy', export)
        print(n)
        
    except ValueError:
        print(traceback.format_exc())
        # continue
    except TypeError :
        print(traceback.format_exc())
        # continue