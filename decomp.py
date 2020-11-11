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
from scipy import interpolate
import vectrinofuncs as vfs

#phase decomposition
#using bricker & monismith method

#location of vectrino data
filepath = '/Volumes/Seagate Backup Plus Drive/FullDataRepo/VectrinoSummer/'

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

#dissipation
epsilon = np.zeros((30,len(burstnums),len(phasebins)))
dudz = np.zeros((30,len(burstnums),len(phasebins)))
tke = np.zeros((30,len(burstnums),len(phasebins)))
tke_wave = np.zeros((30,len(burstnums),len(phasebins)))

for n in burstnums:
    try:
        vec = sio.loadmat(filepath + 'vectrino_' + str(n) + '.mat')

        #direction
        veleast,velnorth = vfs.pa_rotation(vec['velmaj'],vec['velmin'],-thetamaj_summer)
        velmajwave,velminwave = vfs.pa_rotation(veleast,velnorth,wavedir[n])
        u = np.nanmean(velmajwave[-5:], axis = 0) #vertical average
        v = np.nanmean(velminwave[-5:], axis=0) #vertical average

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

            uphase = up[:,idx1]
            vphase = vp[:,idx1]

            tempvec = {'velmaj': velmajwave[:,idx1], 'velmin': velminwave[:,idx1],
                            'w1': vec['velz1'][:,idx1], 'w2':vec['velz2'], 'z': vec['z']}

            waveturb = vfs.get_turb_waves(tempvec, fs,'phase')

            #dissipation process can take many hours
            #see Feddersen et al., 2007
            epsilon[:,n,j] = vfs.get_dissipation(tempvec,fs,method = 'Fedd07')
            #
            tke[:,n,j] = 0.5*(waveturb['uu']  + waveturb['vv'] + waveturb['w1w1'])
            tke_wave[:,n,j] = 0.5*(waveturb['uu_wave'] + waveturb['vv_wave'] + waveturb['w1w1_wave'])

            uw_wave[:,n,j] = waveturb['uw1_wave']
            upwp[:,n,j] = waveturb['uw1']
            z[:,n] = vec['z'].flatten()


            uproftemp = np.nanmean(up[:,idx1],axis = 1)
            profiles[:,n,j] =  uproftemp


            idxgood = (z[:,n] > 0)
            tck = interpolate.splrep(np.flipud(z[idxgood,n]),np.flipud(profiles[idxgood,n,j]))
            znew = np.linspace(np.nanmin(z[idxgood,n]),np.nanmax(z[idxgood,n]),200)
            utemp =  interpolate.splev(znew,tck, der = 0)
            dudz_temp = interpolate.splev(znew,tck, der = 1)
            dudz[:,n,j] = np.interp(z[:,n],znew,dudz_temp)


        print(n)

    except ValueError:
        continue
    except TypeError:
        continue

np.save('phase_stress.npy', {'uw_wave': uw_wave, 'uw': upwp, 'z' : z,
                                      'freestream': ubar, 'dudz':dudz,
                                      'epsilon': epsilon, 'tke':tke,
                                      'tke_wave':tke_wave})

np.save('phaseprofiles.npy', profiles)
