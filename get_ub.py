#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 12:38:23 2020

@author: marianne
"""

import numpy as np
import scipy.io as sio
import scipy.signal as sig
from scipy import interpolate
import vectrinofuncs as vfs

#phase decomposition
#using bricker & monismith method

#location of vectrino data
filepath = '/Volumes/Seagate Backup Plus Drive/VectrinoSummer/vecfiles/'

wavedir = np.load('data/wavedir.npy',allow_pickle = True)

burstnums = list(range(384))
fs = 64
thetamaj_summer =  -28.4547
dphi = np.pi/4 #Discretizing the phase
phasebins = np.arange(-np.pi,np.pi,dphi)

# profiles = np.zeros((30,len(burstnums),len(phasebins)))
# uw_wave = np.zeros((30,len(burstnums),len(phasebins)))
# upwp = np.zeros((30,len(burstnums),len(phasebins)))
# z = np.zeros((30,len(burstnums)))

#dissipation
# epsilon = np.zeros((30,len(burstnums),len(phasebins)))
# dudz = np.zeros((30,len(burstnums),len(phasebins)))
# tke = np.zeros((30,len(burstnums),len(phasebins)))
# tke_wave = np.zeros((30,len(burstnums),len(phasebins)))
ub = np.zeros(len(burstnums))

#this takes many hours to run -- load from .npy files if possible
for n in range(188,213):
    try:
        vec = sio.loadmat(filepath + 'vectrino_' + str(n) + '.mat')

        #direction
        veleast,velnorth = vfs.pa_rotation(vec['velmaj'],vec['velmin'],-thetamaj_summer)
        velmajwave,velminwave = vfs.pa_rotation(veleast,velnorth,wavedir[n])
        u = np.nanmean(velmajwave[0:6,], axis = 0) #vertical average
        #v = np.nanmean(velminwave[-5:,], axis=0) #vertical average

        #Calculating fluctuating velocity
        p,m = np.shape(velmajwave)
        #up = np.zeros((p,m))
        #vp = np.zeros((p,m))
        #w1p = np.zeros((p,m))
        #ubar = np.nanmean(velmajwave,axis = 1)
        #vbar = np.nanmean(velminwave,axis = 1)
        #w1bar = np.nanmean(vec['velz1'],axis = 1)
        #for i in range(m):
        #    up[:,i] = velmajwave[:,i] - ubar
        #    vp[:,i] = velmajwave[:,i] - vbar
        #    w1p[:,i] = vec['velz1'][:,i] - w1bar


        #spectrum to find wave peak
        #fu,pu = sig.welch(u,fs = fs, window = 'hamming', nperseg = len(u)//10,detrend = 'linear')
        #fmax = fu[np.nanargmax(pu)]

        #Filtering in spectral space
        ufilt = vfs.wave_vel_decomp(u,fs = fs, component = 'u')
        #vfilt = vfs.wave_vel_decomp(v,fs = fs, component = 'v')
        
        ub[n] = np.sqrt(np.var(ufilt))
        

        print(n)

    except ValueError:
        continue
    except TypeError:
        continue

#save
#np.save('phase_stress.npy', {'uw_wave': uw_wave, 'uw': upwp, 'z' : z,
#                                      'freestream': ubar, 'dudz':dudz,
#                                      'epsilon': epsilon, 'tke':tke,
#                                      'tke_wave':tke_wave})

#np.save('phaseprofiles.npy', profiles)
