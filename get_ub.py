#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 12:38:23 2020

@author: marianne
"""

import numpy as np
import scipy.io as sio
import vectrinofuncs as vfs

#phase decomposition
#using bricker & monismith method

#location of vectrino data
filepath = '/Volumes/Seagate Backup Plus Drive/VectrinoSummer/vecfiles/'

wavedir = np.load('data/wavedir.npy', allow_pickle=True)

burstnums = list(range(384))
fs = 64
thetamaj_summer = -28.4547
dphi = np.pi/4  #Discretizing the phase
phasebins = np.arange(-np.pi, np.pi, dphi)


ub = np.zeros(len(burstnums))

#this takes many hours to run -- load from .npy files if possible
for n in range(188,213):
    try:
        vec = sio.loadmat(filepath + 'vectrino_' + str(n) + '.mat')

        #direction
        veleast,velnorth = vfs.pa_rotation(vec['velmaj'],vec['velmin'],-thetamaj_summer)
        velmajwave,velminwave = vfs.pa_rotation(veleast,velnorth,wavedir[n])
        u = np.nanmean(velmajwave[0:6,], axis = 0) #vertical average

        #Filtering in spectral space
        ufilt = vfs.wave_vel_decomp(u,fs = fs, component = 'u')        
        ub[n] = np.sqrt(np.var(ufilt))
        

        print(n)

    except ValueError:
        continue
    except TypeError:
        continue

