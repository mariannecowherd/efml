#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 14:43:26 2020

@author: gegan

@title: integrated_stress
"""

import matplotlib.pyplot as plt
import numpy as np

waveturb = np.load('waveturb.npy', allow_pickle = True).item()
bl = np.load('blparams.npy', allow_pickle = True).item()

idx = list(waveturb.keys())
del idx[292]
del idx[203]

allsed = np.load('/Users/gegan/Documents/Python/Research/Erosion_SD6/allsedP1_sd6.npy',
                 allow_pickle = True).item()
ubar = allsed['ubar'][:,idx]
z = allsed['z'][:,idx]
dubardz = np.gradient(ubar,-.001, axis = 0)

data = np.load('phase_stress.npy', allow_pickle = True).item()

dphi = np.pi/4 #Discretizing the phase
phasebins = np.arange(-np.pi,np.pi,dphi)
delta = np.nanmean(bl['delta'][:,idx],axis = 1)


#%% Equation 3.2c
intmask = ((data['z'][:,idx] < 0.0105) & (data['z'][:,idx] > 0.0005)).astype(int)

data['epsilon'][np.isnan(data['epsilon'])] = 0
data['tke'][np.isnan(data['tke'])] = 0
data['tke_wave'][np.isnan(data['tke_wave'])] = 0
data['uw'][np.isnan(data['uw'])] = 0
data['uw_wave'][np.isnan(data['uw_wave'])] = 0
data['dudz'][np.isnan(data['dudz'])] = 0
dubardz[np.isnan(dubardz)] = 0


#%% Trying delta from a mixing length model

epsilon = np.array([np.nanmean( (1/0.01)*np.trapz(data['epsilon'][:,idx,i]*intmask,
                                         np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ])

k = np.array([np.nanmean(np.trapz((1/0.01)*data['tke'][:,idx,i]*intmask,
                                  np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )



# nut = 0.09 * np.array([np.nanmean(np.trapz((data['tke'][:,idx,i]**2/data['epsilon'][:,idx,i])*intmask,
#                                   np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )
nut = 0.09*(k)**2/epsilon


# delta_ml = -np.array([np.nanmean(np.trapz((0.09 * (data['tke'][:,idx,i]**2/data['epsilon'][:,idx,i])/(data['dudz'][:,idx,i]))*intmask,
#                                   np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )
# delta_ml = -np.array([np.nanmean(np.trapz(0.09 * (nut[i]/(data['dudz'][:,idx,i]))*intmask,
#                                   np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )

delta_ml = np.array([np.nanmean(np.nanmean(0.09 * (nut[i]/(data['dudz'][:,idx,i]))*intmask,
                                 axis = 0)) for i in range(8) ] )



# Ps = -np.array([np.nanmean(np.trapz(data['uw'][:,idx,i]*dubardz*intmask,
#                                   np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )

# Pw = -np.array([np.nanmean(np.trapz(data['uw'][:,idx,i]*data['dudz'][:,idx,i]*intmask,
#                                   np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )

# epsilon = np.array([np.nanmean( np.trapz(data['epsilon'][:,idx,i]*intmask,
#                                          np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ])


# dkdt_temp = np.gradient(data['tke'], 3*phasebins/(2*np.pi), axis = 2, edge_order = 2)
# dkdt = np.array([np.nanmean(np.trapz(dkdt_temp[:,idx,i]*intmask,
#                                   np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )



# #Getting wprime from waveturb
# wprime = np.zeros((30,len(idx),8))*np.nan
# for k,i in enumerate(idx):
#     wt = waveturb[i]
#     for j in range(8):
#         try:
#             wprime[:,k,j] = np.sqrt(wt[j]['w1w1'])
#         except KeyError:
#             continue
# wprime[np.isnan(wprime)] = 0

# #Getting dkdz
# dkdz = np.zeros((30,len(idx),8))*np.nan
# for k,i in enumerate(idx):
#     for j in range(8):
#         dkdz[:,k,j] = np.gradient(data['tke'][:,i,j]*intmask[:,k],
#                                   np.flipud(data['z'][:,i]), edge_order = 2)

# dkdz[np.isnan(dkdz)] = 0
# wdkdz = np.array([np.nanmean(np.trapz(dkdz[:,:,i]*wprime[:,:,i]*intmask,
#                                   data['z'][:,idx], axis = 0) ) for i in range(8) ] )

# rhs = Ps + Pw - epsilon - wdkdz
# #Plot
# fig, ax = plt.subplots()

# ax.plot(phasebins, dkdt, label = 'dkdt')

# ax.plot(phasebins, Pw, label = 'Pw')

# ax.plot(phasebins, Ps , label = 'Ps ' )

# ax.plot(phasebins, epsilon, label = 'eps')

# ax.plot(phasebins, wdkdz, label = 'transport')



# ax.legend()

#%% Equation 3.2 b
# kwave = np.array([np.nanmean(np.trapz(data['tke_wave'][:,idx,i]*intmask,
#                                   np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )


# dkdt = np.gradient(kwave,3*phasebins/(2*np.pi), edge_order = 2)

# Pwm = -np.array([np.nanmean(np.trapz(data['uw_wave'][:,idx,i]*dubardz*intmask,
#                                   np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )

# Ptw = -np.array([np.nanmean(np.trapz(data['uw'][:,idx,i]*data['dudz'][:,idx,i]*intmask,
#                                   np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )

# epsilon = np.array([np.nanmean( np.trapz(data['epsilon'][:,idx,i]*intmask,
#                                          np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ])


# fig, ax = plt.subplots()

# ax.plot(phasebins, dkdt, label = 'dkdt')

# ax.plot(phasebins, Pwm - Ptw - epsilon, label = 'Pwm - Ptw - eps')

# ax.legend()

#%%
epsilon = np.array([np.nanmean( np.trapz(data['epsilon'][:,idx,i]*intmask,
                                         np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ])

k = np.array([np.nanmean(np.trapz(data['tke'][:,idx,i]*intmask,
                                  np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )

kwave = np.array([np.nanmean(np.trapz(data['tke_wave'][:,idx,i]*intmask,
                                      np.flipud(data['z'][:,idx]), axis = 0)) for i in range(8) ] )

nut = 0.09*(k + kwave)**2/epsilon

nut_wave = kwave**2/epsilon

# P = np.array([np.nanmean(np.trapz(data['uw'][:,idx,i]*data['dudz'][:,:,i]*intmask,
#                                   np.flipud(data['z']), axis = 0)) for i in range(8) ] )

# Pw = np.array([np.nanmean(np.trapz(data['uw_wave'][:,:,i]*data['dudz'][:,:,i]*intmask,
#                                    np.flipud(data['z']), axis = 0)) for i in range(8) ] )

#%% Eddy viscosity plot

fig, ax = plt.subplots()

ax.plot(phasebins, nut, label = r'$\nu_T$')
# ax.plot(phasebins, nut_wave, label = r'$\nu_W$')

ax.legend()

#%% TKE balance

# fig, ax  = plt.subplots()

# ax.plot(phasebins, k, label = r'$k$')

# ax.plot(phasebins, P, label = r'$P$' )

# ax.plot(phasebins, kwave, label = r'$k_W$')

# ax.plot(phasebins, Pw, label = r'$P_W$')

# ax.plot(phasebins, epsilon, label = r'$\epsilon$')

# ax.legend()
