#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 15:35:26 2020

@author: marianne, gegan

@title: stokesfunctions
"""

import numpy as np
import scipy.signal as sig

'''
this function sets up the stokes solution and either solves it to be as close
as possible to the observations (solve=True) or produces a solution set 
with (solve=False)

phasebins = phase bins
omega = wave omega
u0 = ub, wave orbital velocity
z = heights vector
umeas = measured, thing you're comparing it with
solve = is this to do a fit? if so, True and it will solve stokes. false will return something you can plot
height = does it vary in height. True gets you nu_t(z), False gets you nu_t
'''

dphi = np.pi/4

def make_stokes(phasebins,omega,u0,offset,plot):
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
        
        if plot:
            return ustokes
        else:
            return ustokes.flatten(order = 'F')
    return stokes

def make_stokes_z(phasebins,offset,omega,u0,z,umeas,plot):
    def stokes_z(nu):        
        tf = 1 #how many wave periods to average over
        
        t = np.linspace(-tf*np.pi/omega,tf*np.pi/omega,tf*100) #Time vector  
        uwave = np.zeros((len(z),len(t))) #temporary array for given frequency
        
        #Stokes solution at each height
        for jj in range(len(z)):
            uwave[jj,:] = u0*(np.cos(omega*t) - 
                 np.exp(-np.sqrt(omega/(2*nu[jj]))*(z[jj]-offset))*np.cos(
                     (omega*t) - np.sqrt(omega/(2*nu[jj]))*(z[jj]-offset)))
            
        
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
        
        if plot:
            return ustokes
        else:
            return ustokes.flatten(order = 'F') - umeas
    return stokes_z
    