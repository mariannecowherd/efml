#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:37:06 2020

@author: marianne

@title: june 9 for galen
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.io as sio
from scipy import interpolate
phasebins2=[r'$-\pi$',r'$-3\pi/4$',r'$-\pi/2$',r'$-\pi/4$',r'$0$',r'$\pi/4$',r'$\pi/2$',r'$3\pi/4$']


'''
this code adds together stokes solutions for laminar flow in an oscillating 
pressure gradient with the period of oscillation varies
'''

#%%
'''
this section adds the stokes solutions for a specified amplitude (u0) at
specified values of omega (oms), with a random value of phi
'''


u0=1 #amplitude
dphi = np.pi/4 #Discretizing the phase
phasebins = np.arange(-np.pi,np.pi,dphi)
nu = 1e-6 #kinematic viscosity
z = np.linspace(0.00, 0.020, 100) #height vector
t = np.linspace(-np.pi/omega,np.pi/omega,100) #time vector 
nm = 1000 #how many values of omega
oms = np.linspace(1.5,2.75,nm) #omega vector

omstokes = np.zeros((len(z),len(phasebins),nm))

for k in range(len(oms)):
    omega = oms[k]
    uwave = np.zeros((len(z),len(t))) #temporary array for given frequency
    phi = np.random.rand() * 2*np.pi #random value for phi
    #stokes soln
    #u0/nm means this will peak at 1, pre-emptive average
    for jj in range(len(z)):
        uwave[jj,:] = u0/nm*(np.cos(omega*t-phi) - 
                    np.exp(-np.sqrt(omega/(2*nu))*z[jj])*np.cos(
                        (omega*t-phi) - np.sqrt(omega/(2*nu))*z[jj]))
    huwave = sig.hilbert(np.nanmean(uwave,axis = 0)) 
    pw = np.arctan2(huwave.imag,huwave.real) #phase
    
    ustokes = np.zeros((len(z),len(phasebins)))
    
    for ii in range(len(phasebins)):
       
            if ii == 0:
                #For -pi
                idx2 =  ( (pw >= phasebins[-1] + (dphi/2)) | (pw <= phasebins[0] + (dphi/2))) #analytical
            else:
                #For phases in the middle
                idx2 = ((pw >= phasebins[ii]-(dphi/2)) & (pw <= phasebins[ii]+(dphi/2))) #analytical
           
            ustokes[:,ii] = np.nanmean(uwave[:,idx2],axis = 1)  #Averaging over the phase bin
    
            omstokes[:,ii,k] = ustokes[:,ii]
            
omsum = np.zeros((len(z),len(phasebins)))

#add bursts for the same phase bin for all values of omega
for k in range(len(z)):
    for i in range(len(phasebins)):
        omsum[k,i] = np.nansum(omstokes[k,i,:])

#plot
fig,ax=plt.subplots()
for ii in range(8):
    colorstr = 'C' + str(ii)
    delta = np.sqrt(2e-6/2.4)
    ax.plot(omsum[:,ii],z+0.003,'-',color = colorstr,label = r'$\theta = $' + phasebins2[ii])
    ax.set_xlabel(r'$\tilde{u}$')    
    ax.set_ylabel(r'$z$')
    ax.set_title('fake')
    ax.plot(phase_obs[N,ii],zs[N],'k-')
    

#%%
    
'''
this section adds the stokes solutions for amplitude-omega-phi combinations
from real vectrino bursts
'''


filepath = '/Users/marianne/Desktop/VectrinoSummer/vecfiles/'

def construct(N,plot=False):#burst number
    fs=64 #sampling frequency
    component = 'u'
    vec = sio.loadmat(filepath + 'vectrino_' + str(N) + '.mat')
    #vertical average over the profile
    u = np.nanmean(vec['velmaj'], axis = 0)
    #get amplitude, phi, and omega vectors from real data
    #see below for very slightly modified version of wave_vel_decomp that returns
    #amp, phi, and omeg instead of a timeseries
    amp, phi, omeg = wvd(u,fs,component,False)
    nm=len(omeg)
    
    dphi = np.pi/4 #Discretizing the phase
    phasebins = np.arange(-np.pi,np.pi,dphi)
        
    nu = 1e-6 #kinematic viscosity
    z = np.linspace(0.00, 0.020, 100) #height vector
    t = np.linspace(-np.pi/np.nanmean(omeg),np.pi/np.nanmean(omeg),100) #time vector 
    
    omstokes = np.zeros((len(z),len(phasebins),nm))
    
    for k in range(len(omeg)):
        omega = omeg[k]
        u0 = amp[k]
        p = phi[k]
        uwave = np.zeros((len(z),len(t))) #temporary array for given frequency
        for jj in range(len(z)):
            uwave[jj,:] = u0/len(omeg)*(np.cos(omega*t-p) - 
                        np.exp(-np.sqrt(omega/(2*nu))*z[jj])*np.cos(
                            (omega*t-p) - np.sqrt(omega/(2*nu))*z[jj]))
        huwave = sig.hilbert(np.nanmean(uwave,axis = 0))
        pw = np.arctan2(huwave.imag,huwave.real)
        
        ustokes = np.zeros((len(z),len(phasebins)))
        
        for ii in range(len(phasebins)):
           
                if ii == 0:
                    #For -pi
                    idx2 =  ( (pw >= phasebins[-1] + (dphi/2)) | (pw <= phasebins[0] + (dphi/2))) #Analytical
                else:
                    #For phases in the middle
                    idx2 = ((pw >= phasebins[ii]-(dphi/2)) & (pw <= phasebins[ii]+(dphi/2))) #analytical
               
                ustokes[:,ii] = np.nanmean(uwave[:,idx2],axis = 1)  #Averaging
        
                omstokes[:,ii,k] = ustokes[:,ii]
                
                
    omsum2 = np.zeros((len(phasebins),len(z)))
    
    for k in range(len(z)):
        for i in range(len(phasebins)):
            omsum2[i,k] = np.nansum(omstokes[k,i,:])
            
    if(plot):
        fig,ax=plt.subplots()
        for ii in range(len(phasebins)):
            colorstr = 'C' + str(ii)
            delta = np.sqrt(2e-6/2.4)
            ax.plot(omsum2[ii,:]/np.sqrt(u0),z+offset_fits[N],'-',color = colorstr,label = r'$\theta = $' + phasebins2[ii])
            ax.set_xlabel(r'$\tilde{u}$')    
            ax.set_ylabel(r'$z$')
            ax.set_title('from real [waverange] ' + str(N))
            ax.plot(phase_obs[N,ii],zs[N],'k-')
    return omsum2, z
        
#%%
#find delta based on max value
def get_delta(profiles,z):
    plt.figure()
    delta = np.zeros(len(phasebins))
    znew = np.linspace(0,0.013,1000)
    for ii in range(len(phasebins)):
        prof1 = profiles[ii,:]
        prof = np.abs(profiles[ii,:])
        f = interpolate.interp1d(z, naninterp(prof), kind='cubic')
        prof2 = f(znew)
        dp = np.gradient(prof2)/np.gradient(znew)
        d2p = np.gradient(dp)/np.gradient(znew)
        plt.plot(prof1,z,'k-');plt.plot(dp/300,znew,'r--');plt.plot(d2p/300000,znew,'b:')
        idx = np.nanargmax(prof2)
        #id2 = znew[dp<0][znew>0.002][0]
        id2 = np.nanargmin(d2p)
        plt.hlines(znew[id2],-1,1)
        delta[ii] = znew[id2]
    return delta
#%%
#find delta based on max value
def get_delta_2(profiles,z):
    plt.figure()
    delta = np.zeros(len(phasebins))
    znew = np.linspace(0,0.013,1000)
    for ii in range(len(phasebins)):
        prof1 = profiles[ii,:]
        prof = np.abs(profiles[ii,:])
        f = interpolate.interp1d(z, naninterp(prof1), kind='cubic')
        prof2 = f(znew)
        dp = np.gradient(prof2)/np.gradient(znew)
        d2p = np.gradient(dp)/np.gradient(znew)
        plt.plot(prof1,z,'k-');plt.plot(dp/300,znew,'r--');plt.plot(d2p/300000,znew,'b:')
        idx = np.nanargmax(prof2)
        #id2 = znew[dp<0][znew>0.002][0]
        #delta[ii]=znew[np.abs(dp)<0.001][0]
        mask = (np.abs(dp/300)<0.05)&(np.abs(prof2)>0.05)
        delta[ii]=znew[mask][0]
        id2 = np.nanargmax(dp)
        plt.hlines(znew[id2],-1,1)
        #delta[ii] = znew[id2]
    return delta
#%%
#find delta for every construction
#cdelta = np.zeros((384,8))
for N in range(10):
    try:
        cprof, z= construct(N,True)
        print('constructed ' + str(N))
        cdelta[N,:] = get_delta(cprof,z)
        print('found delta ' + str(N))
    except:
        print(N)
#%%
#find delta for every observation
#this is a bad way to do delta for the observations because they are just
#not shaped that way at the highest phases. if would work for a middle phase
# the green one is probably the most reliable
#odelta = np.zeros((384,8))
for N in range(10,15):
    try:
        oprof = phase_obs[N]
        z = zs[N]
        odelta[N,:] = get_delta_2(oprof,z)
    except:
        print(N)
#%%
plt.figure()
for ii in range(8):
    x = phasebins[ii]
    y = odelta[:,ii]
    plt.errorbar(x,np.nanmean(y),yerr=np.std(y),fmt = 'ko', capsize = 2)
        
#%%
def wvd(u,fs,component,plot = False):
    
    """A method to decompose the wave velocity from the full velocity
    signal. Based on the Bricker & Monismith phase method, but returns the
    full timeseries via IFFT, rather than the wave stresses via spectral sum.
    
    Parameters
    ----------
    u: 1d numpy array
      A velocity time series vector
    
    fs: float
      Sampling frequency (Hz)
    
    component: str
      Either 'u' or 'w' for horizontal and vertical velocity, respectively
      
    plot: bool
      if True, plots decomposed power spectrum. defaults to False
     
    Returns
    ---------
    uw: 1d numpy array
      The wave velocity time series vector
    
    
    """

    n = len(u)
    nfft = n
    
    u = sig.detrend(u)
    
    #amplitude of the wave component
    Amu = np.fft.fft(naninterp(u))/np.sqrt(n)
      
    #degrees of freedom
    df = fs/(nfft-1)
    
    #nyquist frequency: limit of the analysis given the sampling
    nnyq = int(np.floor(nfft/2 +1))
    fm = np.arange(0,nnyq)*df
       
    #Phase
    uph = np.arctan2(np.imag(Amu),np.real(Amu)).squeeze()[:nnyq]
    
    #Computing the full spectra
    Suu = np.real(Amu*np.conj(Amu))/(nnyq*df)
    Suu = Suu.squeeze()[:nnyq]
    
       
    offset = np.sum(fm<=0.1)
    
    uumax = np.argmax(Suu[(fm>0.1) & (fm < 0.7)]) + offset
       
    if component == 'u':
        widthratiolow = 6
        widthratiohigh = 6
    elif component == 'w':
        widthratiolow = 5
        widthratiohigh = 3
    
    
    fmmax = fm[uumax]
    step = 0.2*fmmax
    waverange = np.arange(uumax - (fmmax/widthratiolow)//df,uumax + (fmmax/widthratiohigh)//df).astype(int)  
    #waverange = np.arange(uumax - (step)//df,uumax + (1*step)//df).astype(int)
    interprangeu = np.arange(1,np.nanargmin(np.abs(fm-1))).astype(int)
    waverange = waverange[(waverange>=0) & (waverange<nnyq)]
    interprangeu = interprangeu[(interprangeu >= 0) & (interprangeu < nnyq)]
    
    
    Suu_turb = Suu[interprangeu]
    fmuu = fm[interprangeu]
    Suu_turb = np.delete(Suu_turb,waverange-interprangeu[0])
    fmuu = np.delete(fmuu,waverange-interprangeu[0])
    Suu_turb = Suu_turb[fmuu>0]
    fmuu = fmuu[fmuu>0]
    
       
    #Linear interpolation over turbulent spectra
                        
    F = np.log(fmuu)
    S = np.log(Suu_turb)
    Puu = np.polyfit(F,S,deg = 1)
    Puuhat = np.exp(np.polyval(Puu,np.log(fm)))
    
    #Plotting to test the code
    if plot:
        plt.figure()
        plt.loglog(fmuu,Suu_turb,'ko',ms=1,label='non-wave signal')
        plt.loglog(fm[waverange],Suu[waverange],'r+',label='wave signal')
        plt.loglog(fm,Puuhat,'b-',label='interpolation line')
    
    
    #Wave spectra
    Suu_wave = Suu[waverange] - Puuhat[waverange]
    
    Amuu_wave = np.sqrt((Suu_wave+0j)*df*nnyq)
    
    #phase
    Phase = np.arctan2(np.imag(Amu),np.real(Amu)).squeeze()
    
    Amp = np.zeros_like(fm)
    Amp[waverange] = Amuu_wave
    #Amp = np.concatenate((Amp[1:],np.flipud(Amp[1:])))
    Amp = np.concatenate((Amp[:-1],np.flipud(Amp[:-1])))
    if (len(Amp) == len(Phase)-1):
        Amp = np.zeros_like(fm)
        Amp[waverange] = Amuu_wave
        Amp = np.concatenate((Amp[:],np.flipud(Amp[:-1])))
    
    
    z = Amp*(np.cos(Phase) + 1j*np.sin(Phase))
    
    uw = np.fft.ifft(z)*np.sqrt(n)
    
    omega = np.pi * 2 * fm
    omega = np.concatenate((omega[:-1],np.flipud(omega[:-1])))

    return Amp[waverange], Phase[waverange], omega[waverange]