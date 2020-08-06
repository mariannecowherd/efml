#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 16:03:03 2020

@author: gegan, marianne

@title: vectrinofuncs
"""


'''
Table of Contents:
pa_rotation
wave_vel_decomp
displacement_thickness
calculate_fft
get_turb_waves
remove_outliers
nanrm2
naninterp
'''

#packages used in all functions
#relevant packages are imported as part of the user functions
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import copy
import scipy
import scipy.interpolate


#user defined functions

def pa_rotation(u,v,theta):
    import numpy as np
    
    #Storing as complex variable w = u + iv
    w = u + 1j*v
    
    wr = w*np.exp(-1j*theta*np.pi/180)
    vel_maj = np.real(wr)
    vel_min = np.imag(wr)
    
    return vel_maj,vel_min

def wave_vel_decomp(u,fs,component,plot = False):
    
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
    import scipy.signal as sig
    import numpy as np
    from vectrinofuncs import naninterp
    import matplotlib.pyplot as plt
    
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
    elif component == 'v':
        widthratiolow = 5
        widthratiohigh = 3
    
    
    fmmax = fm[uumax]
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
    
    return uw.real

def displacement_thickness(uprof,z):
    """Calculates a modified displacement thickness for phase resolved boundary layer
    velocity profiles. 
    
    uprof is an n x p array where n is the number of vertical
    measurement bins, and p is the number of phases. 
    
    z is the vertical coordinate and is an n x 1 array. 
    
    """
    import numpy as np
    
    int_range = ((z > 0) & (z < 0.011)) #keep it within the good SNR range
    z_int = np.flipud(z[int_range]) #Flipping for the integral
    uprof_int = np.abs(np.flipud(uprof[int_range,:]))
    
    umax = np.nanmax(uprof_int, axis = 0) #Finding maximum velocity, wherever it may be
    idxmax = np.nanargmax(uprof_int, axis = 0)
    
    delta_1 = np.zeros((uprof.shape[1],))
    for i in range(len(delta_1)):
        delta_1[i] = np.trapz(1 - uprof_int[:idxmax[i],i]/umax[i],z_int[:idxmax[i]])
    
    
    return delta_1

def displacement_thickness_interp(uprof,z):
    """Calculates a modified displacement thickness for phase resolved boundary layer
    velocity profiles. 
    
    uprof is an n x p array where n is the number of vertical
    measurement bins, and p is the number of phases. 
    
    z is the vertical coordinate and is an n x 1 array. 
    
    Use this function when calculating on interpolated, deployment-averaged profiles
    """
    import numpy as np
    
    int_range = ((z > 0) & (z < 0.011)) #keep it within the good SNR range
    z_int = z[int_range] #Flipping for the integral
    uprof_int = np.abs(uprof[int_range,:])
    
    umax = np.nanmax(uprof_int, axis = 0) #Finding maximum velocity, wherever it may be
    idxmax = np.nanargmax(uprof_int, axis = 0)
    
    delta_1 = np.zeros((uprof.shape[1],))
    for i in range(len(delta_1)):
        delta_1[i] = np.trapz(1 - uprof_int[:idxmax[i],i]/umax[i],z_int[:idxmax[i]])
    
    
    return delta_1

def calculate_fft(x,nfft):
    
    import copy
    import numpy as np
    import scipy
    
    X = copy.deepcopy(x)
    
    if X.ndim==1:
        n = np.size(X)
    elif X.ndim==2:
        n,m = np.shape(X)
        if m>n:
            X = X.T
            n,m = np.shape(X)
    
    #num = int(np.floor(4*n/nfft) - 3)
    num = 1
    
    X = X - np.mean(X)
    
    #jj = np.arange(0,nfft)
    
    WIN = np.hamming(n)
    
    A = np.zeros((num,nfft),dtype = np.complex128)

    varXwindtot = 0    
    
    for ii in range(num):
        istart = int(np.ceil(ii*n/4))
        istop = int(np.floor(istart + n))  
        Xwind = X[istart:istop].squeeze()
        lenX = len(Xwind)
        Xwind = Xwind - np.mean(Xwind) #de-mean
        varXwind = np.dot(Xwind,Xwind)/lenX
        Xwind = scipy.signal.detrend(Xwind)
        
        varXwindtot = varXwindtot + varXwind
        Xwind = Xwind*WIN
        tmp = np.dot(Xwind,Xwind)/lenX
        
        if tmp == 0:
            Xwind = Xwind*0
        else:
            Xwind = Xwind*np.sqrt(varXwind/tmp)
            
        Xwind = np.pad(Xwind,(0,nfft-n),'constant')
        
        A[ii,:] = np.fft.fft(Xwind.T)/np.sqrt(n)
        


def get_turb_waves(vec,fs,method):
    
    import numpy as np
    import copy
    from vectrinofuncs import naninterp
    from vectrinofuncs import calculate_fft
    
    waveturb = dict()
    #Implement Bricker and Monismith method
    if method == 'phase':
    
        u = copy.deepcopy(vec['velmaj'])
        v = copy.deepcopy(vec['velmin'])
        w1 = copy.deepcopy(vec['w1'])
        w2 = copy.deepcopy(vec['w1'])
        
        m,n = np.shape(u)
        
        waveturb = dict()  
        
        #Turbulent reynolds stresses
        waveturb['uw1'] = np.empty((m,))*np.NaN
        waveturb['vw1'] = np.empty((m,))*np.NaN
        waveturb['uw2'] = np.empty((m,))*np.NaN
        waveturb['vw2'] = np.empty((m,))*np.NaN
        waveturb['uv'] = np.empty((m,))*np.NaN
        waveturb['uu'] = np.empty((m,))*np.NaN
        waveturb['vv'] = np.empty((m,))*np.NaN
        waveturb['w1w1'] = np.empty((m,))*np.NaN
        waveturb['w2w2'] = np.empty((m,))*np.NaN
        waveturb['w1w2'] = np.empty((m,))*np.NaN
        
        
        #Wave reynolds stresses
        waveturb['uw1_wave'] = np.empty((m,))*np.NaN
        waveturb['vw1_wave'] = np.empty((m,))*np.NaN
        waveturb['uw2_wave'] = np.empty((m,))*np.NaN
        waveturb['vw2_wave'] = np.empty((m,))*np.NaN
        waveturb['uv_wave'] = np.empty((m,))*np.NaN
        waveturb['uu_wave'] = np.empty((m,))*np.NaN
        waveturb['vv_wave'] = np.empty((m,))*np.NaN
        waveturb['w1w1_wave'] = np.empty((m,))*np.NaN
        waveturb['w2w2_wave'] = np.empty((m,))*np.NaN
        waveturb['w1w2_wave'] = np.empty((m,))*np.NaN
        
        for jj in range(vec['z'].size):
            
            if np.sum(np.isnan(u[jj,:])) < np.size(u[jj,:]/2):
                
                nfft = (2**(np.ceil(np.log2(np.abs(u[jj,:].size))))).astype(int)
                Amu = calculate_fft(naninterp(u[jj,:]),nfft)
                Amv = calculate_fft(naninterp(v[jj,:]),nfft)
                Amw1 = calculate_fft(naninterp(w1[jj,:]),nfft)
                Amw2 = calculate_fft(naninterp(w2[jj,:]),nfft)
                
                df = fs/(nfft-1)
                nnyq = int(np.floor(nfft/2 +1))
                fm = np.arange(0,nnyq)*df
                   
                #Phase
                
                Uph = np.arctan2(np.imag(Amu),np.real(Amu)).squeeze()[:nnyq]
                Vph = np.arctan2(np.imag(Amv),np.real(Amv)).squeeze()[:nnyq]
                W1ph = np.arctan2(np.imag(Amw1),np.real(Amw1)).squeeze()[:nnyq]
                W2ph = np.arctan2(np.imag(Amw2),np.real(Amw2)).squeeze()[:nnyq]
                
                #Computing the full spectra
                
                Suu = np.real(Amu*np.conj(Amu))/(nnyq*df)
                Suu = Suu.squeeze()[:nnyq]
                
                Svv = np.real(Amv*np.conj(Amv))/(nnyq*df)
                Svv = Svv.squeeze()[:nnyq]
                
                Sww1 = np.real(Amw1*np.conj(Amw1))/(nnyq*df)
                Sww1 = Sww1.squeeze()[:nnyq]
                
                Sww2 = np.real(Amw2*np.conj(Amw2))/(nnyq*df)
                Sww2 = Sww2.squeeze()[:nnyq]
                
                Suv = np.real(Amu*np.conj(Amv))/(nnyq*df)
                Suv = Suv.squeeze()[:nnyq]
                
                Suw1 = np.real(Amu*np.conj(Amw1))/(nnyq*df)
                Suw1 = Suw1.squeeze()[:nnyq]
                
                Suw2 = np.real(Amu*np.conj(Amw2))/(nnyq*df)
                Suw2 = Suw2.squeeze()[:nnyq]
                
                Svw1 = np.real(Amv*np.conj(Amw1))/(nnyq*df)
                Svw1 = Svw1.squeeze()[:nnyq]
                
                Svw2 = np.real(Amv*np.conj(Amw2))/(nnyq*df)
                Svw2 = Svw2.squeeze()[:nnyq]
                
                Sw1w2 = np.real(Amw1*np.conj(Amw2))/(nnyq*df)
                Sw1w2 = Sw1w2.squeeze()[:nnyq]
                
                
                offset = np.sum(fm<=0.1)
                
                uumax = np.argmax(Suu[(fm>0.1) & (fm < 0.7)]) + offset
                
                #This is the range under which you interpolate--see paper for details, 
                #but generally you'll want to examine a few spectra by eye to determine how
                #wide the wave peak is
                
                widthratiolow = 2.333
                widthratiohigh = 1.4
                fmmax = fm[uumax]
                waverange = np.arange(uumax - (fmmax/widthratiolow)//df,uumax + (fmmax/widthratiohigh)//df).astype(int)
                
                interprange = np.arange(1,np.nanargmin(np.abs(fm - 1))).astype(int)
                
                interprangeW = np.arange(1,np.nanargmin(np.abs(fm-1))).astype(int)
                
                interprange = interprange[(interprange>=0) & (interprange<nnyq)]
                waverange = waverange[(waverange>=0) & (waverange<nnyq)]
                interprangeW = interprangeW[(interprangeW >= 0) & (interprangeW < nnyq)]
                
                Suu_turb = Suu[interprange]
                fmuu = fm[interprange]
                Suu_turb = np.delete(Suu_turb,waverange-interprange[0])
                fmuu = np.delete(fmuu,waverange-interprange[0])
                Suu_turb = Suu_turb[fmuu>0]
                fmuu = fmuu[fmuu>0]
                
                Svv_turb = Svv[interprange]
                fmvv = fm[interprange]
                Svv_turb = np.delete(Svv_turb,waverange-interprange[0])
                fmvv = np.delete(fmvv,waverange-interprange[0])
                Svv_turb = Svv_turb[fmvv>0]
                fmvv = fmvv[fmvv>0]
                
                Sww1_turb = Sww1[interprangeW]
                fmww1 = fm[interprangeW]
                Sww1_turb = np.delete(Sww1_turb,waverange-interprangeW[0])
                fmww1 = np.delete(fmww1,waverange-interprangeW[0])
                Sww1_turb = Sww1_turb[fmww1>0]
                fmww1 = fmww1[fmww1>0]
                
                Sww2_turb = Sww2[interprangeW]
                fmww2 = fm[interprangeW]
                Sww2_turb = np.delete(Sww2_turb,waverange-interprangeW[0])
                fmww2 = np.delete(fmww2,waverange-interprangeW[0])
                Sww2_turb = Sww2_turb[fmww2>0]
                fmww2 = fmww2[fmww2>0]
                
                #Linear interpolation over turbulent spectra
                F = np.log(fmuu)
                S = np.log(Suu_turb)
                Puu = np.polyfit(F,S,deg = 1)
                Puuhat = np.exp(np.polyval(Puu,np.log(fm)))
                
                F = np.log(fmvv)
                S = np.log(Svv_turb)
                Pvv = np.polyfit(F,S,deg = 1)
                Pvvhat = np.exp(np.polyval(Pvv,np.log(fm)))
                                      
                F = np.log(fmww1)
                S = np.log(Sww1_turb)
                Pww1 = np.polyfit(F,S,deg = 1)
                Pww1hat = np.exp(np.polyval(Pww1,np.log(fm)))
                
                F = np.log(fmww2)
                S = np.log(Sww2_turb)
                Pww2 = np.polyfit(F,S,deg = 1)
                Pww2hat = np.exp(np.polyval(Pww2,np.log(fm)))
                
                
                #Wave spectra
                Suu_wave = Suu[waverange] - Puuhat[waverange]
                Svv_wave = Svv[waverange] - Pvvhat[waverange]
                Sww1_wave = Sww1[waverange] - Pww1hat[waverange]
                Sww2_wave = Sww2[waverange] - Pww2hat[waverange]
                
                
                
                ##                
                #This should maybe be nnyq*df? But then the amplitudes are way too big
                Amu_wave = np.sqrt((Suu_wave+0j)*(df))
                Amv_wave = np.sqrt((Svv_wave+0j))*(df)
                Amww1_wave = np.sqrt((Sww1_wave+0j)*(df))
                Amww2_wave = np.sqrt((Sww2_wave+0j)*(df))
                
                #Wave Magnitudes
                Um_wave = np.sqrt(np.real(Amu_wave)**2 + np.imag(Amu_wave)**2)
                Vm_wave = np.sqrt(np.real(Amv_wave)**2 + np.imag(Amv_wave)**2)
                W1m_wave = np.sqrt(np.real(Amww1_wave)**2 + np.imag(Amww1_wave)**2)
                W2m_wave = np.sqrt(np.real(Amww2_wave)**2 + np.imag(Amww2_wave)**2)
                
                
                #Wave reynolds stress
                uw1_wave = np.nansum(Um_wave*W1m_wave*np.cos(W1ph[waverange]-Uph[waverange]))
                uw2_wave = np.nansum(Um_wave*W2m_wave*np.cos(W2ph[waverange]-Uph[waverange]))
                uv_wave =  np.nansum(Um_wave*Vm_wave*np.cos(Vph[waverange]-Uph[waverange]))
                vw1_wave = np.nansum(Vm_wave*W1m_wave*np.cos(W1ph[waverange]-Vph[waverange]))
                vw2_wave = np.nansum(Vm_wave*W2m_wave*np.cos(W2ph[waverange]-Vph[waverange]))
                w1w2_wave = np.nansum(W1m_wave*W2m_wave*np.cos(W2ph[waverange]- W1ph[waverange]))
                
                uu_wave = np.nansum(Suu_wave*df)
                vv_wave = np.nansum(Svv_wave*df)
                w1w1_wave = np.nansum(Sww1_wave*df)
                w2w2_wave = np.nansum(Sww2_wave*df)
                
                
                            
                #Full reynolds stresses
                uu = np.nansum(Suu*df)
                uv = np.nansum(Suv*df)
                uw1 = np.nansum(Suw1*df)
                uw2 = np.nansum(Suw2*df)
                vv = np.nansum(Svv*df)
                vw1 = np.nansum(Svw1*df)
                vw2 = np.nansum(Svw2*df)
                w1w1 = np.nansum(Sww1*df)
                w2w2 = np.nansum(Sww2*df)
                w1w2 = np.nansum(Sw1w2*df)
                
                #Turbulent reynolds stresses
                
                upup = uu - uu_wave
                vpvp = vv - vv_wave
                w1pw1p = w1w1 - w1w1_wave
                w2pw2p = w2w2 - w2w2_wave
                upw1p = uw1 - uw1_wave
                upw2p = uw2 - uw2_wave
                upvp = uv - uv_wave
                vpw1p = vw1 - vw1_wave
                vpw2p = vw2 - vw2_wave
                w1pw2p = w1w2 - w1w2_wave
                
                #Turbulent reynolds stresses
                waveturb['uw1'][jj] = upw1p
                waveturb['vw1'][jj] = vpw1p
                waveturb['uw2'][jj] = upw2p
                waveturb['vw2'][jj] = vpw2p
                waveturb['uv'][jj] = upvp
                waveturb['uu'][jj] = upup
                waveturb['vv'][jj] = vpvp
                waveturb['w1w1'][jj] = w1pw1p
                waveturb['w2w2'][jj] = w2pw2p
                waveturb['w1w2'][jj] = w1pw2p
                
                #Wave reynolds stresses
                waveturb['uw1_wave'][jj] = uw1_wave
                waveturb['vw1_wave'][jj] = vw1_wave
                waveturb['uw2_wave'][jj] = uw2_wave
                waveturb['vw2_wave'][jj] = vw2_wave
                waveturb['uv_wave'][jj] = uv_wave
                waveturb['uu_wave'][jj] = uu_wave
                waveturb['vv_wave'][jj] = vv_wave
                waveturb['w1w1_wave'][jj] = w1w1_wave
                waveturb['w2w2_wave'][jj] = w2w2_wave
                waveturb['w1w2_wave'][jj] = w1w2_wave
        
    return waveturb



def remove_outliers(x,y,outmethod):
    
    import numpy as np
    from sklearn.decomposition import PCA
    
    if outmethod == 'std':
        K = 4
        badidx = ((y > np.nanmedian(y) + K*np.nanstd(y)) | (y < np.nanmedian(y) - K*np.nanstd(y)))
        return (x[~badidx],y[~badidx])
    
    elif outmethod == 'pca':
        X = np.vstack((x,y)).T
        X = (X - np.nanmean(X,axis = 0))/np.nanstd(X,axis = 0)

        pca = PCA(n_components = 2)
        pca.fit(X)
        
        S = pca.transform(X)
        ci95 = 1.96*np.std(S[:,1]) #Can fiddle with this to do more or less than 1.96
        indremove = ((S[:,1] > ci95) | (S[:,1] < -ci95))
        return (x[~indremove],y[~indremove])
    
    else:
        return (x,y)

def nanrm2(x,y):
    
    import numpy as np
    
    y = y[~np.isnan(x)]
    x = x[~np.isnan(x)]
    
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    return(x,y)


def calculate_fft(x,nfft):
    
    import copy
    import numpy as np
    import scipy
    
    X = copy.deepcopy(x)
    
    if X.ndim==1:
        n = np.size(X)
    elif X.ndim==2:
        n,m = np.shape(X)
        if m>n:
            X = X.T
            n,m = np.shape(X)
    
    #num = int(np.floor(4*n/nfft) - 3)
    num = 1
    
    X = X - np.mean(X)
    
    #jj = np.arange(0,nfft)
    
    WIN = np.hamming(n)
    
    A = np.zeros((num,nfft),dtype = np.complex128)

    varXwindtot = 0    
    
    for ii in range(num):
        istart = int(np.ceil(ii*n/4))
        istop = int(np.floor(istart + n))  
        Xwind = X[istart:istop].squeeze()
        lenX = len(Xwind)
        Xwind = Xwind - np.mean(Xwind) #de-mean
        varXwind = np.dot(Xwind,Xwind)/lenX
        Xwind = scipy.signal.detrend(Xwind)
        
        varXwindtot = varXwindtot + varXwind
        Xwind = Xwind*WIN
        tmp = np.dot(Xwind,Xwind)/lenX
        
        if tmp == 0:
            Xwind = Xwind*0
        else:
            Xwind = Xwind*np.sqrt(varXwind/tmp)
            
        Xwind = np.pad(Xwind,(0,nfft-n),'constant')
        
        A[ii,:] = np.fft.fft(Xwind.T)/np.sqrt(n)
        
    
    return A

def naninterp(x):
    import numpy
    import scipy
    
    if ~np.all(np.isnan(x)):
    
        if np.sum(~np.isnan(x)) >= 2:
            f = scipy.interpolate.interp1d(np.reshape(np.array(np.where(~np.isnan(x))),(np.size(x[~np.isnan(x)]),)),x[~np.isnan(x)],
             kind = 'linear',bounds_error =False)
            xnew = np.where(np.isnan(x))
            x[np.isnan(x)]=f(xnew).squeeze()
        
        if np.sum(~np.isnan(x)) >= 2:
            f = scipy.interpolate.interp1d(np.reshape(np.array(np.where(~np.isnan(x))),(np.size(x[~np.isnan(x)]),)),x[~np.isnan(x)],
             kind = 'nearest',fill_value = 'extrapolate')
            xnew = np.where(np.isnan(x))
            x[np.isnan(x)]=f(xnew).squeeze()
    
    return x


def get_dissipation(vectrino,fs,method):
    import numpy as np
    import scipy
    import datetime as dt
    
    if method == 'structure':
        
        probe = 'w2'
        #Calculating w_prime
        m,n = np.shape(vectrino[probe])
        wp = np.zeros((m,n))
        wbar = np.nanmean(vectrino[probe],axis = 1)
        for ii in range(n):
            wp[:,ii] = vectrino[probe][:,ii] - wbar
        
        z = vectrino['z']
        z = z[z>0]
        dz = np.diff(z)
        
        #Want at least 5 above/below following Truleo
        zeps = z[5:-5]
        eps = np.zeros((len(zeps),))
#        D = np.zeros((len(zeps),5))
#        r = np.arange(2,12,2)*np.abs(dz[0])
        def structfunc(r,N,A):
            return N + A*(r**(2/3))        
        
        for ii in range(len(zeps)):
            idx = np.argmin(np.abs(z-zeps[ii]))
            numr = np.min([np.sum(np.arange(len(z))>idx), np.sum(np.arange(len(z)) < idx)])
            
            r = np.linspace(2,2*numr,numr)*np.abs(dz[0])
            D = np.empty_like(r)
            for jj in range(len(r)):
                D[jj] = np.nanmean((wp[idx-jj,:] - wp[idx+jj,:])**2)
                
            try:
                p0, cov = scipy.optimize.curve_fit(structfunc,r,D, p0 = (1e-5,1e-3),maxfev = 10000)
                N = p0[0]
                A = p0[1]
                eps[ii] = (A/2.1)**(3/2)
                
##                #Test plotting
#                plt.figure(ii)
#                plt.plot(r,D,'k*')
#                plt.plot(r,N + A*(r**(2/3)),'r:')     
            except RuntimeError:
                print('Curve fit failed')
                eps[ii] = np.NaN
            except TypeError:
                print('Curve fit failed')
                eps[ii] = np.NaN

        return eps,zeps
    
    elif method == 'Fedd07':
        
        def calcFeddA6(sig1,sig2,sig3,u1,u2,omega):
            arrlen = 120
            rho = np.logspace(-2,4,np.int_(arrlen))
            theta = np.linspace(-np.pi,np.pi,np.int_(arrlen/10))
            phi = np.linspace(-np.pi/2,np.pi/2,np.int_(arrlen/10))
                    
            #This way is pretty slow but I think it works
            I3 = np.zeros((len(theta),len(phi),len(rho)))
            for jj in range(len(phi)):
                for kk in range(len(rho)):
                    gamma = np.sqrt((np.cos(phi[jj])**2)*(((np.cos(theta)**2)/sig1**2) + ((np.sin(theta)**2)/sig2**2))
                        + ((np.sin(phi[jj])**2)/(sig3**2)) )
                    ksquare = (gamma**2)*(rho[kk]**2)
                    k3 = rho[kk]*np.sin(phi[jj])/sig3
                    
                    I3[:,jj,kk] = (gamma**(-11./3))*(1-(k3*k3/ksquare))*np.exp((-(rho[kk]*np.cos(phi[jj])*(
                        (np.cos(theta)*u1/sig1) + (np.sin(theta)*u2/sig2))-omega)**2)/(2*rho[kk]**2))*(
                        np.cos(phi[jj])/(sig1*sig2*sig3))
            I2 = np.trapz(I3,theta,axis = 0)
            
            I1 = np.trapz(I2,phi,axis = 0)

            M33 = np.trapz(rho**(-8./3)*I1,rho)
            return M33
        
        m,n = np.shape(vectrino['velmaj'])
        
        eps = np.zeros((m,))
        omega_range = [2*np.pi*4,2*np.pi*6]
        alpha = 1.5
            
        for ii in range(m):
            u = vectrino['velmaj'][ii,:]
            v = vectrino['velmin'][ii,:]
            w = vectrino['w1'][ii,:]
            
            if np.sum(np.isnan(u)) < len(u)/2:
            
                fw,Pw = sig.welch(w,fs = fs, window = 'hamming', nperseg = len(w)//20,
                                      detrend = 'linear')
                
                
#                noiserange = (fw>=20) & (fw<=30) #Customize this based on instrument and spectra
#                noiselevel = np.nanmean(Pw[noiserange])
                
                omega = 2*np.pi*fw
                
                inds = (omega > omega_range[0]) & (omega < omega_range[1])
                omega = omega[inds]
                Pw = (Pw[inds])/(2*np.pi)
              
                sig1 = np.std(u)
                sig2 = np.std(v)
                sig3 = np.std(w)
                
                u1 = np.abs(np.nanmean(u))
                u2 = np.abs(np.nanmean(v))
                
                
                M33 = np.zeros_like(omega)
                for jj in range(len(omega)): 
                    M33[jj] = calcFeddA6(sig1,sig2,sig3,u1,u2,omega[jj])
                
                epsomega = ((Pw*2*(2*np.pi)**(3/2))/(alpha*M33))**(3/2)
                
                eps[ii] = np.nanmean(epsomega)
        
        return eps
    
    elif method == 'TE01':
        
        #Much of this code taken from Dolfyn package, reworked for Vectrino Profiler
        
        
        def calcA13(sigV,theta):
        #Integral from appendix
           x = np.arange(-20,20,1e-2)
           out = np.empty_like(sigV.flatten())
           for i, (b,t) in enumerate(zip(sigV.flatten(),theta.flatten())):
               out[i] = np.trapz(np.cbrt(x ** 2 - 2 / b * np.cos(t) * x + 
                  b ** (-2)) *np.exp(-0.5 * x ** 2), x)
        
           return out.reshape(sigV.shape)*(2*np.pi)**(-0.5) * sigV**(2/3)
  
        def up_angle(u,v):
            Uh = naninterp(u) + 1j*naninterp(v)
            dt = sig.detrend(Uh)
            fx = dt.imag <= 0
            dt[fx] = dt[fx]*np.exp(1j * np.pi)
            return np.angle(np.mean(dt,-1,dtype = np.complex128))
        
        def U_angle(u,v):
            n = np.nanmean(v)
            e = np.nanmean(u)
            return np.arctan2(n,e)
        
        m,n = np.shape(vectrino['velmaj'])
        
        eps = np.zeros((m,))
        omega_range = [2*np.pi*4,2*np.pi*6] #Customize this based on instrument and spectra
   
        for ii in range(m):
            u = vectrino['velmaj'][ii,:]
            v = vectrino['velmin'][ii,:]
            w = vectrino['w1'][ii,:]
            
            
            if np.sum(np.isnan(u)) < len(u)/2:
                V = np.sqrt(np.nanmean(u**2 + v**2))
                sigma = np.std(np.sqrt(u**2+v**2))
                
                thetaup = up_angle(u,v)
                thetaU = U_angle(u,v)
                theta = thetaU - thetaup
                
                alpha = 1.5
                intgrl = calcA13(sigma/V,theta)
                
                fu,Pu = sig.welch(u,fs = fs, window = 'hamming', nperseg = len(u)//20,
                                  detrend = 'linear')
                fv,Pv = sig.welch(v,fs = fs, window = 'hamming', nperseg = len(v)//20,
                                  detrend = 'linear')
                fw,Pw = sig.welch(w,fs = fs, window = 'hamming', nperseg = len(w)//20,
                                  detrend = 'linear')
                
                noiserange = (fu>=20) & (fu<=30) #Customize this based on instrument and spectra
                noiselevel = np.nanmean(Pu[noiserange] + Pv[noiserange])
                
                omega = 2*np.pi*fu
                inds = (omega > omega_range[0]) & (omega < omega_range[1])
                omega = omega[inds]
                Pu = Pu[inds]/(2*np.pi)
                Pv = Pv[inds]/(2*np.pi)
                Pw = Pw[inds]/(2*np.pi)
                
                uv = (np.mean((Pu + Pv - noiselevel)*(omega)**(5/3))/
                      (21/55*alpha*intgrl))**(3/2)/V
                      
                #Adding w component
                uv += (np.mean((Pw)*(omega)**(5/3))/
                      (12/55*alpha*intgrl))**(3/2)/V
                
               # Averaging
                uv *= 0.5
                
                eps[ii] = uv
            else:
                eps[ii] = np.NaN
        return eps
    
    elif method == 'full':
    
        u = vectrino['velmaj']
        v = vectrino['velmin']
        w = vectrino['w1']
        
        #Calculating fluctuating velocities
        m,n = np.shape(u)
        up = np.zeros((m,n))
        vp = np.zeros((m,n))
        wp = np.zeros((m,n))
        
        
        for ii in range(m):
            if np.sum(np.isnan(u[ii,:])) < len(u[ii,:])/2:
                fu,Pu = sig.welch(u[ii,:],fs = fs, window = 'hamming', nperseg = n//50, detrend = 'linear')
                fv,Pv = sig.welch(v[ii,:],fs = fs, window = 'hamming', nperseg = n//50, detrend = 'linear')
                fw,Pw = sig.welch(w[ii,:],fs = fs, window = 'hamming', nperseg = n//50, detrend = 'linear')
                
                fumax = fu[np.argmax(Pu)]
                fvmax = fv[np.argmax(Pv)]
                fwmax = fw[np.argmax(Pv)]
                
                try:
                    bu,au = sig.butter(2,fumax/(fs/2))
                    bv,av = sig.butter(2,fvmax/(fs/2))
                    bw,aw = sig.butter(2,fwmax/(fs/2))
                except ValueError:
                    bu,au = sig.butter(2,.35/32)
                    bv,av = sig.butter(2,.35/32)
                    bw,aw = sig.butter(2,.35/32)

                
                ufilt = sig.filtfilt(bu,au,u[ii,:])
                vfilt = sig.filtfilt(bv,av,v[ii,:])
                wfilt = sig.filtfilt(bw,aw,w[ii,:])
                
                up[ii,:] = u[ii,:] - ufilt
                vp[ii,:] = v[ii,:] - vfilt
                wp[ii,:] = w[ii,:] - wfilt
        
        ubar = np.nanmean(u, axis = 1)
        vbar = np.nanmean(v, axis = 1)
        dudz = np.gradient(up,np.diff(vectrino['z'])[0],edge_order = 2 , axis = 0)
        dvdz = np.gradient(vp,np.diff(vectrino['z'])[0],edge_order = 2 , axis = 0)
        dwdz = np.gradient(wp,np.diff(vectrino['z'])[0],edge_order = 2 , axis = 0)
        
        dudt = np.gradient(up,(1./64),edge_order = 2, axis = 1)
        dvdt = np.gradient(vp,(1./64),edge_order = 2, axis = 1)
        dwdt = np.gradient(wp,(1./64),edge_order = 2, axis = 1)
        
        dudx = np.empty_like(dudz)
        dudy = np.empty_like(dudz)
        dvdx = np.empty_like(dvdz)
        dvdy = np.empty_like(dvdz)
        dwdx = np.empty_like(dwdz)
        dwdy = np.empty_like(dwdz)
        
        for ii in range(m):
            dudx[ii,:] = dudt[ii,:]/ubar[ii]
            dudy[ii,:] = dudt[ii,:]/vbar[ii]
            dvdx[ii,:] = dvdt[ii,:]/ubar[ii]
            dvdy[ii,:] = dvdt[ii,:]/vbar[ii]
            dwdx[ii,:] = dwdt[ii,:]/ubar[ii]
            dwdy[ii,:] = dwdt[ii,:]/vbar[ii]
        
        eps = np.zeros((m,))
        
        for ii in range(m):
            S11 = dudx[ii,:]
            S12 = 0.5*(dudy[ii,:] + dvdx[ii,:])
            S13 = 0.5*(dudz[ii,:] + dwdx[ii,:])
            S21 = copy.deepcopy(S12)
            S22 = dvdy[ii,:]
            S23 = 0.5*(dvdz[ii,:] + dwdy[ii,:])
            S31 = copy.deepcopy(S13)
            S32 = copy.deepcopy(S23)
            S33 = dwdz[ii,:]
            
            eps[ii] = 2*1e-6*np.nanmean(S11**2 + S12**2 + S13**2 + S21**2 + 
               S22**2 + S23**2 + S31**2 + S32**2 + S33**2)
        
        return eps
    
    if method == 'scaling':
        u = vectrino['velmaj']
        
        Tint, Lint, uprime = IntScales(u,fs)
        
        eps = uprime**3/Lint
        
        return eps
        