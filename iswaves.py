#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 22:49:12 2020

@author: marianne

@title: iswaves
"""

from dirs import dir_data

import sys
sys.path.append(dir_data)



def iswaves(N,ifplot):
    filepath = dir_data
    
    hparams = np.load(filepath + 'hydroparams/sdfix4_'+str(N)+'.npy',allow_pickle = True).item()
    vec = sio.loadmat(filepath + 'vectrino_' + str(N) + '.mat')
    
    #vertical average over some portion of the profile
    #u = np.nanmean(vec['velmaj'][4:12,:],axis = 0) 
    u = np.nanmean(vec['velmaj'],axis=0)
    
    try:
        #spectrum to find wave peak
        fu,pu = sig.welch(u,fs = 64, window = 'hamming', nperseg = len(u)//10,detrend = 'linear') 
        fmax = fu[np.nanargmax(pu)]
        pmax = fu[np.nanargmax(pu)]
        
        #fit a line to the spectrum
        mask1 = np.where((fu<5) & (fu>5e-1))
        mask2 = np.where((fu<5) & (fu>2e-1))
        p1=pu[mask1]
        f1=fu[mask1]
        p2=pu[mask2]
        f2=fu[mask2]
        
        a=np.polyfit(np.log(f1),np.log(p1),1)[0]
        b=np.polyfit(np.log(f1),np.log(p1),1)[1]
        fitline=fu**(a)*np.exp(b)
        
        dpdf=np.gradient(pu)
        peakids=[i for i in range(1,(len(dpdf)-1)) if np.sign(dpdf[i-1])!= np.sign(dpdf[i+1])]
        peakdiffs = pu[peakids]-fitline[peakids]
        
        #look at the 90th percentile of differences between the line and any peak
        #!!!! this percentile is arbitrary
        p = np.percentile(np.abs(peakdiffs), 90)
        mask3 = np.where((np.abs(peakdiffs))<p)
        meandiff=np.mean(peakdiffs[mask3])
        
        prange = np.abs(max(p2-fitline[mask2]))-np.abs(min(p2-fitline[mask2]))
        idx=np.where(p2==max(p2))[0]
        peaksize=(p2[idx]-fitline[mask2][idx])/meandiff
    except:
        peaksize=0
    
    if ifplot:
        fig = plt.figure(N)
        ax1=plt.subplot(1,2,1)
    
        ax1.loglog(fu,pu)
        ax1.loglog(f2,p2)
    
        ax1.loglog(fu,fitline)
        
        ax2=plt.subplot(1,2,2)
        ax2.semilogx(f2,p2-fitline[mask2])
        ax2.semilogx(f2,fitline[mask2]-fitline[mask2])
        ax2.set_title('diff ' + str(N))
        
        ax2.set_title('diff '+str(N)+' '+str(peaksize) + str(peaksize<-1000))
    output = np.abs(peaksize) > 1000
    #these should be no waves. method: I looked at them and thought this
    weirdones=[69, 44, 193, 166, 308]
    if N in weirdones: iswavy = False
    
    return output


