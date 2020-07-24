#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:30:56 2020

@author: marianne

@title: revised wave decomp 2, may 2020
"""
#%%
filepath = '/Users/marianne/Desktop/VectrinoSummer/'

#packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal as sig
import scipy.optimize

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.ticker
from sklearn.metrics import r2_score
#%%
rms = np.zeros([384,8])
rms_z = np.zeros([384,8])

r2 = np.zeros([384,8])
r2_z = np.zeros([384,8])
phase_obs = np.zeros([384,8,30])
theory_nu = np.zeros([384,8,30])
theory_nu_z = np.zeros([384,8,30])

nu_fits=np.zeros([384])
nu_fits_z = np.zeros([384,30])
offset_fits=np.zeros([384])
offset_fits_z=np.zeros([384,30])

fmaxes = np.zeros([384])
zs = np.zeros([384,30])
ds= np.zeros([384])
hcs = np.zeros([384])
waveturb=dict()
waveturb2=dict()

#formatting labels for the phase bins
phasebins2=[r'$-\pi$',r'$-3\pi/4$',r'$-\pi/2$',r'$-\pi/4$',r'$0$',r'$\pi/4$',r'$\pi/2$',r'$3\pi/4$']
#%%
for N in range(384):
    hparams = np.load('/Users/marianne/Desktop/VectrinoSummer/hydroparams/sdfix4_'+str(N)+'.npy',allow_pickle = True).item()
    ds[N] = hparams['d']
    hcs[N] = hparams['hc']
#%%
iswavy = np.empty(384,dtype=bool)
#check if the burst is sufficiently wavy for wave decomposition
for N in range(384): iswavy[N] = iswaves(N,False)

haswaves  = np.arange(384)[iswavy]
#%%
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


def make_stokes(phasebins,omega,u0,plot):
    def stokes(z,nu,offset):

        
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
#%%
def solve_nu_t(N):
    vec = sio.loadmat(filepath + 'vectrino_' + str(N) + '.mat')
    #hparams = np.load('/Users/marianne/Desktop/VectrinoSummer/hydroparams/sdfix4_'+str(N)+'.npy',allow_pickle = True).item()
    
    #vertical average over the profile
    u = np.nanmean(vec['velmaj'], axis = 0)
    
    z = vec['z']
    zidx = vec['z'] > 0.005
    zpos = vec['z'][zidx]
    
    zidx2 = vec['z'] > 0
    
    #saving wave turb results for each burst
    waveturb[N] = {}
    
    #spectrum to find wave peak
    fu,pu = sig.welch(u,fs = 64, window = 'hamming', nperseg = len(u)//10,detrend = 'linear') 
    fmax = fu[np.nanargmax(pu)]
    fs = 64
    
    #filtering using the phase method
    ufilt = wave_vel_decomp(u,fs,'u',plot = False)
    
    #calculate analytic signal based on de-meaned and filtered velocity (just waves)
    hu = sig.hilbert(ufilt - np.nanmean(ufilt)) 
    
    #Phase based on analytic signal
    p = np.arctan2(hu.imag,hu.real) 
    
    #Setting frequency to correspond to wave peak
    om = 2*np.pi*fmax
    
    #Calculating fluctuating velocity 
    n,m = np.shape(vec['velmaj'])
    up = np.zeros((n,m))
    vp = np.zeros((n,m))
    w1p = np.zeros((n,m))
    ubar = np.nanmean(vec['velmaj'],axis = 1)
    vbar = np.nanmean(vec['velmin'],axis = 1)
    w1bar = np.nanmean(vec['velz1'],axis = 1)
    for ii in range(m):
        up[:,ii] = vec['velmaj'][:,ii] - ubar
        vp[:,ii] = vec['velmin'][:,ii] - vbar
        w1p[:,ii] = vec['velz1'][:,ii] - w1bar
    
    dphi = np.pi/4 #Discretizing the phase
    phasebins = np.arange(-np.pi,np.pi,dphi)
    #constants
    u0 = ubvec[N]
    
    uprof = np.zeros((len(zpos),len(phasebins))) #Measured wave velocity profile
    
    uprof2 = np.zeros((len(z[zidx2]),len(phasebins)))
    
    for ii in range(len(phasebins)):
           
        if ii == 0:
            #For -pi
            idx1 = ((p >= phasebins[-1] + (dphi/2)) | (p <= phasebins[0] + (dphi/2))) #Measured
        else:
            #For phases in the middle
            idx1 = ((p >= phasebins[ii]-(dphi/2)) & (p <= phasebins[ii]+(dphi/2))) #measured
        
        uphase = up[:,idx1]
        vphase = vp[:,idx1]
        w1phase = w1p[:,idx1]
        
        # uphase = uphase[~np.isnan(np.mean(uphase,axis=1))]
        # vphase = vphase[~np.isnan(np.mean(vphase,axis=1))]
        # w1phase = w1phase[~np.isnan(np.mean(w1phase,axis=1))]
        
        '''
        big question here: get_wave_turb implements the phase method to just
        look at the wave signal, and even calculates the phase. should I
        remove that step of the wave_turb process? does it have an impact?
        I left it as-is to mess with as little as possible, but I think
        I should maybe remove all the processing steps because it is repetitive
        and I don't know how the phase method would treat an already
        decomposed signal.
        '''
        waveturb[N][ii] = get_wave_turb(uphase,vphase,w1phase,z)
        
        uproftemp = np.nanmean(up[:,idx1],axis = 1)  #Averaging over the indices for this phase bin for measurement
        waveturb[N][ii]['dudz'] = np.gradient(uproftemp[zidx2.flatten(order='F')])/np.gradient(z[zidx2.flatten(order='F')].flatten(order='F'))
        #average \tilde{u} profile for that phase
        uprof[:,ii] =  uproftemp[zidx[:,0]]
        uprof2[:,ii] = uproftemp[zidx2[:,0]]
        phase_obs[N,ii,0:len(uprof2[:,ii])] = uprof2[:,ii]/u0
    
    #non phase-decomposed -- not relevant now
    #waveturb2[N]=get_wave_turb2(up,vp,w1p,z)
    

 #%%   
    # solving for nu
    # all of the phases at the same time
    umeas = uprof.flatten(order = 'F')    
    ##
    ##same nu at all heights
    ##
    
    try:
       popt, pcov = scipy.optimize.curve_fit(make_stokes(phasebins,omega,u0,False),zpos,umeas, 
                                              p0 = (1e-5, hparams['hc']), bounds = ([1e-6,1e-4],[1e-1,6e-3]))
       nu_t= popt[0]
       offset_fit = popt[1]
    except:
        nu_t = np.nan
        offset_fit = np.nan
      
    #######################    
    #error quantification #
    #######################
    for ii in range(uprof.shape[1]):
        try:
            actual=uprof[:,ii]
            predicted1 = make_stokes(phasebins,omega,u0,True)(zpos,nu_t,offset_fit)[:,ii]
            rms[N,ii] = np.sqrt(mean_squared_error(actual, predicted))
            r2[N,ii]=r2_score(actual,predicted)
        except:
            rms[N,ii] = np.nan
            r2[N,ii] = np.nan
    
    
    ##
    ## different nu at each height
    ##
    mask = zpos>=offset_fit
    try:
        resz = scipy.optimize.least_squares(make_stokes_z(phasebins,offset_fit,omega,u0,zpos,umeas,False),
                                              x0 = (np.ones_like(zpos)*1e-5), bounds = (1e-6,1e-2))
        nu_tz= resz.x
    except:
        nu_tz = np.ones_like(zpos)*np.nan
    #######################    
    #error quantification #
    #######################
    for ii in range(uprof.shape[1]):
        try:
            actual=uprof[:,ii]
            predicted = make_stokes_z(phasebins,offset_fit,omega,u0,zpos,umeas,True)(nu_tz)[:,ii]
            rms_z[N,ii]=np.sqrt(mean_squared_error(actual, predicted))
            r2_z[N,ii]=r2_score(actual,predicted)
        except:
            rms_z[N,ii]=np.nan
            r2_z[N,ii]=np.nan
    
    delta = np.sqrt(2*1e-6/omega)
            
    ######################
    #save all the outputs#
    ######################
    for ii in range(uprof.shape[1]):
        #phase_obs[N,ii,0:len(uprof2[:,ii])] = uprof2[:,ii]/u0
        theory_nu_z[N,ii,0:len(uprof[:,ii])]= make_stokes_z(phasebins,offset_fit,omega,u0,zpos,umeas,True)(nu_tz)[:,ii]/u0
        #theory_nu[N,ii,0:len(uprof[:,ii])]= make_stokes(phasebins,omega,u0,True)(zpos,nu_t,offset_fit)[:,ii]/u0
        theory_nu[N,ii,0:len(uprof[:,ii])]= make_stokes(phasebins,omega,u0,True)(zpos,nu_t,offset_fit)[:,ii]/u0

    
    nu_fits[N]=nu_t
    offset_fits[N] = offset_fit
    nu_fits_z[N,0:len(nu_tz)]=nu_tz
     
        
    '''
    question here: should ub actually be ubdir/np.sqrt(2) (same u0 as in stokes)
    another question: should delta be based on the best-fit nu_T or the 
    kinematic viscosity? Right now I used kinematic because the best fit nu_T is 
    a pretty bad fit in stokes for a lot of bursts
    '''

#%%
    
'''
outputs are all of the observed profiles wave phase decomposed, all nu_T fits,
all nu_T(z) fits, rms and r2 metrics, and wave reynolds number
'''
for N in range(384):
    if(iswavy[N]):
        try:
            solve_nu_t(N)
        except:
            print(N)
            
#%%
'''
this function calculates the slope of the middle section of the nu_T(z)
profile for a given burst
'''
def scale_nu(N,plottrue):
    hparams = np.load('/Users/marianne/Desktop/VectrinoSummer/hydroparams/sdfix4_'+str(N)+'.npy',allow_pickle = True).item()
    if iswavy[N]:
        delta = np.sqrt(2e-6/omega[N])
        d=hparams['d']
        hc=hparams['hc']
        zdata = zs[N]
        if np.isnan(d):
            d=delta
        if np.isnan(hc):
            hc=delta
        if np.isnan(zdata[3]):
            zdata=z
        zdata = zdata[zdata>d]
        if ~(np.isnan(delta)):
            #u_fits_z is the output from the optimization code
            ndata=nu_fits_z[N,0:len(zdata)]
            if ~(np.nanmean(ndata)==0):
                ndata2=ndata[zdata>(hc+d)]
                zdata2 = zdata[zdata>(hc+d)]
                zdata2 = zdata2[1:]
                ndata2 = ndata2[1:]
                model=LinearRegression().fit(zdata2.reshape((-1,1)),ndata2.reshape((-1,1)))
                n_fit = (model.intercept_ + model.coef_ * zdata2).flatten(order='F')
                r_sq = model.score(zdata2.reshape((-1,1)), ndata2.reshape((-1,1)))
                linear_r2[N]=r_sq
                #slope
                nu_slope[N] = model.coef_
                #nondimensionalize slope by u* kappa
                nu_slope_nd[N] = model.coef_/(kappa * hparams['ustar'])
                #ubs[N]=hparams['ubdir']
                #ustars[N]=hparams['ustar']
                nu_variance[N]=np.var(ndata2)
                #Re_w[N]= hparams['ubdir']*np.sqrt(2e-6/hparams['omega'])/1e-6
                if plottrue:
                    fig,ax=plt.subplots()
                    ax.plot(ndata,zdata,'ko')
                    ax.plot(n_fit,zdata2,'r-')
                    ax.set_xlabel(r'$\nu_T(z)$')
                    ax.set_ylabel(r'cmab')
                    ax.text(ndata[5],zdata2[5],str(r_sq)+ ', '+str(nu_slope[N]))            
    
    
#%%
nu_slope=np.zeros(384)
nu_slope_nd=np.zeros(384)
nu_variance=np.zeros(384)
linear_r2=np.zeros(384)
                    

for N in range(384):
    try:
        scale_nu(N,False)
    except:
        print(N)
#%% 

#plots the profiles with best-fit for nu_T and nu_T(z)
                   
N=10


#just a plot mr fox
#solve_nu_t(N)
#plt.close('all')

for N in range(N,N+1):
    if iswavy[N]:
        hparams = np.load('/Users/marianne/Desktop/VectrinoSummer/hydroparams/sdfix4_'+str(N)+'.npy',allow_pickle = True).item()
        z = hparams['z']
        zpos = z[z>0]
        z2 = z[z>0.005]
        fig = plt.figure()
        ax1 = plt.subplot(1,2,1) 
        ax2 = plt.subplot(1,2,2) 
        omega = hparams['omega']
        offset = offset_fits[N]
        for ii in range(8):
            colorstr = 'C' + str(ii)
            delta = np.sqrt(2e-6/omega)
            ax1.plot(phase_obs[N,ii,0:len(zpos)],zpos,'-',color = colorstr,label = r'$\theta = $' + phasebins2[ii])
            predicted = theory_nu_z[N,ii,0:len(z2)]
            
            ax1.plot(predicted,z2,':',color = colorstr)
            
            ax2.plot(phase_obs[N,ii,0:len(zpos)],zpos,'-',color = colorstr,label = r'$\theta = $' + phasebins2[ii])
            predicted = theory_nu[N,ii,0:len(z2)]
            ax2.plot(predicted,z2,'--',color = colorstr)
            ax2.hlines(offset,-1,1)
            #shows the nu_T(z) profile as well
            #ax1.plot(nu_fits_z[10,0:len(zpos)]/nu_fits[N],zpos/delta,'ko')
        ax.set_xlabel(r'$\tilde{u}/u_0$')    
        ax.set_ylabel(r'$z/\Delta$')
        chartBox = ax.get_position()
        ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.8, chartBox.height])
        ax.legend(bbox_to_anchor=(1.4,0.5), loc='right')


#%%
plt.close('all')
#x = ubs
y = nu_fits/1e-6
#y = nu_slope_nd
x = Re_d
#y = nu_fits/nu

x,y = nanrm2(x,y)
x,y =remove_outliers(x,y,'pca')

#x = np.array(x[tmask2])
#y = np.array(y[tmask2])
#fig,ax = plt.subplots()
#plt.loglog(x,y,'ko')

plt.plot(x,y,'ko')



ct=10
ymean, edges, bnum = scipy.stats.binned_statistic(naninterp(x),naninterp(y),'mean',bins = ct)
ystd, e, bn  = scipy.stats.binned_statistic(naninterp(x),naninterp(y),'std',bins = ct)
mids = (edges[:-1]+edges[1:])/2

ci = np.zeros_like(mids)
for ii in range(len(ci)):
    ci[ii] = ystd[ii]/np.sqrt(np.sum(bnum == (ii+1)))

fig,ax = plt.subplots()
ax.errorbar(mids,ymean,yerr = ci,fmt = 'ko', capsize = 2)
#ax.set_yscale('log')

ax.set_xlabel(r'$Re_{\Delta}$')
#ax.set_ylabel(r'$P/\kappa u_*$')
#ax.set_xlabel(r'$u_b$')
ax.set_ylabel(r'$\nu_T/nu$')
#ax.set_ylabel(r'$P$')

#%%
#error metrics
np.nanmean(np.nanmean(rms_z,axis=1)[np.nanmean(rms_z,axis=1)>0])

np.nanmean(np.nanmean(rms_z,axis=1)[tmask2])

np.nanmean(r2[(np.nanmean(r2,axis=1)>0.3) ])


np.nanmean(np.nanmean(rms,axis=1)[np.nanmean(rms,axis=1)>0])
