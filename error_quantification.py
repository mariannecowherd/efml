#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 14:37:04 2020

@author: marianne

@title: r2
"""

#%%
#just getting error quantification to work

#%%
plt.close('all')
rmss=np.zeros([384,8])
r_percent=np.zeros([384,8,20],dtype=float)
r2=np.zeros([384,8],dtype=float)
#vec = np.load('/Users/marianne/Desktop/VectrinoSummer/vectrino_8.mat',allow_pickle = True).item()
for N in range(384):
    if iswaves[N]:
        hparams = np.load('/Users/marianne/Desktop/VectrinoSummer/hydroparams/sdfix4_'+str(N)+'.npy',allow_pickle = True).item()
        vec = sio.loadmat(filepath + 'vectrino_' + str(N) + '.mat')
        try:
            #vertical average over some portion of the profile
            #u = np.nanmean(vec['velmaj'][4:12,:],axis = 0) 
            u = np.nanmean(vec['velmaj'], axis = 0)
            
            #spectrum to find wave peak
            fu,pu = sig.welch(u,fs = 64, window = 'hamming', nperseg = len(u)//10,detrend = 'linear') 
            fmax = fu[np.nanargmax(pu)]
            
            #Finding the lower and upper limits of the wave peak by eye 
            #plt.loglog(fu,pu)
            f_low = 0.13
            f_high = 0.5
            
            #N=9
            fs = 64
            #wave_vel_decomp(u,fs,'u',plot = True)

            #Band pass filtering incl. part under the peak
            ufilt = wave_vel_decomp(u,fs,'u',plot = False)
            
            #calculate analytic signal based on de-meaned and low pass filtered velocity
            hu = sig.hilbert(ufilt - np.nanmean(ufilt)) 
            
            #Phase based on analytic signal
            p = np.arctan2(hu.imag,hu.real) 
            
            
            #Setting frequency to correspond to wave peak
            omega = 2*np.pi*fmax
            
            #Constants
            #offset = .0012 #Accounts for worm canopy
            offset = hparams['hc']
            if(np.isnan(offset)):
                offset=0.0012
            
            #Bottom wave-orbital velocity
            #u0 = 0.079
            u0 = hparams['ubdir']/np.sqrt(2)
            #Binning the measured data
            
            #Calculating fluctuating velocity 
            n,m = np.shape(vec['velmaj'])
            up = np.zeros((n,m))
            ubar = np.nanmean(vec['velmaj'],axis = 1)
            for ii in range(m):
                up[:,ii] = vec['velmaj'][:,ii] - ubar
            
            dphi = np.pi/4 #Discretizing the phase
            phasebins = np.arange(-np.pi,np.pi,dphi)
              
            zidx = vec['z'] > 0
            zpos = vec['z'][zidx]
            
            uprof = np.zeros((len(zpos),len(phasebins))) #Measured wave velocity profile

            for ii in range(len(phasebins)):
                   
                if ii == 0:
                    #For -pi
                    idx1 = ( (p >= phasebins[-1] + (dphi/2)) | (p <= phasebins[0] + (dphi/2))) #Measured
                else:
                    #For phases in the middle
                    idx1 = ((p >= phasebins[ii]-(dphi/2)) & (p <= phasebins[ii]+(dphi/2))) #measured
                
                
                uproftemp = np.nanmean(up[:,idx1],axis = 1)  #Averaging over the indices for this phase bin for measurement
                uprof[:,ii] =  uproftemp[zidx[:,0]]
             
        
            
            # Setting up optimization function
            umeas = uprof.flatten(order = 'F')
            
            def make_stokes(phasebins,offset,omega,u0,plot):
            #ef make_stokes(phasebins, omega, u0, plot):
                #def stokes(z,nu,offset):
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
            
            # solving for nu
            
            fmask = np.abs(umeas)/u0 >0.01
            fmask = zpos > hparams['hc']
            fmask8 = np.concatenate((fmask, fmask, fmask, fmask, fmask, fmask, fmask, fmask))
                
            popt, pcov = scipy.optimize.curve_fit((make_stokes(phasebins,offset,omega,u0,False)),(zpos[fmask]),naninterp(umeas[fmask8]), p0 = 1e-4,maxfev=2000)
    
            sigma_actual=np.std(naninterp(umeas[fmask8]))
            sigma_predicted = np.sqrt(np.diag(pcov))
            nu_t = popt[0]
            rms=[]
            for ii in range(uprof.shape[1]):
                actual=uprof[:,ii]
                predicted = make_stokes(phasebins,offset,omega,u0,True)(zpos,nu_t)[:,ii]
                actual=actual[fmask]
                predicted = predicted[fmask]
                rms.append(np.sqrt(mean_squared_error(actual, predicted)))
                #r2 calc
                ss_tot=np.sum((actual-np.nanmean(actual))**2)
                residual=actual-predicted
                ss_res=np.sum(residual**2)
                #p=3
                #n=len(actual)
                r_squared = 1-(ss_res/ss_tot)#*(n-1)/(n-p-1)
                r2[N,ii]=r_squared
                for jj in range(len(actual)):
                    res_ht=actual[jj]-predicted[jj]
                    r_percent[N,ii,jj]=(1-(res_ht**2/ss_tot))/r_squared
        
        except:
            nu_t = np.nan
            fmax = np.nan
            omega = np.nan
            pcov = [np.nan]
            rms = np.nan
        rmss[N,:]=rms

#%%
plt.close('all')
plt.figure(1)
for N in range(8):
    colorstr = 'C' + str(N)
    plt.errorbar(phasebins[N],np.nanmean(rmss[tmask,N]),yerr=np.std(rmss[tmask,N]),fmt='*',color=colorstr)
    plt.errorbar(phasebins[N],np.nanmean(rms_z[tmask,N]),yerr=np.std(rms_z[tmask,N]),fmt='o',color=colorstr)
    #plt.errorbar(phasebins[N],np.nanmean(rmss[tmask,N])-np.nanmean(rms_z[tmask,N]),yerr=np.std(rmss[tmask,N]-rms_z[tmask,N]),fmt='o',color='k')

plt.show()
#%%
np.nanmean(rmss[tmask])
np.nanmean(rms_z[tmask])
#pi/s
(np.nanmean(rmss[tmask],axis=0)[2]+np.nanmean(rmss[tmask],axis=0)[6])/2
(np.nanmean(rmss[tmask],axis=0)[0]+np.nanmean(rmss[tmask],axis=0)[4])/2

(np.nanmean(rms_z[tmask],axis=0)[2]+np.nanmean(rms_z[tmask],axis=0)[6])/2
(np.nanmean(rms_z[tmask],axis=0)[0]+np.nanmean(rms_z[tmask],axis=0)[4])/2

np.nanmean(np.nanmean(rmss[tmask],axis=0)-np.nanmean(rms_z[tmask],axis=0))/np.nanmean(rmss[tmask])


#%%

plt.close('all')
plt.plot(phasebins2,np.nanmean(rmss[tmask],axis=0),'ko')
plt.plot(phasebins2,np.nanmean(rms_z[tmask],axis=0),'go')

np.nanmean(np.nanmean(rms_z[tmask],axis=0)-np.nanmean(rmss[tmask],axis=0))