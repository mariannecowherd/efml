#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 14:39:46 2020

@author: marianne

@title: vel_ens
"""

import numpy as np
import scipy.signal as sig
from scipy import interpolate
import vectrinofuncs as vfs
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from vectrinofuncs import naninterp

from stokesfunctions import make_stokes

params = {
   'axes.labelsize': 28,
   'font.size': 28,
   'legend.fontsize': 26,
   'xtick.labelsize': 28,
   'ytick.labelsize': 28,
   'text.usetex': True,
   'font.family': 'serif',
   'axes.grid' : False,
   'image.cmap': 'plasma'
   }

plt.rcParams.update(params)
plt.close('all')


#data
blparams = np.load('blparams.npy',allow_pickle=True).item()
profiles = np.load('phaseprofiles.npy',allow_pickle=True)
stress = np.load('phase_stress.npy',allow_pickle=True).item()
delta = blparams['delta']
phasebins = blparams['phasebins']
ustarwc_gm = blparams['ustarwc_gm']
omega = blparams['omega']
ubvec = blparams['ubvec']/np.sqrt(2)
zs = stress['z']
u0s = stress['freestream']


burstnums = list(range(384))
'''
construct stokes solution given omega, phi, and u0
omega is a range around the average wave peak for the entire deployment
u0 is the average bottom wave orbital velocity
phi is a random phase offset
'''

dphi = np.pi/4 #Discretizing the phase
phasebins = np.arange(-np.pi,np.pi,dphi)
nu = 1e-6 #kinematic viscosity

ub_bar = np.nanmean(ubvec)
u0 = np.nanmean(np.abs(u0s))
omega_bar = np.nanmean(omega)
omega_std = np.nanstd(omega)
z = np.linspace(0.00, 0.015, 100) #height vector
t = np.linspace(-np.pi/omega_bar,np.pi/omega_bar,100) #time vector
nm = 1000 #how many values of omega
oms = np.linspace(omega_bar-omega_std,omega_bar+omega_std,nm) #omega vector

omstokes = np.zeros((len(z),len(phasebins),nm)) #initialized output

for k in range(len(oms)):
    om = oms[k]
    uwave = np.zeros((len(z),len(t))) #temporary array for given frequency
    phi = np.random.rand() * 2*np.pi #random value for phi

    #stokes solution
    for jj in range(len(z)):
        uwave[jj,:] = (np.cos(om*t-phi) -
                    np.exp(-np.sqrt(om/(2*nu))*z[jj])*np.cos(
                        (om*t-phi) - np.sqrt(om/(2*nu))*z[jj]))
    huwave = sig.hilbert(np.nanmean(uwave,axis = 0))  #hilbert transform
    pw = np.arctan2(huwave.imag,huwave.real) #phase

    ustokes = np.zeros((len(z),len(phasebins)))

    #allocate into phase bins
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

#average bursts for the same phase bin for all values of omega
for k in range(len(z)):
    for i in range(len(phasebins)):
        omsum[k,i] = np.nanmean(omstokes[k,i,:])

'''
ensemble average measured velocity profiles at each phase bin
interpolate onto standard z scale
normalize by bottom wave orbital velocity
'''
vel_interp = np.zeros([384,15,8]) #initialize output
velidx=[]
znew = np.linspace(0.001, 0.015, 15)

#ensemble average, normalize by ubvec
for n in burstnums:
    for i in range(8):
        try:
            vel_old = profiles[:,n,i]/ubvec[n]
            zold = stress['z'][:,n]
            zold = zold.flatten()
            zold,vel_old = vfs.nanrm2(zold,vel_old)
            f_vel = interpolate.interp1d(zold,vfs.naninterp(vel_old),kind='cubic')
            vel_interp[n,:,i] = (f_vel(znew))
            velidx.append(n)
        except ValueError:
            continue

velidx=np.unique(velidx)

vel_ens = np.nanmean(vel_interp[velidx,:,:],axis=0)
u1 = np.nanmax(abs(vel_ens))
phasebins2 = ['$-\\pi$', '$-3\\pi/4$', '$-\\pi/2$','$-\\pi/4$', '$0$',
              '$\\pi/4$', '$\\pi/2$', '$3\\pi/4$']

delta_plot = 2*vfs.displacement_thickness_interp(vel_ens,znew)
delta_theory = 2*vfs.displacement_thickness_interp(u1*omsum,z+0.001)
phaselabels = [r'$-\frac{3\pi}{4}$',r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$',
               r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',
               r'$\pi$']

fig,ax = plt.subplots(1,2)
obs_blt = []
model_blt = []
obs_y=[]
model_y=[]

for i in range(8):
    z2 = z+0.001
    colorstr = 'C' + str(i)
    ax[0].plot((u1*omsum[:,i]),100*z2,':',color = colorstr)
    ax[0].plot(vel_ens[:,i],znew*100,label = phasebins2[i])

    #Spline fit to velocity profiles to add BL thickness
    tck = interpolate.splrep(znew,vel_ens[:,i], s = 0)
    zinterp = np.linspace(0.001,0.015,200)
    velinterp = interpolate.splev(zinterp,tck,der = 0)

    blidx = np.argmin(np.abs(delta_plot[i] - zinterp))
    ax[0].plot(velinterp[blidx],zinterp[blidx]*100,'o', color = colorstr)

    theory = u1*omsum[:,i]
    #blt_theory = vfs.displacement_thickness_interp(theory,z2)
    #blidxm = np.nanargmax(np.abs(u1*omsum[:,i]))
    blidxm = np.nanargmin(np.abs(delta_theory[i]-z))

    obs_blt.append(zinterp[blidx])
    obs_y.append(velinterp[blidx])
    model_blt.append(z[blidxm])
    model_y.append(theory[blidxm])
    ax[0].plot(theory[blidxm],100*(z2[blidxm]),'o', fillstyle='none', color = colorstr)



observed = ax[0].plot([300, 300], color = 'black', linestyle='-', label='observation')
model = ax[0].plot([300,300],color='black',linestyle = ':', label='laminar')
fit = ax[0].plot([300,300],color='black',linestyle = '--',label = 'fit')
handles, labels = ax[0].get_legend_handles_labels()

ax[0].set_ylim(0,1.5)
l1=ax[1].legend(handles[0:11], labels[0:11],ncol=1,frameon=False,loc='center left',fontsize=18,bbox_to_anchor=(1, 0.5))
#l2=ax[1].legend(handles[8:10],labels[8:10],ncol=1,frameon=False,loc='upper right',fontsize=12,bbox_to_anchor=(1.05, 1))
ax[1].add_artist(l1)


ax[0].set_ylabel(r'$z$ (cmab)')
ax[0].set_xlabel(r'$\frac{\tilde{u}}{u_b}$')
#plt.savefig('plots/vel_ens.pdf')
#fit stokes function to the whole-burst velocity profiles
offset = 0.0025
idx = znew>offset
z = znew[idx]

actual = vel_ens[idx,:]/np.nanmax(np.abs(vel_ens[idx,:]))
popt, pcov = scipy.optimize.curve_fit((make_stokes(phasebins,
            omega_bar,1,offset,False)),(z),actual.flatten(order='F'),p0 = 1e-4,maxfev=2000)
nu_t=popt[0]
r2 = []
residual = []
#fig,ax = plt.subplots()
predictions = np.zeros((8,13))
for i in range(8):
    colorstr = 'C' + str(i)
    real = actual[:,i]
    predicted = make_stokes(phasebins,omega_bar,1,offset,True)(z,nu_t)[:,i]
    ax[1].plot(vel_ens[:,i],znew, color = colorstr)
    ax[1].plot(predicted*np.nanmax(np.abs(vel_ens[idx,:])),z,'--',color=colorstr)
    r2.append(r2_score(real,predicted))
    residual.append(mean_squared_error(real,predicted))
    predictions[i]=predicted

ax[1].set_xlabel(r'$\frac{\tilde{u}}{u_b}$')
#ax.set_ylabel(r'$z$ (cmab)')
ax[1].set_ylim(0,1.5/100)
ax[1].set_yticks([],[])


ax[0].set_title('(a)')
ax[1].set_title('(b)')

#%%

np.nanmean(blparams['ustarwc_meas'])*np.nanmean(blparams['delta'][0,:])

np.nanmean(blparams['ustarwc_meas'])**2 /omega_bar

#dblt = np.array(obs_blt)-np.array(model_blt)

np.nanmean(np.abs(np.array(residual)/np.array(np.sum(actual,axis=0)) * 100))
np.nanstd(np.abs(np.array(residual)/np.array(np.sum(actual,axis=0)) * 100))


np.nanmean(((obs_blt)-np.array(model_blt))/np.array(obs_blt)) *100
np.nanstd(((obs_blt)-np.array(model_blt))/np.array(obs_blt)) *100

#%%
#spline fit blts for observations and model, separately
model_y.append(model_y[0])
model_blt.append(model_blt[0])
obs_y.append(obs_y[0])
obs_blt.append(obs_blt[0])

#separate positive and negative phases becasuse spline requires monotonic input
my1 = np.array(model_y)[0:5]
my2 = np.array(model_y)[4:9]
mb1 = np.array(model_blt)[0:5]
mb2 = np.array(model_blt)[4:9]

#Spline fit
tck = interpolate.splrep(my1,mb1, s = 0)
zinterp = np.linspace(np.nanmin(my1),np.nanmax(my1),50)
mb_interp = interpolate.splev(zinterp,tck,der = 0)

ax[0].plot(zinterp,(mb_interp+0.001)*100,linestyle='dashdot',color='k',linewidth=0.8)

tck = interpolate.splrep(np.flipud(my2),np.flipud(mb2), s = 0)
zinterp = np.linspace(np.nanmin(my2),np.nanmax(my2),50)
mb_interp = interpolate.splev(zinterp,tck,der = 0)

ax[0].plot(zinterp,(mb_interp+0.001)*100,linestyle='dashdot',color='k',linewidth=0.8)

oy1 = np.array(obs_y)[0:5]
oy2 = np.array(obs_y)[4:9]
ob1 = np.array(obs_blt)[0:5]
ob2 = np.array(obs_blt)[4:9]

#Spline fit
tck = interpolate.splrep(oy1,ob1, s = 0)
zinterp = np.linspace(np.nanmin(oy1),np.nanmax(oy1),50)
ob_interp = interpolate.splev(zinterp,tck,der = 0)

ax[0].plot(zinterp,(ob_interp)*100,linestyle='dashdot',color='k',linewidth=0.8)

tck = interpolate.splrep(np.flipud(oy2),np.flipud(ob2), s = 0)
zinterp = np.linspace(np.nanmin(oy2),np.nanmax(oy2),50)
ob_interp = interpolate.splev(zinterp,tck,der = 0)

ax[0].plot(zinterp,(ob_interp)*100,linestyle='dashdot',color='k',linewidth=0.8)
