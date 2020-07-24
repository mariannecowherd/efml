#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:07:10 2020

@author: marianne

@title: turbulence figures
"""
#%%
from scipy import interpolate
#%%
#phasebins3=phasebins+np.pi
# jloc = [4,5,6,7,0,1,2,3]
# #controu plot of turbulence component for a single burst
# #for N in haswaves[0:10]:
# for N in [7,8,9,10,129]:
#     uwmesh = np.zeros((30,8))
#     for j in range(8):
#         for k in range(13):
#             l = jloc[j]
#             uwmesh[k,l] = (waveturb[N][j]['uw1_wave'][k])
#     z = waveturb[N][1]['z']        
#     z_mesh, y_mesh = np.meshgrid(z,phasebins)
#     levuw1 = np.linspace(np.nanmin(uwmesh),np.nanmax(uwmesh),20);
#     plt.figure(figsize=(15,10))
#     cf = plt.contourf(y_mesh, z_mesh, uwmesh.T, levuw1, extend='both')
#     plt.colorbar(cf)
#     plt.xlabel('wave phase', fontsize=16)
#     plt.ylabel('uw1', fontsize=16)
#     plt.title('turb ' + str(N), fontsize=18)
#     plt.ylim(-0.001,0.016)
#     plt.tight_layout()
#     plt.show()
#%%
doesntwork=[]
works=[]

for N in waveturb.keys():
#for N in [1,4,10]:
        for i in range(8):
            try:
                wave_nd_old = waveturb[N][i]['uw1_wave']/ubvec[N]/ubvec[N]
                nd_old = waveturb[N][i]['uw1']/ustarwc_gm[N]/ustarwc_gm[N]
                delt = np.sqrt(2*nu_fits[N]/omega[N])
                delt = np.sqrt(2e-6/omega[N])
                dudz_old = waveturb[N][i]['dudz']/(ubvec[N]/delt)
                vel_old = phase_obs[N,i,:]/(ubvec[N])
                #
                wave_nd_old = wave_nd_old[np.where(~np.isnan(wave_nd_old))]
                nd_old = nd_old[np.where(~np.isnan(wave_nd_old))]
                #
                t_old = waveturb2[N]['uw1']/(ustarwc_gm[N]**2)
                tw_old = waveturb2[N]['uw1_wave']/(ubvec[N]/ubvec[N])
                #
                t_old = t_old[np.where(~np.isnan(tw_old))]
                tw_old = tw_old[np.where(~np.isnan(tw_old))]
                #
                zold = waveturb[N][i]['z']
                zold = zold[np.where(~np.isnan(wave_nd_old))].flatten(order='F')
                znew = np.linspace(0.001, 0.015, 15)
                #
                f_wave = interpolate.interp1d(zold, naninterp(wave_nd_old), kind='cubic')
                f = interpolate.interp1d(zold, naninterp(nd_old), kind='cubic')
                f_d = interpolate.interp1d(zold,naninterp(dudz_old),kind='cubic')
                #
                f_t = interpolate.interp1d(zold, naninterp(t_old), kind='cubic')
                f_tw = interpolate.interp1d(zold, naninterp(tw_old), kind='cubic')
                #
                waveturb[N][i]['uw_wave_nd'] = f_wave(znew)
                waveturb[N][i]['uw_nd']= f(znew)
                waveturb[N][i]['dudz_nd'] = f_d(znew)
                waveturb[N]['znew'] = znew

                #
                waveturb2[N]['t_nd'] = f_t(znew)
                waveturb2[N]['tw_nd'] = f_tw(znew)
                works.append(N)
            except:
                try:
                    temp = np.array([0.15])
                    zold = np.concatenate((zold,temp))
                    wave_nd_old = np.concatenate((wave_nd_old,temp2))
                    nd_old = np.concatenate((nd_old,temp2))
                    dudz_old = np.concatenate((dudz_old,temp2))
                    #
                    t_old = np.concatenate((t_old,temp2))
                    tw_old = np.concatenate((tw_old,temp2))
                    #
                    f_wave = interpolate.interp1d(zold, naninterp(wave_nd_old), kind='cubic')
                    f = interpolate.interp1d(zold, naninterp(nd_old), kind='cubic')
                    f_d = interpolate.interp1d(zold,naninterp(dudz_old),kind='cubic')
                    #
                    f_t = interpolate.interp1d(zold,naninterp(t_old),kind='cubic')
                    f_tw = interpolate.interp1d(zold,naninterp(tw_old),kind='cubic')
                    #
                    waveturb[N][i]['uw_wave_nd'] = f_wave(znew)
                    waveturb[N][i]['uw_nd']= f(znew)
                    waveturb[N][i]['dudz_nd'] = f_d(znew)
                    waveturb[N]['znew'] = znew
                    waveturb2[N]['t_nd'] = f_t(znew)
                    waveturb2[N]['tw_nd'] = f_tw(znew)
                    works.append(N)
                except:
                    doesntwork.append(N)
        
#%%
uw_nd_ens = np.zeros([8,15])
uw_wave_nd_ens = np.zeros([8,15])
dudz_ens=np.zeros([8,15])
#el_ens = np.zeros([8,15])
for i in range(8):
    temp = []
    temp_wave = []
    temp_dudz=[]
    temp_vel = []
    for N in np.unique(works):
        temp.append(waveturb[N][i]['uw_nd'])
        temp_wave.append(waveturb[N][i]['uw_wave_nd'])
        temp_dudz.append(waveturb[N][i]['dudz_nd'])
        #temp_vel.append(waveturb[N][i]['velnew'])
                
    uw_nd_ens[i] = np.nanmean(temp,axis=0)
    uw_wave_nd_ens[i] = np.nanmean(temp_wave,axis=0)
    dudz_ens[i] = np.nanmean(temp_dudz,axis=0)
    #vel_ens[i] = np.nanmean(temp_vel,axis=0)

temp = []
temp_wave = []
for N in np.unique(works):
    temp.append(waveturb2[N]['t_nd'])
    temp_wave.append(waveturb2[N]['tw_nd'])
uw2=np.nanmean(temp,axis=0)
uw2_wave=np.nanmean(temp_wave,axis=0)
#%%
#contour plot of wave turbulence component for all bursts
uwmesh = np.zeros((15,8))
for j in range(8):
    for k in range(15):            
        l = jloc[j]
        uwmesh[k,l] = -uw_wave_nd_ens[j][k]
z_mesh, y_mesh = np.meshgrid(znew,phasebins)
levuw1 = np.linspace(np.nanmin(uwmesh),np.nanmax(uwmesh),20);
plt.figure(figsize=(15,10))
cf = plt.contourf(y_mesh, z_mesh, uwmesh.T, levuw1, extend='both')
plt.colorbar(cf, label=r'$-\overline{\tilde{u}\tilde{w}}/ u_b^2$')

plt.ylim(np.nanmin(znew),np.nanmax(znew))

plt.plot(phasebins,usbins[4,:],'o:',color="white")
plt.xticks(ticks = phasebins, labels = phaselabels)


plt.xlabel('wave phase', fontsize=16)
plt.ylabel('z', fontsize=16)
#plt.title(r'$\overline{\tilde{u}\tilde{w}}/ u_b^2$', fontsize=18)

plt.tight_layout()
plt.show()

plt.savefig('waveturb.pdf')

#%% 
# reg turb for all bursts, averaged
uwmesh = np.zeros((15,8))
for j in range(8):
    for k in range(15):
        l = jloc[j]
        uwmesh[k,l] = uw_nd_ens[j][k]
z_mesh, y_mesh = np.meshgrid(znew,phasebins)
levuw1 = np.linspace(np.nanmin(uwmesh),np.nanmax(uwmesh),20);
plt.figure(figsize=(15,10))
cf = plt.contourf(y_mesh, z_mesh, uwmesh.T, levuw1, extend='both')
plt.colorbar(cf,label=r"$\overline{u'w'}/u_*^2$")

plt.plot(phasebins,usbins[4,:],'o:',color="white")
plt.xticks(ticks = phasebins, labels = phaselabels)


plt.xlabel('wave phase', fontsize=16)
plt.ylabel('z', fontsize=16)
#plt.title(r"$\overline{u'w'}/u_*^2$", fontsize=18)
plt.ylim(-0.001,0.016)
plt.tight_layout()
plt.show()

#%%
uwmesh = np.zeros((15,8))
for j in range(8):
    for k in range(15):
        l = jloc[j]
        uwmesh[k,l] = dudz_ens[j][k]
z_mesh, y_mesh = np.meshgrid(znew,phasebins)
levuw1 = np.linspace(np.nanmin(uwmesh),np.nanmax(uwmesh),20);
plt.figure(figsize=(15,10))
cf = plt.contourf(y_mesh, z_mesh, uwmesh.T, levuw1, extend='both')
plt.colorbar(cf)

plt.plot(phasebins,usbins[5,:],'o:',color="white")

plt.xticks(ticks = phasebins, labels = phaselabels)
plt.xlabel('wave phase', fontsize=16)
plt.ylabel('z', fontsize=16)
plt.title(r'$du/dz / \Delta u_b$', fontsize=18)
plt.ylim(-0.001,0.016)
plt.tight_layout()
plt.errorbar(phasebins,np.nanmean(y,axis=0),yerr=ystd/np.sqrt(len(haswaves)+1),fmt='ko:',capsize=2)
plt.show()
#%%
fig,ax=plt.subplots()
for j in range(8):
    colorstr = 'C' + str(j)
    plt.plot(dudz_ens[j,:],znew,color=colorstr,label = r'$\theta = $' + phasebins2[j])

    #plt.plot(uw_wave_nd_ens[j,:],znew,color=colorstr,label = r'$\theta = $' + phasebins2[j])
    
#plt.plot(uw2_wave,znew,'k-')
#plt.plot(np.nansum(uw_wave_nd_ens,axis=0),znew,'k:')
plt.ylabel('z')
#plt.xlabel(r'$\overline{\tilde{u}\tilde{w}}/u_b^2$')   
#plt.xlabel(r"$\overline{u'w'}/u_*^2$")   
plt.xlabel(r"$du/dz$")   
chartBox = ax.get_position()
ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.8, chartBox.height])
ax.legend(bbox_to_anchor=(1.4,0.5), loc='right')
#%%
vel_interp = np.zeros([383,15,8])
velidx=[]
for N in waveturb.keys():
    for i in range(8):
        try:
            vel_old = phase_obs[N,i,:]/ubvec[N]
            zold = waveturb[N][0]['z']
            zold = zold.flatten()
            znew = waveturb[N]['znew']
            f_vel = interpolate.interp1d(zold,naninterp(vel_old),kind='cubic')
            vel_interp[N,:,i] = (f_vel(znew))
            velidx.append(N)
            
        except:
            print(N)
            
            
            #%%
velidx=np.unique(velidx)

vel_ens = np.nanmean(vel_interp[velidx,:,:],axis=0)


fig,ax = plt.subplots()
for i in range(8):
    ax.plot(vel_ens[:,i],znew)
    
plt.savefig('vel_ens.pdf')
