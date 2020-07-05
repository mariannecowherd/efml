def displacement_thickness(uprof,z):
    """Calculates a modified displacement thickness for phase resolved boundary layer
    velocity profiles. 
    
    uprof is an n x p array where n is the number of vertical
    measurement bins, and p is the number of phases. 
    
    z is the vertical coordinate and is an n x 1 array. 
    
    """
    int_range = ((z > 0) & (z < 0.011)) #keep it within the good SNR range
    z_int = np.flipud(z[int_range]) #Flipping for the integral
    uprof_int = np.abs(np.flipud(uprof[int_range,:]))
    
    umax = np.nanmax(uprof_int, axis = 0) #Finding maximum velocity, wherever it may be
    idxmax = np.nanargmax(uprof_int, axis = 0)
    
    delta_1 = np.zeros((uprof.shape[1],))
    for i in range(len(delta_1)):
        delta_1[i] = np.trapz(1 - uprof_int[:idxmax[i],i]/umax[i],z_int[:idxmax[i]])
    
    
    return delta_1
#%%
    

blt3 = np.zeros((384,8))
blt2 = np.zeros((384,8))
for N in haswaves:
    uprof = np.transpose(phase_obs[N])
    zo = zs[N]-offset_fits[N]
    z = zs[N]
    try:
        bl_thickness3 = 3*(displacement_thickness(uprof,zo))
        bl_thickness2 = 2 * displacement_thickness(uprof,z)#Seems about right 
    except:
        print(N)
    blt3[N] = bl_thickness3
    blt2[N] = bl_thickness2
    
#%%
#plt.figure()
#plt.plot(phasebins, blt[8],'ko')

y=blt3[haswaves]
fig,ax=plt.subplots()
ystd = np.nanstd(y)
ax.errorbar(phasebins,np.nanmean(y,axis=0),yerr=ystd/np.sqrt(len(haswaves)+1),fmt='ro',capsize=2)
#ax.set_title('3')
y=blt2[haswaves]
#fig,ax=plt.subplots()
ystd = np.nanstd(y)
ax.errorbar(phasebins,np.nanmean(y,axis=0),yerr=ystd/np.sqrt(len(haswaves)+1),fmt='bo',capsize=2)
#ax.set_title('2')


#%%
for N in range(10):
    y=blt3[N]
    fig,ax=plt.subplots()
    ax.plot(phasebins,blt3[N],'ro--')
    ax.plot(phasebins,blt2[N],'bo--')
    for ii in range(len(phasebins)):
        colorstr = 'C' + str(ii)
        ax.plot(phasebins[ii]+phase_obs[N,ii],zs[N])
#%%
fig,ax = plt.subplots(2,1)
for j in range(len(phasebins)):  
#for j in range(1):
    y2 = blt2[:,j]
    y = blt3[:,j]
    #y=np.nanmean(blt2,axis=1)
    x = np.array(Re_d[haswaves])
    y = np.array(y[haswaves])
    y2 = np.array(y2[haswaves])
    mask = (x>0)&(x<160)
    x=x[mask]
    y=y[mask]
    y2=y2[mask]
    
    ct=10
    ymean, edges, bnum = scipy.stats.binned_statistic(naninterp(x),naninterp(y),'mean',bins = ct)
    ystd, e, bn  = scipy.stats.binned_statistic(naninterp(x),naninterp(y),'std',bins = ct)
    
    ymean2, edges2, bnum2 = scipy.stats.binned_statistic(naninterp(x),naninterp(y2),'mean',bins = ct)
    ystd2, e2, bn2  = scipy.stats.binned_statistic(naninterp(x),naninterp(y2),'std',bins = ct)

    mids = (edges[:-1]+edges[1:])/2
    mids2 = (edges2[:-1]+edges2[1:])/2
    
    ci = np.zeros_like(mids)
    for ii in range(len(ci)):
        ci[ii] = ystd[ii]/np.sqrt(np.sum(bnum == (ii+1)))
    
    ci2 = np.zeros_like(mids2)
    for ii in range(len(ci2)):
        ci2[ii] = ystd2[ii]/np.sqrt(np.sum(bnum2 == (ii+1)))
    
    colorstr = 'C' + str(j)
    ax[0].errorbar(mids,ymean,yerr = ci,fmt = 'o:', color = colorstr, capsize = 2)
    ax[1].errorbar(mids2,ymean2,yerr = ci2,fmt = 'o:', color = colorstr, capsize = 2)


    #ax.set_yscale('log')

    #ax[0].set_xlabel(r'$Re_{\Delta}$')
    ax[0].set_ylabel(r'$blt$')
    ax[1].set_xlabel(r'$Re_{\Delta}$')
    ax[1].set_ylabel(r'$blt$')
    ax[0].set_title(r'$3\Delta-{offset}$')
    ax[1].set_title(r'$2\Delta$')

x = np.array(Re_d[haswaves])
y2 = np.nanmean(blt2[haswaves],axis=1)
y = np.nanmean(blt3[haswaves],axis=1)
ymean, edges, bnum = scipy.stats.binned_statistic(naninterp(x),naninterp(y),'mean',bins = ct)
ystd, e, bn  = scipy.stats.binned_statistic(naninterp(x),naninterp(y),'std',bins = ct)

ymean2, edges2, bnum2 = scipy.stats.binned_statistic(naninterp(x),naninterp(y2),'mean',bins = ct)
ystd2, e2, bn2  = scipy.stats.binned_statistic(naninterp(x),naninterp(y2),'std',bins = ct)

ax[0].errorbar(mids,ymean,yerr = ci,fmt = '*-', color = 'k', capsize = 2)
ax[1].errorbar(mids2,ymean2,yerr = ci2,fmt = '*-', color = 'k', capsize = 2)
    

#%%
x = phasebins
y = np.nanmean(blt3,axis=0)
y2 = np.nanmean(blt2,axis=0)
fig,ax = plt.subplots()
ax.plot(x,y+np.nanmean(y),'bo:')
ax.plot(x,y2,'ro:')

