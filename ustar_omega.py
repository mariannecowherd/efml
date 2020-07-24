#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:31:02 2020

@author: marianne
"""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#%%

blparams = np.load('/Users/Marianne/Desktop/VectrinoSummer/blparams.npy',allow_pickle=True).item()
delta = blparams['delta']
phasebins = blparams['phasebins']
ustarwc_sg17 = blparams['ustarwc_sg17']
ustarwc_gm = blparams['ustarwc_gm']
ustar_adv5 = blparams['ustar_adv5']
ustar_logfit = blparams[ 'ustar_logfit']
ustarwc_meas = blparams['ustarwc_meas']
ustarc_meas = blparams[ 'ustarc_meas']
ubadv = blparams['ubadv']
ubvec = blparams['ubvec']
omega = blparams['omega']
dirspread = blparams['dirspread']

Re_d = ubvec * np.sqrt(2e-6/omega)/1e-6

#%%
params = {
   'axes.labelsize': 14,
   'font.size': 14,
   'legend.fontsize': 12,
   'xtick.labelsize': 14,
   'ytick.labelsize': 14,
   'text.usetex': True,
   'font.family': 'serif',
   'axes.grid' : False
   }



plt.rcParams.update(params)
fig,ax = plt.subplots()
#%%
plt.close('all')
fig,ax = plt.subplots()
sources = ['gm','sg17','meas']
xs = [ustarwc_gm,ustarwc_sg17,ustarwc_meas]/omega
colors = ['lightgray','gray','black']

for i in range(3):
    y=(delta[1,:]+delta[5,:])/2
    y = y[iswavy]
    x = xs[i][iswavy]
    #x = Re_d[iswavy]
    x,y = nanrm2(x,y)
    x,y = remove_outliers(x,y,'pca')

    
    ct=7
    ymean, edges, bnum = scipy.stats.binned_statistic(naninterp(x),naninterp(y),'mean',bins = ct)
    ystd, e, bn  = scipy.stats.binned_statistic(naninterp(x),naninterp(y),'std',bins = ct)
    mids = (edges[:-1]+edges[1:])/2
    
    
    ci = np.zeros_like(mids)
    for ii in range(len(ci)):
        ci[ii] = ystd[ii]/np.sqrt(np.sum(bnum == (ii+1)))
    ax.errorbar(mids,ymean,yerr = ci,fmt = 'o', color=colors[i],capsize = 2,label=(sources[i]))
    
    model=LinearRegression().fit(x.reshape((-1,1)),y.reshape((-1,1)))
    yfit = (model.intercept_ + model.coef_ * x).flatten(order='F')
    
    print(model.coef_)
    
    ax.plot(x,yfit,':',color = colors[i], label = 'm='+str(round(model.coef_[0][0],4)))
    
    #ax.text(0.005,0.004,'slope = ' + str(round(model.coef_[0][0],5)))
    #ax.set_yscale('log')

handles, labels = ax.get_legend_handles_labels()
order = [3,4,5,0,1,2]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],ncol=2)

ax.set_ylabel(r'$\Delta$')
    #ax.set_title(str(source))
ax.set_xlabel(r'$u_*/\omega$')

fig.savefig('all_delta.png')

