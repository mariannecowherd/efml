#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:53:00 2020

@author: marianne, gegan
"""

import numpy as np
import scipy.signal as sig
import vectrinofuncs as vfs
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy import interpolate
import scipy.special as sc
import warnings

warnings.filterwarnings("ignore")

params = {
    'axes.labelsize': 28,
    'font.size': 28,
    'legend.fontsize': 26,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'text.usetex': True,
    'font.family': 'serif',
    'axes.grid': False,
    'image.cmap': 'plasma'
}

plt.rcParams.update(params)
plt.close('all')

# data
profiles = np.load('data/phaseprofiles.npy')
stress = np.load('data/phase_stress.npy', allow_pickle=True).item()
blparams = np.load('data/blparams.npy', allow_pickle=True).item()
omega = blparams['omega']
delta = blparams['delta']
ustarwc_gm = blparams['ustarwc_gm']
ubvec = np.load('ub.npy', allow_pickle=True)
zs = stress['z']
u0s = stress['freestream']

delta_e = 0.15  # cm
h = 0.375  # cm

burstnums = list(range(384))
'''
construct stokes solution given omega, phi, and u0
omega is a range around the average wave peak for the entire deployment
u0 is the average bottom wave orbital velocity
phi is a random phase offset
'''

dphi = np.pi / 4  # Discretizing the phase
phasebins = np.arange(-np.pi, np.pi, dphi)
nu = 1e-6  # kinematic viscosity

ub_bar = np.nanmean(ubvec)
u0 = np.nanmean(np.abs(u0s))
omega_bar = np.nanmean(omega)
omega_std = np.nanstd(omega)
z = np.linspace(0.00, 0.015, 100)  # height vector
t = np.linspace(-np.pi / omega_bar, np.pi / omega_bar, 100)  # time vector
nm = 1000  # how many values of omega
oms = np.linspace(omega_bar - omega_std, omega_bar + omega_std, nm)  # omega vector

omstokes = np.zeros((len(z), len(phasebins), nm))  # initialized output

for k in range(len(oms)):
    om = oms[k]
    uwave = np.zeros((len(z), len(t)))  # temporary array for given frequency
    phi = np.random.rand() * 2 * np.pi  # random value for phi

    # stokes solution
    for jj in range(len(z)):
        uwave[jj, :] = (np.cos(om * t - phi) -
                        np.exp(-np.sqrt(om / (2 * nu)) * z[jj]) * np.cos(
                    (om * t - phi) - np.sqrt(om / (2 * nu)) * z[jj]))
    huwave = sig.hilbert(np.nanmean(uwave, axis=0))  # hilbert transform
    pw = np.arctan2(huwave.imag, huwave.real)  # phase

    ustokes = np.zeros((len(z), len(phasebins)))

    # allocate into phase bins
    for ii in range(len(phasebins)):

        if ii == 0:
            # For -pi
            idx2 = ((pw >= phasebins[-1] + (dphi / 2)) | (pw <= phasebins[0] + (dphi / 2)))  # analytical
        else:
            # For phases in the middle
            idx2 = ((pw >= phasebins[ii] - (dphi / 2)) & (pw <= phasebins[ii] + (dphi / 2)))  # analytical

        ustokes[:, ii] = np.nanmean(uwave[:, idx2], axis=1)  # Averaging over the phase bin

        omstokes[:, ii, k] = ustokes[:, ii]

omsum = np.zeros((len(z), len(phasebins)))

# average bursts for the same phase bin for all values of omega
for k in range(len(z)):
    for i in range(len(phasebins)):
        omsum[k, i] = np.nanmean(omstokes[k, i, :])

'''
ensemble average measured velocity profiles at each phase bin
interpolate onto standard z scale
normalize by bottom wave orbital velocity
'''
vel_interp = np.zeros([384, 15, 8])  # initialize output
velidx = []
znew = np.linspace(0.001, 0.015, 15)

# ensemble average, normalize by ubvec
for n in burstnums:
    for i in range(8):
        try:
            vel_old = profiles[:, n, i] / ubvec[n]
            zold = zs[:, n]
            # zold = zold.flatten()
            zold, vel_old = vfs.nanrm2(zold, vel_old)
            f_vel = interpolate.interp1d(zold, vfs.naninterp(vel_old), kind='cubic')
            vel_interp[n, :, i] = (f_vel(znew))
            velidx.append(n)
        except ValueError:
            continue

velidx = np.unique(velidx)

# vel_ens = np.nanmean(vel_interp[velidx,:,:],axis=0)

vel_ens = np.nanmean(vel_interp[velidx, :, :], axis=0) / np.sqrt(2)

u1 = np.nanmax(abs(vel_ens))
phasebins2 = ['$-\\pi$', '$-3\\pi/4$', '$-\\pi/2$', '$-\\pi/4$', '$0$',
              '$\\pi/4$', '$\\pi/2$', '$3\\pi/4$']

delta_plot = 2 * vfs.displacement_thickness_interp(vel_ens, znew)
delta_theory = 2 * vfs.displacement_thickness_interp(u1 * omsum, z + 0.001)
phaselabels = [r'$-\frac{3\pi}{4}$', r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$',
               r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
               r'$\pi$']

fig, ax = plt.subplots(1, 2, figsize=(24, 16)) #some systems better with 16,24
obs_blt = []
model_blt = []
obs_y = []
model_y = []

# plot figure 1a and save boundary layer thicknesses
for i in range(8):
    z2 = z + 0.001
    colorstr = 'C' + str(i)
    ax[0].plot((u1 * omsum[:, i]), 100 * z2, ':', color=colorstr)
    ax[0].plot(vel_ens[:, i], znew[:] * 100, '-', color=colorstr, label=phasebins2[i])

    # Spline fit to velocity profiles to add BL thickness
    tck = interpolate.splrep(znew, vel_ens[:, i], s=0)
    zinterp = np.linspace(0.001, 0.015, 200)
    velinterp = interpolate.splev(zinterp, tck, der=0)

    blidx = np.argmin(np.abs(delta_plot[i] - zinterp))
    ax[0].plot(velinterp[blidx], zinterp[blidx] * 100, 'o', color=colorstr)

    theory = u1 * omsum[:, i]
    blidxm = np.nanargmin(np.abs(delta_theory[i] - z))

    obs_blt.append(zinterp[blidx])
    obs_y.append(velinterp[blidx])
    model_blt.append(z[blidxm])
    model_y.append(theory[blidxm])
    ax[0].plot(theory[blidxm], 100 * (z2[blidxm]), 'o', fillstyle='none', color=colorstr)

"""
nu fit int
"""
us_fit = np.array([0.03861628, 0.0290057, 0.01163489, 0.05124935, 0.04170248,
                   0.03013098, 0.01302687, 0.05235832])
nu_fit = 0.41 * us_fit * (np.nanmean(np.linspace(0.0035, .0125, 10)))

# defining gm function
def make_gm_offset(omega, kb, u0, offset,l_theta):
    def gm(z, ustar):
        kappa = 0.41
        c1 = 0.14
        l = kappa * ustar / omega
        #l = kappa * delta / c1
        #l = np.nanmean(nu_fit) / (z * omega)
        #l = l_theta
        zeta = (z - offset) / l
        zeta0 = kb / (30 * l)

        uw = u0 * (1 - ((sc.ker(2 * np.sqrt(zeta)) + 1j * sc.kei(2 * np.sqrt(zeta))) /
                        (sc.ker(2 * np.sqrt(zeta0)) + 1j * sc.kei(2 * np.sqrt(zeta0)))))

        return uw.real

    return gm


# fitting gm function
kb = 0.01
omega = np.nanmean(omega)
tvec = np.arange(-np.pi, np.pi, np.pi / 4) / omega

uinf = 0.86 * np.exp(1j * tvec * omega)
offset = 0.0025

ustar = np.zeros((8,))
u0 = np.zeros((8,))
r2 = np.zeros((8,))

for i in range(8):
    if i>0:
        l_theta = np.nanmean(delta[i-1, :])
    else:
        l_theta = np.nanmean(delta[7, :])
    popt, pcov = curve_fit(make_gm_offset(omega, kb, uinf[i], offset, l_theta), znew[3:-3], vel_ens[3:-3, i],
                           p0=1e-2, bounds=(1e-4, 1e-1))

    ustar[i] = popt[0]

# plotting figure 1b and calculating error
r2_gm = []
residual_gm = []
diffsum = []
diffsum2 = []
vel_blt = []
temp = []
for i in range(8):
    ax[1].plot(vel_ens[:, i], znew[:] * 100, '-', color='C' + str(i))
    if i > 0:
        l_theta = np.nanmean(delta[i - 1, :])
    else:
        l_theta = np.nanmean(delta[7, :])
    zint = np.linspace(0.001, 0.015, 100)
    ax[1].plot(make_gm_offset(omega, kb, uinf[i], offset,l_theta)(zint, ustar[i])[14:], zint[14:] * 100, '--',
               color='C' + str(i))

    real = vel_ens[-13:, i]
    z_r = znew[-13:]
    predicted = make_gm_offset(omega, kb, uinf[i], offset,l_theta)(z_r, ustar[i])
    r2_gm.append(r2_score(real, predicted))
    residual_gm.append(mean_squared_error(real, predicted))
    diffsum.append(np.nansum(np.abs(real - predicted)))
    # error just above the boundary layer
    diffsum2.append(np.nansum(np.abs(real[z_r > obs_blt[i]] - predicted[z_r > obs_blt[i]])))
    vel_blt.append(np.nansum(np.abs(vel_ens[znew > obs_blt[i], i])))

    blt_gm = 2 * 0.41 * ustar[i] / omega
    idx = np.nanargmin(np.abs(blt_gm - z_r))
    temp.append(blt_gm)
    ax[1].plot(predicted[idx], z_r[idx] * 100, 'o', fillstyle='none')

plt.figure(2)
plt.plot(phasebins, temp, '*--')
plt.plot(phasebins, obs_blt, 'o:')

rmspe = np.nansum(diffsum) / np.nansum(np.abs(vel_ens[-13:, :]))
rmspe_blt = np.nansum(diffsum2) / np.nansum(vel_blt)

# adding boundary layer lines
# spline fit blts for observations and model, separately
model_y.append(model_y[0])
model_blt.append(model_blt[0])
obs_y.append(obs_y[0])
obs_blt.append(obs_blt[0])

# separate positive and negative phases because spline requires monotonic input
my1 = np.array(model_y)[0:5]
my2 = np.array(model_y)[4:9]
mb1 = np.array(model_blt)[0:5]
mb2 = np.array(model_blt)[4:9]

# Spline fit
tck = interpolate.splrep(my1, mb1, s=0)
zinterp = np.linspace(np.nanmin(my1), np.nanmax(my1), 50)
mb_interp = interpolate.splev(zinterp, tck, der=0)

ax[0].plot(zinterp, (mb_interp + 0.001) * 100, linestyle='dashdot', color='k', linewidth=0.8)

tck = interpolate.splrep(np.flipud(my2), np.flipud(mb2), s=0)
zinterp = np.linspace(np.nanmin(my2), np.nanmax(my2), 50)
mb_interp = interpolate.splev(zinterp, tck, der=0)

ax[0].plot(zinterp, (mb_interp + 0.001) * 100, linestyle='dashdot', color='k', linewidth=0.8)

oy1 = np.array(obs_y)[0:5]
oy2 = np.array(obs_y)[4:9]
ob1 = np.array(obs_blt)[0:5]
ob2 = np.array(obs_blt)[4:9]

# Spline fit
tck = interpolate.splrep(oy1, ob1, s=0)
zinterp = np.linspace(np.nanmin(oy1), np.nanmax(oy1), 50)
ob_interp = interpolate.splev(zinterp, tck, der=0)

ax[0].plot(zinterp, ob_interp * 100, linestyle='dashdot', color='k', linewidth=0.8)

tck = interpolate.splrep(np.flipud(oy2), np.flipud(ob2), s=0)
zinterp = np.linspace(np.nanmin(oy2), np.nanmax(oy2), 50)
ob_interp = interpolate.splev(zinterp, tck, der=0)

ax[0].plot(zinterp, ob_interp * 100, linestyle='dashdot', color='k', linewidth=0.8)

# figure details
# labels
observed = ax[0].plot([300, 300], color='black', linestyle='-', label='observation')
model = ax[0].plot([300, 300], color='black', linestyle=':', label='laminar')
fit = ax[0].plot([300, 300], color='black', linestyle='--', label='GM fit')
blt = ax[0].plot([300, 300], color='black', linestyle='dashdot', label='boundary layer')
handles, labels = ax[0].get_legend_handles_labels()

ax[0].set_ylim(0, 1.5)
ax[1].set_ylim(0, 1.5)

ax[0].set_xlim(-1, 1)
ax[1].set_xlim(-1, 1)

l1 = ax[1].legend(handles[0:12], labels[0:12], ncol=1, frameon=False, loc='center left', fontsize=18,
                  bbox_to_anchor=(1, 0.5))
ax[1].add_artist(l1)

ax[0].set_ylabel(r'$z$ (cmab)')
ax[0].set_xlabel(r'$u(z, \theta)u_b^{-1}$')
ax[1].set_xlabel(r'$u(z, \theta)u_b^{-1}$')

ax[0].set_title(r'(a)')
ax[1].set_title(r'(b)')

fig.subplots_adjust(top=0.91, bottom=0.14, left=0.095, right=0.805, hspace=0.2, wspace=0.2)

fig.show()
fig.savefig('plots/vel_ens_gm.pdf')
