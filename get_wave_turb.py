#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:36:57 2020

@author: marianne

@title: get_wave_turb
"""
def get_wave_turb(u,v,w1,z):
    waveturb = dict()
    m,n = np.shape(u)
    fs=64
    
    #Turbulent reynolds stresses
    waveturb['uw1'] = np.empty((m,))*np.NaN
    waveturb['vw1'] = np.empty((m,))*np.NaN
    waveturb['uv'] = np.empty((m,))*np.NaN
    waveturb['uu'] = np.empty((m,))*np.NaN
    waveturb['vv'] = np.empty((m,))*np.NaN
    waveturb['w1w1'] = np.empty((m,))*np.NaN
    waveturb['z'] = z

    #Wave reynolds stresses
    waveturb['uw1_wave'] = np.empty((m,))*np.NaN
    waveturb['vw1_wave'] = np.empty((m,))*np.NaN
    waveturb['uv_wave'] = np.empty((m,))*np.NaN
    waveturb['uu_wave'] = np.empty((m,))*np.NaN
    waveturb['vv_wave'] = np.empty((m,))*np.NaN
    waveturb['w1w1_wave'] = np.empty((m,))*np.NaN


    for jj in range(z.size):
        if np.sum(np.isnan(u[jj,:])) < np.size(u[jj,:]/2):
            nfft = u[jj,:].size
            Amu = np.fft.fft(naninterp(u[jj,:]))/np.sqrt(nfft)
            Amv = np.fft.fft(naninterp(v[jj,:]))/np.sqrt(nfft)
            Amw1 = np.fft.fft(naninterp(w1[jj,:]))/np.sqrt(nfft)

            df = fs/(nfft-1)
            nnyq = int(np.floor(nfft/2 +1))
            fm = np.arange(0,nnyq)*df

            #Phase
            Uph = np.arctan2(np.imag(Amu),np.real(Amu)).squeeze()[:nnyq]
            Vph = np.arctan2(np.imag(Amv),np.real(Amv)).squeeze()[:nnyq]
            W1ph = np.arctan2(np.imag(Amw1),np.real(Amw1)).squeeze()[:nnyq]

            #Computing the full spectra
            Suu = np.real(Amu*np.conj(Amu))/(nnyq*df)
            Suu = Suu.squeeze()[:nnyq]

            Svv = np.real(Amv*np.conj(Amv))/(nnyq*df)
            Svv = Svv.squeeze()[:nnyq]

            Sww1 = np.real(Amw1*np.conj(Amw1))/(nnyq*df)
            Sww1 = Sww1.squeeze()[:nnyq]

            Suv = np.real(Amu*np.conj(Amv))/(nnyq*df)
            Suv = Suv.squeeze()[:nnyq]

            Suw1 = np.real(Amu*np.conj(Amw1))/(nnyq*df)
            Suw1 = Suw1.squeeze()[:nnyq]


            Svw1 = np.real(Amv*np.conj(Amw1))/(nnyq*df)
            Svw1 = Svw1.squeeze()[:nnyq]

            offset = np.sum(fm<=0.1)

            uumax = np.argmax(Suu[(fm>0.1) & (fm < 0.7)]) + offset

            #width ratio around the peak
            widthratiolow = 6
            widthratiohigh = 6

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

            #Wave spectra
            Suu_wave = Suu[waverange] - Puuhat[waverange]
            Svv_wave = Svv[waverange] - Pvvhat[waverange]
            Sww1_wave = Sww1[waverange] - Pww1hat[waverange]

            #This should maybe be nnyq*df? But then the amplitudes are way too big
            Amu_wave = np.sqrt((Suu_wave+0j)*(df))
            Amv_wave = np.sqrt((Svv_wave+0j))*(df)
            Amww1_wave = np.sqrt((Sww1_wave+0j)*(df))

            #Wave Magnitudes
            Um_wave = np.sqrt(np.real(Amu_wave)**2 + np.imag(Amu_wave)**2)
            Vm_wave = np.sqrt(np.real(Amv_wave)**2 + np.imag(Amv_wave)**2)
            W1m_wave = np.sqrt(np.real(Amww1_wave)**2 + np.imag(Amww1_wave)**2)

            #Wave reynolds stress
            uw1_wave = np.nansum(Um_wave*W1m_wave*np.cos(W1ph[waverange]-Uph[waverange]))
            uv_wave =  np.nansum(Um_wave*Vm_wave*np.cos(Vph[waverange]-Uph[waverange]))
            vw1_wave = np.nansum(Vm_wave*W1m_wave*np.cos(W1ph[waverange]-Vph[waverange]))

            uu_wave = np.nansum(Suu_wave*df)
            vv_wave = np.nansum(Svv_wave*df)
            w1w1_wave = np.nansum(Sww1_wave*df)

            #Full reynolds stresses
            uu = np.nansum(Suu*df)
            uv = np.nansum(Suv*df)
            uw1 = np.nansum(Suw1*df)
            vv = np.nansum(Svv*df)
            vw1 = np.nansum(Svw1*df)
            w1w1 = np.nansum(Sww1*df)
            #Turbulent reynolds stresses

            upup = uu - uu_wave
            vpvp = vv - vv_wave
            w1pw1p = w1w1 - w1w1_wave
            upw1p = uw1 - uw1_wave
            upvp = uv - uv_wave
            vpw1p = vw1 - vw1_wave

            #Turbulent reynolds stresses
            waveturb['uw1'][jj] = upw1p
            waveturb['vw1'][jj] = vpw1p
            waveturb['uv'][jj] = upvp
            waveturb['uu'][jj] = upup
            waveturb['vv'][jj] = vpvp
            waveturb['w1w1'][jj] = w1pw1p


            #Wave reynolds stresses
            waveturb['uw1_wave'][jj] = uw1_wave
            waveturb['vw1_wave'][jj] = vw1_wave
            waveturb['uv_wave'][jj] = uv_wave
            waveturb['uu_wave'][jj] = uu_wave
            waveturb['vv_wave'][jj] = vv_wave
            waveturb['w1w1_wave'][jj] = w1w1_wave
    return waveturb
