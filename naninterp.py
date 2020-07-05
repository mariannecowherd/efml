#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:09:44 2020

@author: gegan

@title: naninterp
"""

import datetime
import numpy as np
import scipy.interpolate


def naninterp(x):
    
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