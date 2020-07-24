#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:54:06 2020

@author: marianne

@title: nanrm2
"""
import numpy as np
def nanrm2(x,y):
    y = y[~np.isnan(x)]
    x = x[~np.isnan(x)]
    
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    return(x,y)
