#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@ package pi_FPE
@ file marginsofstability.py
@ author Nick Kluft
@ brief calculates the margins of stability in ap and ml direction. 
    Spits out the relavant PIs in yaml format.

@usage:
events = calcMOS(joint,com,events,trlinfo)

Copyright (C) 2020 Vrije Universiteit Amsterdam (FGB-Human Movement Sciences)
Distributed under the pache 2.0 license.
"""
import numpy as np

def calcMOS(joint,com,events,trlinfo):
    print("Calculating Margins of Stability...")
    # get time vector
    t = joint.time
    # get sampling frequency
    fs = 1/np.nanmean(np.diff(t))
    # set the gravitational acceleration
    g = 9.81 #m/s
    # event timing to index number
    lhs = np.array(events.l_heel_strike)*fs
    rhs = np.array(events.r_heel_strike)*fs
    lto = np.array(events.l_toe_off)*fs
    rto = np.array(events.r_toe_off)*fs
    # get toe markers
    ltoe = np.transpose(np.vstack((joint.l_TTII_x,joint.l_TTII_y,joint.l_TTII_z)))
    rtoe = np.transpose(np.vstack((joint.r_TTII_x,joint.r_TTII_y,joint.r_TTII_z)))
    # get heel markers
    lheel = np.transpose(np.vstack((joint.l_heel_x,joint.l_heel_y,joint.l_heel_z)))
    rheel = np.transpose(np.vstack((joint.r_heel_x,joint.r_heel_y,joint.r_heel_z)))
    # get hips
    rhip = np.transpose(np.vstack((joint.l_hip_x,joint.l_hip_y,joint.l_hip_z)))
    lhip = np.transpose(np.vstack((joint.r_hip_x,joint.r_hip_y,joint.r_hip_z)))
    # determine walking directions
    walkdir = np.argmax(np.ptp(np.array(lheel),axis=0))  
    mldir = np.argmax(np.nanmean(np.abs(rhip-lhip),axis=0))
    # get com array [change COMx -> x]
    com_arr = np.array(np.vstack([com.x,com.y,com.z])).transpose()
    # determine pendulum length
    ll_all = com_arr[lhs.astype(int),:]-lheel[lhs.astype(int),:]
    rl_all = com_arr[rhs.astype(int),:]-rheel[rhs.astype(int),:]
    # get length of pendulum
    ll = np.nanmean(np.sqrt(np.sum(ll_all**2,axis=1)))
    rl = np.nanmean(np.sqrt(np.sum(rl_all**2,axis=1)))
    # vcom
    veloarr = [0,0,0]
    veloarr[walkdir] = trlinfo.tm_speed
    vcom = (np.gradient(com_arr,axis=0)*fs)+veloarr
    # calculate extrapolated CoM
    xcom = vcom/np.sqrt(g/((ll+rl)/2))
    # get bos at lto
    lto_mos = np.abs(rtoe[lto.astype(int),:]-com_arr[lto.astype(int),:])-np.abs(xcom[lto.astype(int),:])
    rto_mos = np.abs(ltoe[rto.astype(int),:]-com_arr[rto.astype(int),:])-np.abs(xcom[rto.astype(int),:])
    # now get the margins of stability in the desired directions
    l_mos_ap = rto_mos[:,walkdir]
    l_mos_ml = rto_mos[:,mldir]
    r_mos_ap = lto_mos[:,walkdir]
    r_mos_ml = lto_mos[:,mldir]
    return l_mos_ap,l_mos_ml,r_mos_ap,r_mos_ml,walkdir,mldir

