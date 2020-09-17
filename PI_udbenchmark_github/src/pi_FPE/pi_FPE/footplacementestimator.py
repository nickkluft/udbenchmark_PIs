#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:42:06 2020
@author: Nick1
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import yaml
from scipy import optimize
from termcolor import colored
from .getevents import read_events
from .colfun import prod_col,norm_col,transpose_col

def calcFPE(joint,com,angMom,comIR,mass,trlinfo):
# =============================================================================
#     Calculate Foot Placement Estimator
# =============================================================================
    # get time vector
    t = joint.time/1000
    # get sampling frequency
    fs = 1/np.nanmean(np.diff(t))
    # determine the gravitational acceleration
    g = 9.81 #m/s
    # get CoM position in m
    com = com/1000;
    # create empty omega
    omega = np.ones([t.shape[0],3])*np.nan
    # get angular momentum matrix
    H = np.vstack([angMom.H_x,angMom.H_y,angMom.H_z]).transpose()
    # rearrange IR and calculate omega   
    for i in range(comIR.shape[0]):
        # get inertial tensor and reshape 
        IR = np.vstack([np.hstack([comIR.IRxx[i],comIR.IRxy[i],comIR.IRxz[i]]),
                        np.hstack([comIR.IRyx[i],comIR.IRyy[i],comIR.IRyz[i]]),
                        np.hstack([comIR.IRzx[i],comIR.IRzy[i],comIR.IRzz[i]])])
        # multiply the angular momentum with the inverse of the inertial tensor
        omega[i,:] = np.matmul(np.transpose(H[i,:]),np.linalg.inv(IR))

    l_foot = pd.concat([(joint.l_TTII_x+joint.l_heel_x)/2,(joint.l_TTII_y+joint.l_heel_y)/2,(joint.l_TTII_z+joint.l_heel_z)/2],axis=1)    
    walkdir = np.argmax(np.ptp(np.array(l_foot),axis=0))

    # Bring the data into the direction of the angular momentum
    com_arr = np.array(np.vstack([com.COMx,com.COMy,com.COMz])).transpose()
    com_proj = np.array(np.vstack([com.COMx,com.COMy,com.COMz*0])).transpose()
    # calculate the velocity of the CoM and add the current treadmill speed [m/s]
    veloarr = [0,0,0]
    veloarr[walkdir] = trlinfo.tm_speed
    vcom = (np.gradient(com_arr,axis=0)*fs)+veloarr
    H_y_ax = H+np.cross((com_arr-com_proj)*mass,vcom)
    yaxis = np.vstack([H_y_ax[:,0],H_y_ax[:,1],np.zeros([1,com.shape[0]])]).transpose()
    zaxis = np.divide((com_arr-com_proj),com_arr)
    xaxis = np.cross(yaxis,zaxis)
    yaxis = yaxis/norm_col(yaxis)
    zaxis = zaxis/norm_col(zaxis)    
    xaxis = xaxis/norm_col(xaxis)
    R_proj = np.hstack([xaxis,yaxis,zaxis])
    omega_proj = prod_col(transpose_col(R_proj),omega)
    vcom_proj = prod_col(transpose_col(R_proj),vcom)
    com_proj = prod_col(transpose_col(R_proj),com_arr)

    anglearr = np.array([0,0,0])
    # treadmill angle expressed in the direction of angular momentum
    if not(trlinfo.condition.find('roll')): #roll
        anglearr[walkdir]=trlinfo.tm_angle
        angTM=np.tile(np.radians(anglearr),[vcom.shape[0],1])
    else:# pitch
        nonwalkdir = np.array([1,0])
        anglearr[nonwalkdir[walkdir]]
        angTM=np.tile(np.radians(anglearr),[vcom.shape[0],1])
    theta_proj = prod_col(transpose_col(R_proj),angTM)
    
    # create empty matrices
    lFPE = np.zeros([vcom.shape[0],1])*np.nan
    phi =  np.zeros([vcom.shape[0],1])*np.nan
    phi[0]=np.radians(3)
    # fmsval =  np.zeros([vcom.shape[0],1])*np.nan
    Jcom =  np.zeros([vcom.shape[0],1])*np.nan

    print('Optimisation of foot placement estimator...')
    # loop over all frames
    for i in range(vcom.shape[0]):
        # Get inertial tensor and reshape
        IR = np.vstack([np.hstack([comIR.IRxx[i],comIR.IRxy[i],comIR.IRxz[i]]),
                np.hstack([comIR.IRyx[i],comIR.IRyy[i],comIR.IRyz[i]]),
                np.hstack([comIR.IRzx[i],comIR.IRzy[i],comIR.IRzz[i]])])
        # calculate the moment of inertia in the new direction
        Jcom[i] = np.dot(np.dot(yaxis[i,:],IR),yaxis[i,:])
        if vcom_proj[i,0] == vcom_proj[i,0]:
            phi[i] = optimize.fmin(fFPE,phi[i-1],args = (Jcom[i,0],com_proj[i,2],g,mass,omega_proj[i,1],theta_proj[i,1],vcom_proj[i,0],vcom_proj[i,1]),disp=0)
            # phi[i] = optimize.brentq(fFPE,0,3,args = (Jcom[i,0],com_proj[i,2],g,mass,omega_proj[i,1],theta_proj[i,1],vcom_proj[i,0],vcom_proj[i,1]),xtol=1e-6,rtol=1e-7)             
            lFPE[i] = ((np.cos(theta_proj[i,1])*(com_arr[i,1]))/(np.cos(phi[i])))*np.sin(theta_proj[i,1]+phi[i])
    return lFPE
        
def fFPE(phi,Jcom,hcom,g,m,omega,theta,vx,vy):
# =============================================================================
#     Foot placement estimator formula to optimze
# =============================================================================
    t2 = np.cos(phi)
    t3 = np.cos(theta)
    t4 = theta+phi
    t5 = t3**2
    t7 = 1/t2
    t8 = t7**2
    t9 = hcom**2
    fpefun = (Jcom/2+(m*t5*t8*t9)/2)*((Jcom*omega+m*t3*t7*hcom*(vx*np.cos(t4)+vy*np.sin(t4)))**2)*1/((Jcom+m*t5*t8*t9)**2)-g*m*t3*t7*hcom+g*m*t3*t7*hcom*np.cos(t4)
    return fpefun**2
    
def store_result(file_out, value):
    file = open(file_out, 'w')
    file.write('type: \'vector\'\nvalues: ')
    for line in value:
        file.write('\n'+format(line[0], '.5f'))
    file.close()
    return True

USAGE = """usage: run_pi file_in_joint file_in_com file_in_angmom file_in_comIR file_in_events file_in_trialinfo folder_out
file_in_joint: csv file containing the  positonal joint data
file_in_events: csv file containing the  timing of gait events
folder_out: folder where the PI yaml files will be stored
"""

def main():
#    joint,com,angMom,comIR,events,trialinfo
    file_in_joint = sys.argv[1]
    file_in_com = sys.argv[2]
    file_in_angmom = sys.argv[3]
    file_in_comIR = sys.argv[4]
    file_in_events = sys.argv[5]
    file_in_trialinfo = sys.argv[6]
    folder_out = sys.argv[7]    
    
    if len(sys.argv) != 8:
        print(colored("Wrong input parameters !", "red"))
        print(colored(USAGE, "yellow"))
        return -1
    
    # check input parameters are good
    if not os.path.exists(file_in_joint):
        print(colored("Input file {} does not exist".format(file_in_joint), "red"))
        return -1
    if not os.path.isfile(file_in_joint):
        print(colored("Input path {} is not a file".format(file_in_joint), "red"))
        return -1
    if not os.path.exists(file_in_events):
        print(colored("Input file {} does not exist".format(file_in_events), "red"))
        return -1
    if not os.path.isfile(file_in_events):
        print(colored("Input path {} is not a file".format(file_in_events), "red"))
        return -1
    if not os.path.exists(file_in_com):
        print(colored("Input file {} does not exist".format(file_in_com), "red"))
        return -1
    if not os.path.isfile(file_in_com):
        print(colored("Input path {} is not a file".format(file_in_com), "red"))
        return -1
    if not os.path.exists(file_in_comIR):
        print(colored("Input file {} does not exist".format(file_in_comIR), "red"))
        return -1
    if not os.path.isfile(file_in_comIR):
        print(colored("Input path {} is not a file".format(file_in_comIR), "red"))
        return -1
    if not os.path.exists(file_in_angmom):
        print(colored("Input file {} does not exist".format(file_in_angmom), "red"))
        return -1
    if not os.path.isfile(file_in_angmom):
        print(colored("Input path {} is not a file".format(file_in_angmom), "red"))
        return -1
    if not os.path.exists(file_in_trialinfo):
        print(colored("Input file {} does not exist".format(file_in_trialinfo), "red"))
        return -1
    if not os.path.isfile(file_in_trialinfo):
        print(colored("Input path {} is not a file".format(file_in_trialinfo), "red"))
        return -1
    

    if not os.path.exists(folder_out):
        print(colored(
            "Output folder {} does not exist".format(folder_out),
            "red"))
        return -1

    if not os.path.isfile(file_in_joint):
        print(colored("Output path {} is not a folder".format(file_in_joint), "red"))
        return -1
    if not os.path.isfile(file_in_events):
        print(colored("Output path {} is not a folder".format(file_in_events), "red"))
        return -1
    if not os.path.isfile(file_in_com):
        print(colored("Output path {} is not a folder".format(file_in_com), "red"))
        return -1
    if not os.path.isfile(file_in_comIR):
        print(colored("Output path {} is not a folder".format(file_in_comIR), "red"))
        return -1
    if not os.path.isfile(file_in_angmom):
        print(colored("Output path {} is not a folder".format(file_in_angmom), "red"))
        return -1
    if not os.path.isfile(file_in_trialinfo):
        print(colored("Output path {} is not a folder".format(file_in_trialinfo), "red"))
        return -1
    
    # load events structure
    events = read_events(file_in_events)
    # load joint data
    joint = pd.read_csv(file_in_joint)
    # load CoM data
    com = pd.read_csv(file_in_com)
    # load CoM angular momentum data
    angmom = pd.read_csv(file_in_angmom)
    # load CoM inertia tensor
    comIR = pd.read_csv(file_in_comIR)
    # load trail information file
    with open(file_in_trialinfo) as fileinfo:
    	trialinfo = yaml.full_load(fileinfo)
    # from joint data calculate the sampling frequency
    fs = 1000/np.mean(np.diff(np.array(com.time)))

    class strct():
        pass
    rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", trialinfo[1:-2])
    trlinfo = strct()
    if not(trialinfo.find('roll')):
        trlinfo.condition = 'pitch'
    else:
        trlinfo.condition = 'roll'
    trlinfo.tm_angle = int(rr[1])
    trlinfo.tm_speed = float(rr[3])
    print('Reading condition...')
    print('--> '+trlinfo.condition + ' ' + format(rr[1])+' deg')
    print('--> speed '+ format(rr[3])+' m/s')
    mass = 81
    #  Estimate the foot placement using the foot placement estimator 
    fpe = calcFPE(joint,com,angmom,comIR,mass,trlinfo)
    tos = np.hstack([events.l_heel_strike,events.r_heel_strike])
    itos = np.round(np.sort(tos)*fs)
    
    
    file_out0 = folder_out + "/pi_fpe.yaml"
    if not store_result(file_out0, fpe[itos.astype(int)]):
        return -1
    print(colored(
        "Foot Placement Estimator: vector with size {}x1, stored in {}".format(int(tos.shape[0]),file_out0),
        "green"))
    return 0
    