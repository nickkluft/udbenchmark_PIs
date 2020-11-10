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
from .marginsofstability import calcMOS
from .colfun import prod_col,norm_col,transpose_col

def calcFPE(joint,com,angMom,comIR,events,mass,trlinfo):
# =============================================================================
#     Calculate Foot Placement Estimator
# =============================================================================
    # get time vector
    t = joint.time
    # get sampling frequency
    fs = 1/np.nanmean(np.diff(t))
    # set the gravitational acceleration
    g = 9.81 #m/s
    # create empty omega
    omega = np.ones([t.shape[0],3])*np.nan
    # get angular momentum matrix
    H = np.vstack([angMom.x,angMom.y,angMom.z]).transpose()
    # rearrange IR and calculate omega   
    for i in range(comIR.shape[0]):
        # get inertial tensor and reshape 
        IR = np.vstack([np.hstack([comIR.xx[i],comIR.xy[i],comIR.xz[i]]),
                        np.hstack([comIR.yx[i],comIR.yy[i],comIR.yz[i]]),
                        np.hstack([comIR.zx[i],comIR.zy[i],comIR.zz[i]])])
        # multiply the angular momentum with the inverse of the inertial tensor
        omega[i,:] = np.matmul(np.transpose(H[i,:]),np.linalg.inv(IR))

    l_foot = pd.concat([(joint.l_TTII_x+joint.l_heel_x)/2,(joint.l_TTII_y+joint.l_heel_y)/2,(joint.l_TTII_z+joint.l_heel_z)/2],axis=1)    
    walkdir = np.argmax(np.ptp(np.array(l_foot),axis=0))

    # Bring the data into the direction of the angular momentum
    com_arr = np.array(np.vstack([com.x,com.y,com.z])).transpose()
    com_proj = np.array(np.vstack([com.x,com.y,com.z*0])).transpose()
    # calculate the velocity of the CoM and add the current treadmill speed [m/s]
    veloarr = [0,0,0]
    veloarr[walkdir] = trlinfo.tm_speed
    vcom = (np.gradient(com_arr,axis=0)*fs)+veloarr
    H_y_ax = H+np.cross((com_arr-com_proj)*mass,vcom)
    yaxis = np.vstack([H_y_ax[:,0],H_y_ax[:,1],np.zeros([1,com.shape[0]])]).transpose()
    zaxis = np.divide((com_arr-com_proj),com_arr)
    xaxis = np.cross(yaxis,zaxis)
    yaxis = np.cross(zaxis,yaxis)
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
    phi_in = np.radians(3)
    # fmsval =  np.zeros([vcom.shape[0],1])*np.nan
    Jcom =  np.zeros([vcom.shape[0],1])*np.nan
    print('Optimisation of foot placement estimator...')
    sys.stdout.write("[%s]" % ("." * 10))
    sys.stdout.flush()
    sys.stdout.write("\b" * (10+1))
    steps = 9
    # loop over all frames
    for i in range(vcom.shape[0]):
        if np.greater_equal((i/float(vcom.shape[0]))*100,steps):
                sys.stdout.write("#")
                sys.stdout.flush()
                steps =steps+10
        # Get inertial tensor and reshape
        IR = np.vstack([np.hstack([comIR.xx[i],comIR.xy[i],comIR.xz[i]]),
                np.hstack([comIR.yx[i],comIR.yy[i],comIR.yz[i]]),
                np.hstack([comIR.zx[i],comIR.zy[i],comIR.zz[i]])])
        # calculate the moment of inertia in the new direction
        Jcom[i] = np.dot(np.dot(yaxis[i,:],IR),yaxis[i,:])
        if vcom_proj[i,0] == vcom_proj[i,0]:
            phi[i] = optimize.fmin(fFPE,phi_in,args = (Jcom[i,0],com_proj[i,2],g,mass,omega_proj[i,1],theta_proj[i,1],vcom_proj[i,0],vcom_proj[i,1]),disp=0)
            # phi[i] = optimize.brentq(fFPE,0,3,args = (Jcom[i,0],com_proj[i,2],g,mass,omega_proj[i,1],theta_proj[i,1],vcom_proj[i,0],vcom_proj[i,1]),xtol=1e-6,rtol=1e-7)             
            phi_in = phi[i]
            lFPE[i] = ((np.cos(theta_proj[i,1])*(com_arr[i,1]))/(np.cos(phi[i])))*np.sin(theta_proj[i,1]+phi[i])
    sys.stdout.write("]\n") # end progress bar
    gFPE = prod_col(R_proj,np.hstack((lFPE, np.zeros([vcom.shape[0],2]))))+com_arr;
    lhs = np.array(events.l_heel_strike)*fs
    rhs = np.array(events.r_heel_strike)*fs
    lankle = np.transpose(np.vstack((joint.l_ankle_x,joint.l_ankle_y,joint.l_ankle_z)))
    rankle = np.transpose(np.vstack((joint.r_ankle_x,joint.r_ankle_y,joint.r_ankle_z)))
    lhsFPE = np.abs(lankle-com_arr)-np.abs(gFPE-com_arr)
    rhsFPE = np.abs(rankle-com_arr)-np.abs(gFPE-com_arr)
    dfpe = np.vstack((lhsFPE[lhs.astype(int),:],rhsFPE[rhs.astype(int),:]))
    return dfpe
        
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
    file.write('---\ntype: \'vector\'\nvalues:')
    for line in value:
        file.write('\n '+format(line[0], '.5f'))
    file.write('\n')
    file.close()
    return True

def store_result2(file_out, value):
    file = open(file_out, 'w')
    file.write('---\ntype: \'vector\'\nvalues:')
    for line in value:
        file.write('\n '+format(line, '.5f'))
    file.write('\n')
    file.close()
    return True

USAGE = """usage: run_pi file_in_joint file_in_com file_in_angmom file_in_comIR file_in_events file_in_trialinfo folder_out
-> file_in_joint: csv file containing the  positional joint data
-> file_in_com: csv file containing the positional com data
-> file_in_angmom: csv file containing the angular momentum relative to CoM data
-> file_in_comIR: csv file containing the inertia tensor around CoM
-> file_in_events: csv file containing the  timing of gait events
-> file_in_trialinfo: trial info yaml file
-> folder_out: folder where the PI yaml files will be stored
"""

def main():
    if len(sys.argv) != 8:
        print(colored("Wrong input parameters !", "red"))
        print(colored(USAGE, "yellow"))
        return -1

#    joint,com,angMom,comIR,events,trialinfo
    file_in_joint = sys.argv[1]
    file_in_com = sys.argv[2]
    file_in_angmom = sys.argv[3]
    file_in_comIR = sys.argv[4]
    file_in_events = sys.argv[5]
    file_in_trialinfo = sys.argv[6]
    folder_out = sys.argv[7]    
    
    
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
    # calculate margins of stability
    l_mos_ap,l_mos_ml,r_mos_ap,r_mos_ml,apdir,mldir= calcMOS(joint,com,events,trlinfo)
    #  Estimate the foot placement using the foot placement estimator 
    dfpe = calcFPE(joint,com,angmom,comIR,events,mass,trlinfo)

    file_out0 = folder_out + "/pi_FPE_ap.yaml"
    if not store_result2(file_out0,dfpe[:,apdir]):
        return -1
    print(colored(
        "Foot Placement Estimator: vector with size {}x1, stored in {}".format(int(dfpe.shape[0]),file_out0),
        "green"))
    
    file_out1 = folder_out + "/pi_FPE_ml.yaml"
    if not store_result2(file_out1,dfpe[:,mldir]):
        return -1
    print(colored(
        "Foot Placement Estimator: vector with size {}x1, stored in {}".format(int(dfpe.shape[0]),file_out1),
        "green"))
    
    file_out2 = folder_out + "/pi_l_MOS_ap.yaml"
    if not store_result2(file_out2, l_mos_ap):
        return -1
    print(colored(
        "Margins of stability [ap]: vector with size {}x1, stored in {}".format(l_mos_ap.shape[0],file_out2),
        "green"))
    
    file_out3 = folder_out + "/pi_r_MOS_ap.yaml"
    if not store_result2(file_out3, r_mos_ap):
        return -1
    print(colored(
        "Margins of stability [ap]: vector with size {}x1, stored in {}".format(r_mos_ml.shape[0],file_out3),
        "green"))
    
    file_out4 = folder_out + "/pi_l_MOS_ml.yaml"
    if not store_result2(file_out4, l_mos_ml):
        return -1
    print(colored(
        "Margins of stability [ap]: vector with size {}x1, stored in {}".format(l_mos_ml.shape[0],file_out4),
        "green"))
    
    file_out5 = folder_out + "/pi_r_MOS_ml.yaml"
    if not store_result2(file_out5, r_mos_ml):
        return -1
    print(colored(
        "Margins of stability [ap]: vector with size {}x1, stored in {}".format(r_mos_ml.shape[0],file_out5),
        "green"))
    return 0
    