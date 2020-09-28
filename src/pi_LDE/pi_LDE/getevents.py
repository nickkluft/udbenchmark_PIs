#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@ package pi_SpatTemp
@ file getevents.py
@ author Nick Kluft
@ brief retrieves gait events from gaitEvents yaml file

@usage:
events = read_events("subject_##_cond_##_run_##_gaitEvents.yaml")

Copyright (C) 2020 Vrije Universiteit Amsterdam (FGB-Human Movement Sciences)
Distributed under the pache 2.0 license.
"""

def read_events(fname):
    l_toe_off = None
    r_toe_off = None
    l_heel_strike = None
    r_heel_strike = None
    f = open(fname,'r')
    txt = f.read()
    txt.replace(" ","")
    for ieve in range(4):
        ibeg = txt.find("[")
        iend = txt.find("]")
        if txt[ibeg-10:ibeg-1] in ["l_toe_off"]:
            l_toe_off = eval(txt[ibeg:iend+1])
        if txt[ibeg-10:ibeg-1] in ["r_toe_off"]:
            r_toe_off = eval(txt[ibeg:iend+1])
        if txt[ibeg-14:ibeg-1] in ["l_heel_strike"]:
            l_heel_strike = eval(txt[ibeg:iend+1])
        if txt[ibeg-14:ibeg-1] in ["r_heel_strike"]:
            r_heel_strike = eval(txt[ibeg:iend+1])
        txt = txt[iend+1:]
    class strct():
        pass
    events = strct()
    events.l_toe_off = l_toe_off
    events.r_toe_off = r_toe_off
    events.l_heel_strike = l_heel_strike
    events.r_heel_strike = r_heel_strike
    return events




