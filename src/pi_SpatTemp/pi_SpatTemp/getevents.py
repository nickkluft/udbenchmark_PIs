#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@ package pi_SpatTemp
@ file getevents.py
@ author Nick Kluft
@ brief retrieves gait events from gaitEvents yaml file

@usage:
events = read_events("subject_##_gaitEvents_##")

Copyright (C) 2020 Vrije Universiteit Amsterdam (FGB-Human Movement Sciences)
Distributed under the pache 2.0 license.
"""
import yaml

def read_events(fname):
    with open(fname, 'r', encoding='utf8') as stream:
        try:
            yaml_eve = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


    l_toe_off = None
    r_toe_off = None
    l_heel_strike = None
    r_heel_strike = None
    yaml_eve = yaml_eve.replace(":","=")
    yaml_eve = yaml_eve.replace(" ","")
    yaml_eve = yaml_eve.replace("\n","")
    print("Retrieving gait events...")
    for ieve in range(4):
        ibeg = yaml_eve.find("[")
        iend = yaml_eve.find("]")
        if yaml_eve[0:9] in ["l_toe_off"]:
            l_toe_off = eval(yaml_eve[ibeg:iend+1])
        if yaml_eve[0:9] in ["r_toe_off"]:
            r_toe_off = eval(yaml_eve[ibeg:iend+1])
        if yaml_eve[0:13] in ["l_heel_strike"]:
            l_heel_strike = eval(yaml_eve[ibeg:iend+1])
        if yaml_eve[0:13] in ["r_heel_strike"]:
            r_heel_strike = eval(yaml_eve[ibeg:iend+1])
        yaml_eve = yaml_eve[iend+1:]
        class strct():
            pass
        events = strct()
        events.l_toe_off = l_toe_off
        events.r_toe_off = r_toe_off
        events.l_heel_strike = l_heel_strike
        events.r_heel_strike = r_heel_strike
    return events




