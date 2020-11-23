#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@ package pi_FPE
@ file getevents.py
@ author Nick Kluft
@ brief retrieves gait events from gaitEvents yaml file

@usage:
events = read_events("subject_##_cond_##_run_##_gaitEvents.yaml")

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
    lto_txt = yaml_eve['l_toe_off']
    rto_txt = yaml_eve['r_toe_off']
    lhs_txt = yaml_eve['l_heel_strike']
    rhs_txt = yaml_eve['r_heel_strike']
    l_toe_off = eval("["+lto_txt.replace(" ",",")+"]")
    r_toe_off = eval("["+rto_txt.replace(" ",",")+"]")
    l_heel_strike = eval("[" + lhs_txt.replace(" ",",")+"]")
    r_heel_strike = eval("[" + rhs_txt.replace(" ",",")+"]")
    class strct():
        pass
    events = strct()
    events.l_toe_off = l_toe_off
    events.r_toe_off = r_toe_off
    events.l_heel_strike = l_heel_strike
    events.r_heel_strike = r_heel_strike
    return events