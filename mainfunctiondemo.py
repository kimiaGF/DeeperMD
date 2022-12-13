#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:51:59 2022

@author: michaelmacisaac
"""
import hyperparam_train_test
import post_training_handling
import train

params={"model descriptor rcut":[4.0,5.0,6.0,7.0],"model descriptor rcut_smth":[1.0,1.5,2.0,2.5],"model descriptor axis_neuron":[4,8,12,16]}
hyperparam_train_test.hyperparam_optimize(directory='/blue/subhash/michaelmacisaac/functions/deepmd/model1',\
     base_json='/blue/subhash/michaelmacisaac/functions/deepmd/model1/base.json',\
     param_dict=params,n=30,test_model='graph-compress.pb',\
     d1_dir='1d_gridsearch',frozen_model='graph.pb',compression=True,\
     compressed_model='graph-compress.pb',\
     path_to_cval='/blue/subhash/michaelmacisaac/functions/deepmd/model1/cval'\
         ,gen_cval_data=True,crossval=True,training_path=None,\
         validation_path=None,test_path=None,multisystem=True,lammps=True,\
         lammps_model='graph-compress.pb',lammps_data='/blue/subhash/michaelmacisaac/functions/deepmd/model1/data.b4c_cell',\
            lammps_script='/blue/subhash/michaelmacisaac/functions/deepmd/model1/in.lattice_constants',\
            ref_len=5.65,ref_coh=-7.2183)
