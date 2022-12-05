#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 19:29:45 2022

@author: michaelmacisaac
"""

import hyperparam_optimization
import hyperparam_train_test
import post_training_handling
import train
params={"model descriptor rcut":[4.0,5.0,6.0,7.0],"model descriptor rcut_smth":[1.0,1.5,2.0,2.5],"model descriptor axis_neuron":[4,8,12,16],"model fitting_net neuron":[[10,10],[50,50],[50,50,50],[10,10,10],[100,100],[100,100,100]]}
hyperparam_optimization.json_dir_gen_1d(\
        base_json='base.json', param_dict=params,\
        training_paths='/home/kimia.gh/blue2/python_course/test_03/virials/training_data',\
        validation_paths='/home/kimia.gh/blue2/python_course/test_03/virials/validation_data',\
            crossval=False,d1_dir='1d_gridsearch')
post_training_handling.lammps_lat_const_modifier(model='graph-compress.pb',\
    data='/blue/subhash/michaelmacisaac/functions/deepmd/model1/data.b4c_cell',lammps_script='in.lattice_constants')
hyperparam_train_test.hyperparam_train_test(directory='/blue/subhash/michaelmacisaac/functions/deepmd/model1',d1_dir='1d_gridsearch',\
            lammps_script='/blue/subhash/michaelmacisaac/functions/deepmd/model1/in.lattice_constants', ref_len=5.65,\
                ref_coh=-7.2183,test_path='/home/kimia.gh/blue2/python_course/test_03/virials/validation_data'\
                    ,n=40,multisystem=True,compression=True,crossval=False)
