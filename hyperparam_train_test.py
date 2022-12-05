#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 13:04:48 2022

@author: michaelmacisaac
"""

import hyperparam_optimization
import post_training_handling
import train
import glob as glob
import os 
import numpy as np

def hyperparam_train_test(directory,d1_dir,lammps_script,ref_len,ref_coh,\
                          compression=False,crossval=True,\
                              frozen_model='graph.pb',\
                                  compressed_model='graph_compress.pb'):
    """
    This function trains models using input json files that were created
    in order to perform a 1d hyperparameter optimization/evaluation study.
    There are options for model compression and cross-validation. This 
    function will evaluate trained models ability to predict lattice constants
    and cohesive energies. Model predictions and errors/residuals will be 
    reported in a text file.

    Parameters
    ----------
    directory : ste
        Parent directory, where script will be run.
    d1_dir : str
        directory name for 1d gridsearch.
    lammps_script : .in file
        LAMMPS script to be exexuted in each model directory.
    ref_len : float
        Reference lattice constant.
    ref_coh : float
        Reference cohesive energy.
    compression : Bool, optional
        Whether model will be compressed. The default is False.
    crossval : Bool, optional
        Whether cross-validation will be employed. The default is True.
    frozen_model : str, optional
        Name of frozen model. The default is 'graph.pb'.
    compressed_model : str, optional
        Name of compressed model. The default is 'graph_compress.pb'.

    Raises
    ------
    ValueError
        If too many training input json files.

    Returns
    -------
    Results file that reports predicted cohesive energies and lattice\
        lattice constants, as well as residuals.

    """
    #parameters to be evaluated
    keyparams=glob.glob(os.path.join(directory,d1_dir,'*'))
    shortparams=[os.path.split(x)[1] for x in keyparams]
    #file to save metrics too
    with open('gridsearchresults.txt','w') as file:
        file.write('Results of grid search, results are\nseparated by \
                   parameter.\n')
        file.write(f"Parameters evaluated in grid search were:\
                   \n{shortparams}\n\n")
                   #for loop that goes over each keyparam
        for keyparam in keyparams:
            file.write(f"Key parameter : {os.path.split(keyparam)[1]}\n")
            #individual models for each key
            models=glob.glob(os.path.join(keyparam,'*'))
            #initalizing list to store lattice constant and cohesive energy
            #predictions
            latticeconstants=[]
            cohesiveenergies=[]
            #looping through all models for a given key
            for model in models:
                file.write(f"    Model = {os.path.split(model)[1]}\n")
                if crossval==True:
                    #sub models for kfold cross validation
                    kmodelpaths=glob.glob(os.path.join(model,'*'))
                    #new lists for kfold cross val
                    klatticeconstants=[]
                    kcohesiveenergies=[]
                    #training 'k' models, freezing them, compressing them 
                    #if necessary, and evaluating them in LAMMPS
                    for kmodelpath in kmodelpaths:
                        input_json=glob.glob(os.path.join(kmodelpath,'*.json'))
                        if len(input_json)==1:
                            train.train(input_script=input_json[0],directory=\
                                        kmodelpath)
                            post_training_handling.freeze(directory=kmodelpath\
                                                ,frozen_model=frozen_model)
                            if compression==False:
                                lattice_constant,cohesive_energy=\
                                    post_training_handling.lattice_constants(\
                                    lammps_script=lammps_script,\
                                        directory=kmodelpath,ref_len=ref_len,\
                                            ref_coh=ref_coh)
                                klatticeconstants.append(lattice_constant)
                                kcohesiveenergies.append(cohesive_energy)
                            elif compression==True:
                                post_training_handling.compress(directory=\
                                    kmodelpath,frozen_model=frozen_model,\
                                        compressed_model=compressed_model)
                                lattice_constant,cohesive_energy=\
                                post_training_handling.lattice_constants\
                                    (lammps_script=lammps_script,\
                                     directory=kmodelpath,\
                                         ref_len=ref_len, ref_coh=ref_coh)
                                        #appending predictions
                                klatticeconstants.append(lattice_constant)
                                kcohesiveenergies.append(cohesive_energy)
                        else: 
                            raise ValueError('kfold directory should contain\
                                            only one .json file, currently\
                                                there is more than one.')
                    #Predicition average for all folds of given model
                    latticeconstants.append(np.mean(klatticeconstants))
                    cohesiveenergies.append(np.mean(kcohesiveenergies))
                    latticeresidual=np.abs(np.mean(klatticeconstants)-ref_len)
                    cohesiveresidual=np.abs(np.mean(kcohesiveenergies)-ref_coh)
                    file.write(f"      Lattice Constant = {np.mean(klatticeconstants)}, residual = {latticeresidual}\n")
                    file.write(f"      Cohesive Energy = {np.mean(kcohesiveenergies)}, residual = {cohesiveresidual}\n")
                        #if cross val is false, only one model will be trained
                        #for given training parameters, not 'k' models
                elif crossval==False:
                    #ensuring only one json file in each model folder,
                    #used to reduce overwriting results
                    input_json=glob.glob(os.path.join(model,'*.json'))
                    #training, freezing, comp (if necessary) and eval in 
                    #LAMMPS
                    if len(input_json)==1:
                        train.train(input_script=input_json[0],directory=\
                                    model)
                        post_training_handling.freeze(directory=model,frozen_model=frozen_model)
                        if compression==False:
                            lattice_constant,cohesive_energy=\
                                post_training_handling.lattice_constants(\
                                lammps_script=lammps_script,\
                                    directory=model,ref_len=ref_len,\
                                        ref_coh=ref_coh)
                            latticeconstants.append(lattice_constant)
                            cohesiveenergies.append(cohesive_energy)
                        elif compression==True:
                            post_training_handling.compress(directory=\
                                model,frozen_model=frozen_model,\
                                    compressed_model=compressed_model)
                            lattice_constant,cohesive_energy=\
                            post_training_handling.lattice_constants\
                                (lammps_script=lammps_script,\
                                 directory=model,\
                                     ref_len=ref_len, ref_coh=ref_coh)
                            latticeconstants.append(lattice_constant)
                            cohesiveenergies.append(cohesive_energy)
                            latticeresidual=np.abs(lattice_constant-ref_len)
                            cohesiveresidual=np.abs(cohesive_energy-ref_coh)
                        file.write(f"      Lattice Constant = {lattice_constant}, residual = {latticeresidual}\n")
                        file.write(f"      Cohesive Energy = {cohesive_energy}, residual = {cohesiveresidual}\n")
                    else: 
                        raise ValueError('kfold directory should contain\
                                        only one .json file, currently\
                                            there is more than one.')
            