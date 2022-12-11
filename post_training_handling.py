#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:49:39 2022

@author: michaelmacisaac
"""
import subprocess 
from pathlib import Path
import fileinput
import sys
import time
import numpy as np


def freeze(directory, frozen_model='graph.pb'):
    """
    Function freezes trained models for later compression (if not using 
    type embedding) and other testing (e.g. dp test, lattice constants).

    Parameters
    ----------
    directory : path to model directory.
    frozen_model_name: desired name of frozen model, must have .pb extension.

    Returns
    -------
    Frozen model. Will be available in current working directory.

    """
    try:
        if frozen_model.endswith(".pb"):
            with open(Path(directory,"freeze.out"),"w+") as fout:
                with open(Path(directory,"freeze.err"),"w") as ferr:
                    subprocess.run(["dp","freeze","-o", f"{frozen_model}"],
                                   cwd=directory,stdout=fout,stderr=ferr)
                    flag = 0
                    while flag<1:
                        with open(Path(directory,'freeze.err')) as f:
                            lines=f.readlines()
                            #will look at the last 5 lines
                            for i in np.arange(-5,0,1):
                                if lines[i].find('ops in the final graph.'):
                                    flag=2
                            time.sleep(5)
            time.sleep(2)
        else:
            raise ValueError("Extension must be .pb")
    except AttributeError:
        print("There is no 'model_checkpoint_path'.")


def compress(directory, frozen_model='graph.pb', \
             compressed_model='graph-compress.pb'):
    """
    Function compresses the frozen model. Models that employ 'type_embedding' 
    duringmodel training cannot be compressed.
    
    Parameters
    ----------
    directory (str): Path to directory of frozen model.
    frozen_model : str, optional
        DESCRIPTION. Name of frozen model, the default is 'graph.pb'.
    compressed_model : str, optional
        DESCRIPTION. Name of compressed model, the
        default is 'graph-compress.pb'.

    Returns
    -------
    Compressed model. Will be available in current working directory.

    """
    try:
        if ((frozen_model.endswith(".pb")) and \
            (compressed_model.endswith(".pb"))):
            if Path(directory,frozen_model).exists():
                with open(Path(directory,"compress.out"),"w+") as fout:
                    with open(Path(directory,"compress.err"),"w") as ferr:
                        subprocess.run(["dp","compress","-i",f"{frozen_model}"\
                                        ,"-o",f"{compressed_model}"],\
                                       cwd=directory,stdout=fout,stderr=ferr)
                        flag = 0
                        while flag<1:
                            with open(Path(directory,'compress.err')) as f:
                                lines=f.readlines()
                                #will look at the last 5 lines
                                for i in np.arange(-5,0,1):
                                    if lines[i].find('ops in the final graph.'):
                                        flag=2
                                time.sleep(5)
                time.sleep(2)
            else:                            
                raise ValueError("There is no 'frozen_model' path")
        else:
            raise ValueError("frozen_model and compressed_model extensions\
                         must be .pb")
    except KeyError:
        print('Cannot compress model when using type embedding')


def test(directory, test_data, n, results = 'results',\
         frozen_model = 'graph.pb',compressed_model = 'graph-compress.pb',\
             compressed = False,multisystem=False):
    """
    Function evaluates the frozen and/or compressed model.

    Parameters
    ----------
    directory : str
        Path to model directory.
    test_data : str
        Path to test data.
    n : int
        Number of data points to test from test data.
    results: str
        Name of results file(s) head, will have 'e.out', 'f.out', 
        and 'v.out' appended.
    frozen_model : str, optional
        Name of frozen model, the default is 'graph.pb'.
    compressed_model : str, optional
        Name of compressed model, the default is 
        'graph-compress.pb'.
    compressed : bool, optional
        Specify whether model has been compressed following
        model freezing, the default is 'False', assume model training employs
        'type_embedding.'

    Returns
    -------
    results files, including energy (results.e.out), force (results.f.out) and 
    virials (results.v.out). These files will appear in the current working
    directory following function execution.

    """
    if compressed==False:
        if frozen_model.endswith(".pb"):
            if Path(Path(directory,frozen_model)).exists():
                with open(Path(directory,"test.out"),"w+") as fout:
                    with open(Path(directory,"test.err"),"w") as ferr:
                        subprocess.run(["dp","test","-m", f"{frozen_model}",\
                                        "-s",f"{test_data}","-n",f"{n}","-d",\
                                        f"{results}"],\
                                       cwd=directory,stdout=fout,stderr=ferr)
                        flag = 0
                        while flag<1:
                            if multisystem==False:
                                with open(Path(directory,'test.err')) as f:
                                    lines=f.readlines()
                                    if lines[-2].find('Virial RMSE/Natoms'):
                                        error=float(lines[-5].split()[5])
                                        flag=2
                                time.sleep(5)
                            if multisystem==True:
                                with open(Path(directory,'test.err')) as f:
                                    lines=f.readlines()
                                    if lines[-5].find('number of systems'):
                                        error=float(lines[-4].split()[5])
                                        flag=2
                                time.sleep(5)
                time.sleep(2)
            else:                           
                raise ValueError("There is no 'frozen_model' path")
        else:   
            raise ValueError("'frozen_model' extension must be .pb")
    elif compressed==True:
        if compressed_model.endswith(".pb"):
            if Path(Path(directory,compressed_model)).exists():
                with open(Path(directory,"test.out"),"w+") as fout:
                    with open(Path(directory,"test.err"),"w") as ferr:
                        subprocess.run(["dp","test","-m",\
                                        f"{compressed_model}","-s",\
                                        f"{test_data}","-n",f"{n}","-d",\
                                        f"{results}"],\
                                       cwd=directory,stdout=fout,stderr=ferr)
                        flag = 0
                        while flag<1:
                            if multisystem==False:
                                with open(Path(directory,'test.err')) as f:
                                    lines=f.readlines()
                                    if lines[-2].find('Virial RMSE/Natoms'):
                                        error=float(lines[-5].split()[5])
                                        flag=2
                                time.sleep(5)
                            if multisystem==True:
                                with open(Path(directory,'test.err')) as f:
                                    lines=f.readlines()
                                    if lines[-5].find('number of systems'):
                                        error=float(lines[-4].split()[5])
                                        flag=2
                                time.sleep(5)
                time.sleep(2)
            else:
                raise ValueError("There is no 'frozen_model' path")
                
        else: 
            raise ValueError("'frozen_model' extension must be .pb")
        
    return error


def lammps_lat_const_modifier(model,data,lammps_script='in.lattice_constants',\
                              units='metal',boundary='p p p'):
    """
    This function modifies the lammps input script for lattice
    constant evaluation.  The default parasmeters for this script are for
    a cubic SiC (SiC-3C) system. Therefore all parameters should be modified 
    for the system of interest. Defaults are provided for users to become 
    adept with the code package.

    Parameters
    ----------
    model : str
        Path to model for LAMMPS simulation. Should be .pb file.
        data : str
            Material data file to be read
    lammps_script : str, optional
        LAMMPS input script to be modified. 
        The default is 'in.lattice_constants'.
    units : str, optional
        Unit type to be used in LAMMPS simulation. The default is 'metal'.
    boundary : str, optional
        Boundary conditions for LAMMPS simulation. The default is 'p p p'. 

    Returns
    -------
    Will modify an existing file.

    """
    if model.endswith('.pb')==True:
        #key text words to search for in LAMMPS input script
        search_text=['boundary','units','read_data','pair_style']
        replace_text=[f"boundary {boundary} \n",f"units {units} \n",\
                      f"read_data {data} \n",f"pair_style deepmd {model} \n"]
         #opening lammps_script, reading, and replacing file with arguments
        with open(f"{lammps_script}",'r') as file:
             lines=file.readlines()
        newlines=[]
        for line in lines:
            found=False
            for st,rt in zip(search_text,replace_text):
                 if line.startswith(st):
                     newlines.append(rt)
                     found=True
            if found==False:
                newlines.append(line)
        print(newlines[0:10])
        with open(lammps_script,'w') as file:
             file.writelines(newlines)
                
    else:
        raise ValueError("Model must have .pb extension.") 
        

def lattice_constants(lammps_script, directory, ref_len, ref_coh):
    """
    Function will calculate the lattice constants of a structure. 

    Parameters
    ----------
    lammps_script : LAMMPS input script
        LAMMPS input script. Should have correct potential and correct 
        material structure in script. 
    directory : str
        Working directory for LAMMPS simulation to be performed in.
    ref_len : numeric
        Reference lattice constant, in angstroms..
    ref_coh : numeric
        Reference cohesive energy. Should be in eV/atom.

    Returns
    -------
    lattice Constant, cohesive energy.

    """
    
    flag=0 #setting flag variable, will become 2 when simulation completion
    
    #opening up error and out files for simulation
    with open(Path(directory,"lattice_constant.out"),"w+") as fout:
        with open(Path(directory,"lattice_constant.err"),"w") as ferr:
            
            #subprocess command that executes lammps simulation
            subprocess.run(["lmp","-i",f"{lammps_script}"], cwd=directory,\
                           stdout=fout,stderr=ferr)
            #in this while loop we will look through the log.lammps file for 
            #the lattice constant, cohesive energy, and if the job has
            #finished
            while flag<1:
                with open(Path(directory,'log.lammps')) as f:
                    lines=f.readlines()
                    count=0
                    #will look at the last 15 lines
                    for i in np.arange(-15,0,1):
                        
                        if lines[i].find("Total wall time:"):
                            lattice_constant=float(lines[-8].split()[-1].replace(';',''))
                            cohesive_energy = float(lines[-6].split()[-1].replace(';',''))
                            flag=2
                        count+=1
                    time.sleep(5)
    time.sleep(2)
                    
    return lattice_constant, cohesive_energy
