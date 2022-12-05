#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 13:33:53 2022

@author: michaelmacisaac
"""

import subprocess 
from pathlib import Path
import fileinput
import sys
import time
import numpy as np

def train(input_script,directory):
    """
    

    Parameters
    ----------
    input_script : json
        Input json.
    directory : str
        Working directory for function to be executed in.

    Returns
    -------
    None.

    """
    if input_script.endswith(".json"):
        if Path(directory).exists():
            with open(Path(directory,"train.out"),"w+") as fout:
                with open(Path(directory,"train.err"),"w") as ferr:
                    subprocess.run(["dp","train",f"{input_script}"],\
                                   cwd=directory,stdout=fout,stderr=ferr)
                    flag=0
                    while flag<1:
                        with open(Path(directory,'train.err')) as f:
                            lines=f.readlines()
                            for i in np.arange(-5,0,1):
                                words=lines[i].split()
                                if (("finished" in words) and ("training" in words)):
                                    flag=2
                            time.sleep(30)
                    time.sleep(10)
        else:
            raise ValueError("Provided directory does not exist.")
    else:
        raise ValueError("Extension must be .json")