#%%

from deepermd.training import post_training_handling
from deepermd.training import train
import json
import glob as glob
import os 
import numpy as np
import logging
from functools import reduce
import operator
import subprocess
import shutil

def log_maker(file_name):
    """
    Function that makes logging file.

    Parameters
    ----------
    file_name : str
        File for log messages to be appended to.

    Returns
    -------
    Log file.

    """
    logging.basicConfig(filename=file_name,
                        format='%(asctime)s %(message)s',
                        filemode='w')
     
    # Creating an object
    logger = logging.getLogger()
     
    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)
    
    return logger
    
def set_switcher(path_to_cval,switch_set_index):
    """
    Function switches sets in training and validation paths. Used when 
    gen_cval_data function is used. Will be used whether cross-validation 
    is employed or not.

    Parameters
    ---------- 
    path_to_cval : str
        Path to where data generated using gen_cval_data function is.
    switch_set_index : integer
        Index of set list, this index will be set to the validation set.

    Returns
    -------
    None.

    """
    #move all sets from each system to training 
    #move switch set from each system to validation
    
    if os.path.isdir(path_to_cval)==False:
        raise ValueError(f'Path: {path_to_cval},\ndoes not exist')
    if type(switch_set_index)!=int:
        raise ValueError("Argument 'switch_set' must be 'int' type.")
        
    #Systems in cval_data directory
    cval_data_systems=glob.glob(os.path.join(path_to_cval,'all_data','*'))
    #List of names of individual systems
    ind_systems=[os.path.split(x)[1] for x in cval_data_systems]
    #Defining a list of sets, this list will be used to verify that all systems
    #have the same amount of sets
    sets=glob.glob(os.path.join(cval_data_systems[0],'set.*'))
    #Defining the 'switch_set', the 'switch_set' is the set that will be 
    #moved to the 'valid' folder of each system. It is dependent on the 
    #'switch_set_index' argument. During cross validation, the switch_set_index
    #will change. We use 'os.path.split() to grab the last item of the path,
    #this item is the set
    switch_set=os.path.split(sets[switch_set_index])[1]
    #In this loop we will verify that each system has the same amount of sets
    for system in cval_data_systems:
        if len(glob.glob(os.path.join(system,'set.*')))!=len(sets):
            raise ValueError('Systems must have the same amount of sets')
        if len(glob.glob(os.path.join(system,'set.*')))==0:
            raise ValueError('Systems should be populated with sets')
    #In this loop we will move all sets from each system in the cval_data 
    #folder to their corresponding system folder in the train folder. Once 
    #system sets have been moved to their system folder within the train 
    #folder, the switch_set (validation set) will be moved from the system 
    #folder in train to the system folder in valid
    for system in ind_systems:
        cval_sets=glob.glob\
            (os.path.join(path_to_cval,'all_data',system,'set.*'))
        training_system=os.path.join(path_to_cval,'train',system)
        train_switch_set=os.path.join(training_system,switch_set)
        validation_system=os.path.join(path_to_cval,'valid',system)
        for cval_set in cval_sets:
            subprocess.run(["mv","-t",f"{training_system}",f"{cval_set}"])
        subprocess.run(["mv","-t",f"{validation_system}"\
                        ,f"{train_switch_set}"])


def set_return(path_to_cval):
    
    #Systems in cval_data directory
    cval_data_systems=glob.glob(os.path.join(path_to_cval,'all_data','*'))
    #List of names of individual systems
    ind_systems=[os.path.split(x)[1] for x in cval_data_systems]
    
    for system in ind_systems:
        training_sets=glob.glob\
            (os.path.join(path_to_cval,'train',system,'set.*'))
        validation_set=glob.glob\
            (os.path.join(path_to_cval,'valid',system,'set.*'))[0]
        cval_data_system=os.path.join(path_to_cval,'all_data',system)       
        subprocess.run(["mv","-t",f"{cval_data_system}",f"{validation_set}"])
        for training_set in training_sets:
            subprocess.run(["mv","-t",f"{cval_data_system}",f"{training_set}"])

def getFromDict(base_json,key_list):
    """
    Function evealuates whether provided key to modify exists in base_json.

    Parameters
    ----------
    base_json : json file
        Base json file to be modified.
    key_list : list
        List of keys to check.

    """
    return reduce(operator.getitem,key_list,base_json)

def setInDict(base_json,key_list,value):
    """
    

    Parameters
    ----------
    base_json : json file
        Base json to be modified.
    key_list : list
        Key corresponding to value.
    value : list, numeric
        value to be changed.

    Returns
    -------
    None.

    """
    getFromDict(base_json,key_list[:-1])[key_list[-1]] = value



def json_dir_gen_1d(base_json, param_dict,path_to_cval=None,\
        gen_cval_data=True,training_path=None,validation_path=None,\
            crossval=True,d1_dir='1d_gridsearch'):
    """
    Function will take a base json as well as a json of NN params
    and generate json files that 

    Parameters
    ----------
    base_json: json file 
        Base json file to be modified, all parameters to be modified
        must be present in this json. Parameters that will not be optimized,
        such as parameters with only value/set of values to be tested, should
        have this value/set of values set in the base json.
    param_dict: dict
        Dictionary with parameters to be modified. Should be one dictionary,
        NO NESTED dictionaries. Dictionary keys should be strings of space
        separated keys. Values should be lists.
    path_to_cval: str
        Path to data.
    gen_cval_data: bool
        Whether data was generated using 'gen_cval_data' function within the 
        'data_prep' module. If 'True' training and validation paths
         will be made by appending "train" and 
        "valid" to the provided 'path_to_cval' path, making the training and 
        validation paths, respectively. 'path_to_cval' must be provided
        if 'cval' set to True.
    crossval: bool
        Whether cross validation sub directories and jsons should be made.
    d1_dir: str
        Name of 1-dimensional gridsearch directory. This directory is where 
        all jsons and models will be stored. Sub directories pertaining to 
        each parameter and their various values will be made to store
        individual jsons and files.

    Returns
    ------
    Directories for different json parameter combinations and jsons.
    """
    #evaluating whether d1_dir exists
    if os.path.isdir(d1_dir):
        shutil.rmtree(d1_dir)
        os.mkdir(d1_dir)
    else: 
        os.mkdir(d1_dir)
        
    #opening initial json file
    with open(base_json) as jsonfile:
        base_json=json.load(jsonfile)
    original_input=base_json
    if gen_cval_data==True:
        if os.path.isdir(path_to_cval)==False:
            raise ValueError(f"No such path exists: {path_to_cval}")
        if path_to_cval==None:
            raise ValueError("Path to cval data must be provided if parameter\
                             'cval' set to 'True'")
        base_json["training"]["training_data"]["systems"]=\
            os.path.join(path_to_cval,'train')
        base_json["training"]["validation_data"]["systems"]=\
            os.path.join(path_to_cval,"valid")
    else:
        base_json["training"]["training_data"]["systems"]=training_path
        base_json["training"]["validation_data"]["systems"]=validation_path
    #loop through all keys in input dictonary
    for key,values in param_dict.items():
        count=0
        #splitting string into keys
        key_list=key.split(' ')
        #making directories relevant to keys investigated
        key_dir=os.path.join(d1_dir,f"{key.replace(' ','_')}")
        os.mkdir(key_dir)
        try:
            getFromDict(base_json,key_list)
        except:
            raise ValueError(f"{key} is not valid")
            break
        #looping through all values/list of values in each key
        for value in values:
            #setting json keys to desired values
            setInDict(base_json, key_list, value)
            #making directories for each model
            dir_name=os.path.join(key_dir,f"model_{count}")
            os.mkdir(dir_name)
            #if cross val is selected
            if crossval==True:
                if gen_cval_data==False:
                    raise ValueError("To perform cross-validation, gen_cval\
                                     _data must be True.")
               # kcount=0
                systems=glob.glob(os.path.join(path_to_cval,'all_data','*'))
                kfolds=len(glob.glob(os.path.join(systems[0],'set.*')))
                for i in range(kfolds):
                    kdir_name=os.path.join(dir_name,f"k_{i}")
                    os.mkdir(kdir_name)
                    file_name=os.path.join(kdir_name,'input.json')
                    with open(file_name,'w+') as modified_json:
                        json.dump(base_json,modified_json,indent=4)
                    #kcount+=1
            elif crossval==False:
                file_name=os.path.join(dir_name,'input.json')
                with open(file_name,'w+') as modified_json:
                    json.dump(base_json,modified_json,indent=4)
            count+=1
            base_json=original_input



def hyperparam_train(directory,d1_dir,path_to_cval=None,gen_cval_data=True\
        ,compression=False,crossval=True,frozen_model='graph.pb',\
        compressed_model='graph-compress.pb'):
    """
    This function trains models using input json files that were created
    in order to perform a 1d hyperparameter optimization/evaluation study.
    There are options for model compression and cross-validation. This 
    function will evaluate trained models ability to predict lattice constants
    and cohesive energies. Model predictions and errors/residuals will be 
    reported in a text file.

    Parameters
    ----------
    directory : str
        Parent directory, where script will be run.
    d1_dir : str
        directory name for 1d gridsearch.
    path_to_cval : str or None
        Path to data generated via gen_cval_data function. Must use gen_cval_
        data function if crossval is to be used.
    compression : Bool, optional
        Whether model will be compressed. The default is False. If using 
        type embedding set to False.
    crossval : Bool, optional
        Whether cross-validation will be employed. The default is True. Only 
        use if cross-validation sets were generated.
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
    #below try and except block is to ensure all data is within the all data
    #folder in the folder location specified by 'path_to_cval'
    try:
        set_return(path_to_cval=path_to_cval)
    except IndexError:
        pass
        
    logger=log_maker("hyperparam_train.log")
    #Checking for correct inputs
    if os.path.isdir(directory)!=True:
        raise ValueError('Provided directory argument is not a path.')
    if os.path.isdir(os.path.join(directory,d1_dir))!=True:
        raise ValueError('Provided d1_dir argument is not within provided\
                         directory.')
    if ((gen_cval_data!=True) and (gen_cval_data!=False)):
        raise ValueError('gen_cval_data must be either True or False')
    if ((compression!=True) and (compression!=False)):
        raise ValueError('compression must be either True or False')
    if ((crossval!=True) and (crossval!=False)):
        raise ValueError('gen_cval_data must be either True or False')
    if ((frozen_model.endswith('.pb')!=True) or \
        (isinstance(frozen_model,str)!=True)):
        raise ValueError\
            ('Frozen model must be a string and have a .pb extension')
    if compression==True:
        if ((compressed_model.endswith('.pb')!=True) or \
            (isinstance(compressed_model,str)!=True)):
            raise ValueError('Compressed model must be a string and \
                             have a .pb extension.')
    #parameters to be evaluated
    keyparams=glob.glob(os.path.join(directory,d1_dir,'*'))
    shortparams=[os.path.split(x)[1] for x in keyparams]
                   #for loop that goes over each keyparam
    for keyparam in keyparams:
        logger.info\
            (f"Investigating key parameter : {os.path.split(keyparam)[1]}\n")
        #individual models for each key
        models=glob.glob(os.path.join(keyparam,'*'))
        #looping through all models for a given key
        for model in models:
            logger.info(f"Training model : {model}")
            if crossval==True:
                if os.path.isdir(path_to_cval)!=True:
                    raise ValueError('Provided path_to_cval argument is not a\
                                     valid directory')
                if gen_cval_data!=True:
                    raise ValueError('gen_cval_data function must be used and\
                        argument must equal True if crossval equal True.')
                logger.info("Training using cross validation")
                #sub models for kfold cross validation
                kmodelpaths=glob.glob(os.path.join(model,'*'))
                #training 'k' models, freezing them, compressing them 
                #if necessary
                for i,kmodelpath in enumerate(kmodelpaths):
                    logger.info(f"Training fold {i}")
                    input_json=glob.glob(os.path.join(kmodelpath,'*.json'))
                    if len(input_json)==1:
                        logger.info("Training began")
                        set_switcher(path_to_cval=path_to_cval\
                                     ,switch_set_index=i)
                        train.train(input_script=input_json[0],directory=\
                                    kmodelpath)
                        logger.info(f"Training completed for fold {i}")
                        logger.info(f"Freezing model {model} fold {i}")
                        post_training_handling.freeze(directory=kmodelpath\
                                            ,frozen_model=frozen_model)
                        logger.info("Freezing complete")
                        set_return(path_to_cval=path_to_cval)
                        if compression==True:
                            post_training_handling.compress(directory=\
                                kmodelpath,frozen_model=frozen_model,\
                                    compressed_model=compressed_model)
                            logger.info("Model compressed")
                    else: 
                        raise ValueError('kfold directory should contain\
                                        only one .json file, currently\
                                            there is more than one.')
            elif crossval==False:
                logger.info("Training without cross validation")
                #ensuring only one json file in each model folder,
                #used to reduce overwriting results
                input_json=glob.glob(os.path.join(model,'*.json'))
                #training, freezing, comp (if necessary) and eval in 
                #LAMMPS
                if len(input_json)==1:
                    if gen_cval_data==True:
                        logger.info(f"Training model : {model}")
                        set_switcher(path_to_cval=path_to_cval,\
                                     switch_set_index=0)
                        train.train(input_script=input_json[0],directory=\
                                    model)
                        logger.info("Training completed")
                        set_return(path_to_cval=path_to_cval)
                    elif gen_cval_data==False:
                        logger.info(f"Training model : {model}")
                        train.train(input_script=input_json[0],directory=\
                                    model)
                        logger.info("Training completed")
                else: 
                    raise ValueError('kfold directory should contain\
                                    only one .json file, currently\
                                        there is more than one.')
                logger.info(f"Freezing model {model}")
                post_training_handling.freeze(directory=model\
                                    ,frozen_model=frozen_model)
                logger.info("Freezing complete")
                if compression==True:
                    post_training_handling.compress(directory=\
                        model,frozen_model=frozen_model,\
                            compressed_model=compressed_model)
                    logger.info("Model compressed")
        logger.info("\n\n")
            

def hyperparam_test(directory,d1_dir,param_dict,test_model,n,path_to_cval=None\
                    ,multisystem=True,test_path=None,crossval=True):
    """
    

    Parameters
    ----------
    directory : str
        Directory where script is.
    d1_dir : str
        Name of directory where generated jsons are. Must match directory name
        argument used in train and json_dir_gen_1d functions.
    param_dict : dict
        Dictionary of parameter values used to generate jsons.
    test_model : str
        Name of model to be tested, must have .pb extension.
    n : integer
        Number of data points to test from test data.
    multisystem : bool
        Whether multiple systems were used during model training and hence 
        whether multiple systems should be tested.
    test_path : str
        Path to test data. The default is None.
    crossval : bool
        Whether cross validation was employed. The default is True.

    Returns
    -------
    None.

    """
    logger=log_maker("hyperparam_test.log")
    keys=param_dict.keys()
    with open('gridsearchtestresults.txt','w') as file:
        file.write('Results of grid search are separated by parameter.\n')
        file.write(f"Parameters evaluated in grid search were:\
                   \n{keys}\n\n")
        #list of directories that pertain models with different values for 
        #specific keys
        keymodels=glob.glob(os.path.join(directory,d1_dir,'*'))
            #will loop through each different key
        for keymodel in keymodels:
            file.write("Error is being evaluated for the following key")
            file.write(f" {os.path.split(keymodel)[1]}\n\n")
            #List of different models with different key values 
            #for a given key
            subkeymodels=glob.glob(os.path.join(keymodel,'*'))
            #wil loop through each model that pertains to each different value
            #for a specific key
            for subkeymodel in subkeymodels:
                file.write("  Error is being evaluated for the following ")
                file.write(f"model {os.path.split(subkeymodel)[1]}\n")
                if test_path==None:
                    if os.path.isdir(path_to_cval)!=True:
                        raise ValueError(f"Directory {path_to_cval} \
                                         does not exist")
                    if crossval==True:
                        kmodels=glob.glob(os.path.join(subkeymodel,'*'))
                        #initalizing empty list for crossval errors
                        crossvalerrors=[]
                        for i,kmodel in enumerate(kmodels):
                            set_switcher(path_to_cval=path_to_cval,\
                                         switch_set_index=i)
                            test_path=os.path.join(path_to_cval,'valid')
                            error=post_training_handling.test(directory=kmodel\
                                ,test_data=test_path,model=test_model,n=n,\
                                        multisystem=multisystem)
                            crossvalerrors.append(error)
                            set_return(path_to_cval=path_to_cval)
                        meanerror=np.mean(crossvalerrors)
                        file.write(f"  Error = {meanerror}")
                    elif crossval==False:
                        set_switcher(path_to_cval=path_to_cval,\
                                     switch_set_index=0)
                        test_path=os.path.join(path_to_cval,'valid')
                        error=post_training_handling.test(directory=\
                            subkeymodel,test_data=test_path,model=test_model\
                                ,n=n,multisystem=multisystem)
                        file.write(f"  Error = {error}\n")
                    else:
                        raise ValueError('crossval argument must be True or\
                                         False')
                else:
                    if os.path.isdir(test_path)!=True:
                        raise ValueError('Provided test_path is not a\
                                         directory')
                    error=post_training_handling.test(directory=subkeymodel,\
                        test_data=test_path,model=test_model,n=n,\
                                multisystem=multisystem)
                    file.write(f"  Error = {error}\n")
                        
                
                   
def hyperparam_lammps(directory,d1_dir,lammps_script,ref_len,ref_coh,\
                      crossval=None):
    logger=log_maker("hyperparam_lammps.log")
    with open('gridsearchlammpsresults.txt','w') as file:
        #parameters to be evaluated
        keyparams=glob.glob(os.path.join(directory,d1_dir,'*'))
        shortparams=[os.path.split(x)[1] for x in keyparams]
                       #for loop that goes over each keyparam
        for keyparam in keyparams:
            logger.info("Investigating key parameter : ")
            logger.info(f"{os.path.split(keyparam)[1]}\n")
            file.write(f"Key parameter : {os.path.split(keyparam)[1]}\n")
            #individual models for each key
            models=glob.glob(os.path.join(keyparam,'*'))
            #looping through all models for a given key
            for model in models:
                file.write(f"    Model = {os.path.split(model)[1]}\n")
                if crossval==True:
                    #sub models in kfold cross validation
                    kmodelpaths=glob.glob(os.path.join(model,'*'))
                    #new lists for kfold cross val
                    klatticeconstants=[]
                    kcohesiveenergies=[]
                    for i,kmodelpath in enumerate(kmodelpaths):
                        file.write(f"      Fold {i}\n")
                        lattice_constant,cohesive_energy=\
                            post_training_handling.lattice_constants(\
                            lammps_script=lammps_script,\
                                directory=kmodelpath,ref_len=ref_len,\
                                    ref_coh=ref_coh)
                        logger.info(f"Lattice constants calculated for fold {i}")
                        klatticeconstants.append(lattice_constant)
                        kcohesiveenergies.append(cohesive_energy)
                        latticeconstant=np.mean(klatticeconstants)
                        cohesiveenergy=np.mean(kcohesiveenergies)
                        latticeresidual=np.abs(latticeconstant-ref_len)
                        cohesiveresidual=np.abs(cohesiveenergy-ref_coh)
                        file.write("        Lattice Constant = ")
                        file.write(f"{latticeconstant}, residual = ")
                        file.write(f"{latticeresidual}\n")
                        file.write("        Cohesive Energy = ")
                        file.write(f"{cohesiveenergy}, residual = ")
                        file.write(f"{cohesiveresidual}\n")
                elif crossval==False:
                        lattice_constant,cohesive_energy=\
                            post_training_handling.lattice_constants(\
                            lammps_script=lammps_script,\
                                directory=model,ref_len=ref_len,\
                                    ref_coh=ref_coh)
                        logger.info("Lattice constants calculated")
                        latticeresidual=np.abs(lattice_constant-ref_len)
                        cohesiveresidual=np.abs(cohesive_energy-ref_coh)
                        file.write("      Lattice Constant = ")
                        file.write(f"{lattice_constant}, residual = ")
                        file.write(f"{latticeresidual}\n")
                        file.write("      Cohesive Energy = ")
                        file.write(f"{cohesive_energy}, residual = ")
                        file.write(f"{cohesiveresidual}\n")
            logger.info("\n\n")
            
def hyperparam_optimize(directory,base_json,param_dict,n,test_model,\
          d1_dir='1d_gridsearch',frozen_model='graph.pb'\
          ,compression=True,compressed_model='graph-compress.pb',\
              path_to_cval=None,gen_cval_data=None,crossval=False,\
              training_path=None,validation_path=None,test_path=None,\
                  multisystem=True,lammps=True,lammps_model=None,\
                lammps_script=None,lammps_data=None,units='metal',\
                      boundary='p p p',ref_len=None,ref_coh=None):
    """
    This function wraps the following functions together: 
        hyperparam_train_test.json_dir_gen_1d
        hyperparam_train_test.hyperparam_train
        hyperparam_train_test.hyperparam_test
        post_training_handling.lammps_lat_const_modifier
        hyperparam_train_test.hyperparam_lammps
        
        This function goal is to generate json and model directories 
        corresponding to a user specified parameter dictionary. Models will 
        be trained based on json, frozen, compressed (if desicred), 
        tested, and their lattice constants and cohesive energies evaluated
        via LAMMPS simulations (if desired).
        

    Parameters
    ----------
    directory : str
        Directory where script is located.
    base_json : json file
        Base json to be modified.
    param_dict : dict
        Dictionary of parameters and their values. 
    n : int
        Number of data points to use when testing trained models.
    test_model : str (model with .pb extension)
        Name of model to be tested.
    d1_dir : str, 
        Name of parent directory where generated jsons will be stored. 
        The default is '1d_gridsearch'.
    frozen_model : str
        Desired name of frozen model, must have .pb extension. 
        The default is 'graph.pb'.
    compression : bool
        Whether to compress model following model training
        and freezing. Compression cannot be enabled if 'type_embedding' is 
        used. The default is True.
    compressed_model : str
        Name of compressed model, must have .pb extension.
        The default is 'graph-compress.pb'.
    path_to_cval : str
        Path to date made via gen_cval_data. The default is None.
    gen_cval_data : bool
        Whether data was made with gen_cval_data function. The default is None.
    crossval : bool
        Whether cross validation should be employed. The default is False.
    training_path : str
        Path to training data. Should be supplied if gen_cval_data is False.
        The default is None.
    validation_path : str
        Path to validation data. Must be provided if gen_cval_data is False.
        The default is None.
    test_path : str
        Path to testing data. If gen_cval_data is True and path_to_cval input
        is valid, then the validation set will be used and this argument can
        be set to None. However, if both of these conditions are not satisfied
        and traing_path and validation_path arguments are supplied, the 
        test_path must be provided. The default is None.
    multisystems : bool
        Whether models were trained with multisystem data.
        The default is False.
    lammps : bool
        Whether to perform lammps simulations on developed models.
        The default is False.
    lammps_model: str
        model to be written in lammps input file, must have .pb extension.
    lammps_data: str
        data for lammps simulation.
    units: str
        Units to condict lammps simulation with. String will be written in
        lammps input script. The default is metal.
    boundary: str
        boundary conditions for lammps simulation. The default is p p p.
    lammps_script : .in file
        Name of lammps input file. Full path should be provided.
        Argument must be provided if 'lammps' argument equals True.
        The default is None.
    ref_len : float
        Reference lattice constant. The default is None.
    ref_coh : float
        Reference cohesive energy. The default is None.

    Returns
    -------
    None.

    """
    #calls functio to generate jsons for each parameter and value
    #to be tested
    json_dir_gen_1d(base_json=base_json,param_dict=param_dict,\
        path_to_cval=path_to_cval,gen_cval_data=gen_cval_data,\
        training_path=training_path,validation_path=validation_path,\
            crossval=crossval,d1_dir=d1_dir)
    #calls function to train models, freeze, and compress if necessary
    hyperparam_train(directory=directory,d1_dir=d1_dir,\
        path_to_cval=path_to_cval,gen_cval_data=gen_cval_data,\
        compression=compression,crossval=crossval,frozen_model=frozen_model,\
            compressed_model=compressed_model)
    hyperparam_test(directory=directory,d1_dir=d1_dir,param_dict=param_dict,\
        test_model=test_model,n=n,path_to_cval=path_to_cval,\
        multisystem=multisystem,test_path=test_path,crossval=crossval)
    if lammps==True:
        post_training_handling.lammps_lat_const_modifier(lammps_model=lammps_model,lammps_script=lammps_script,\
            lammps_data=lammps_data,units=units,boundary=boundary)
        hyperparam_lammps(directory=directory,d1_dir=d1_dir,\
            lammps_script=lammps_script,ref_len=ref_len,ref_coh=ref_coh,\
            crossval=crossval)
    else:
        pass
            
    


# if __name__ == '__main__':
#     main(directory,base_json,param_dict,n,test_model,\
#              d1_dir='1d_gridsearch',frozen_model='graph.pb'\
#              ,compression=True,compressed_model='graph-compress.pb',\
#                  path_to_cval=None,gen_cval_data=None,crossval=False,\
#                  training_path=None,validation_path=None,test_path=None,\
#                      multisystems=False,lammps=False,lammps_model=None,\
#                     lammps_script=None,lammps_data=None,units='metal',\
#                          boundary='p p p',ref_len=None,ref_coh=None)