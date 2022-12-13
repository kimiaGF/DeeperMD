
import os 
import glob as glob
import json
from functools import reduce
import operator

def data_path_gen(path_to_cval):
    """
    Function will generate data paths to be inputted into input json files.

    Parameters
    ----------
    path_to_cval : str
        Path to cross-validation data

    Returns
    -------
    Training and testing data path(s).

    """
    if os.path.isdir(path_to_cval):
        systempaths=glob.glob(os.path.join(path_to_cval,'*'))
        systems=[]
        for systempath in systempaths:
            systems.append(os.path.split(systempath)[1])
    else:
        raise NameError('provided "path_to_cval" is not a directory')
    #check if all systems have the same amount of sets
    for systempath in systempaths:
        if len(glob.glob(os.path.join(systempath,'set*')))==\
            len(glob.glob(os.path.join(systempaths[0],'set*'))):
            continue
        else:
            raise ValueError("Systems must have the same amount of sets")
    sets=[]
    for x in glob.glob(os.path.join(systempaths[0],'set*')):
        sets.append(os.path.split(x)[1])
    
    training_paths=[]
    validation_paths=[]
    if len(sets)>1:
        for valset in sets:
            train_path=[]
            validation_path=[]
            trainsets=[set for set in sets if set!=valset]
            for system in systems:
                validation_path.append(os.path.join(path_to_cval,\
                                                    system,valset))
                for trainset in trainsets:
                    train_path.append(os.path.join(path_to_cval,\
                                                   system,trainset))
            training_paths.append(train_path)
            validation_paths.append(validation_path)
    else:
        raise ValueError('Cannot perform cross-validation with a single set')
    
    return training_paths, validation_paths

def exists(obj, chain):
    _key = chain.pop(0)
    if _key in obj:
        return exists(obj[_key], chain) if chain else obj[_key]

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



def json_dir_gen_1d(base_json, param_dict,training_paths,validation_paths,crossval=True,d1_dir='1d_gridsearch'):
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

    Returns
    ------
    Directories for different json parameter combinations and jsons.
    """
    #evaluating whether d1_dir exists
    if not os.path.isdir(d1_dir):
        os.mkdir(d1_dir)
    else: 
        raise ValueError('Directory already exists, choose different name \
                         or risk overwriting.')
    #opening initial json file
    with open(base_json) as jsonfile:
        base_json=json.load(jsonfile)
    original_input=base_json
    #loop through all keys in input dictonary
    for key,values in param_dict.items():
        count=0
        #splitting string into keys
        key_list=key.split(' ')
        #making directories relevant to keys investigated
        key_dir=os.path.join(d1_dir,f"{key_list[-1]}")
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
                kcount=0
                for training_path, validation_path in zip(training_paths,validation_paths):
                    kdir_name=os.path.join(dir_name,f"k_{kcount}")
                    os.mkdir(kdir_name)
                    file_name=os.path.join(kdir_name,'input.json')
                    base_json["training"]["training_data"]["systems"]=training_path
                    base_json["training"]["validation_data"]["systems"]=validation_path
                    with open(file_name,'w+') as modified_json:
                        json.dump(base_json,modified_json,indent=4)
                    kcount+=1
            elif crossval==False:
                file_name=os.path.join(dir_name,'input.json')
                base_json["training"]["training_data"]["systems"]=training_paths
                base_json["training"]["validation_data"]["systems"]=validation_paths
                with open(file_name,'w+') as modified_json:
                    json.dump(base_json,modified_json,indent=4)
            count+=1
            base_json=original_input

                
        
