#%%
from dpdata import MultiSystems,LabeledSystem
import os 
import random
import numpy as np
import shutil


def OUTCAR_to_ms(parent_dir,sub_dirs=[''],run_types=['all'],flags=['OUTCAR']):
    """Converts OUTCAR files to MultiSystem objects

    Args:
        parent_dir (str): path to parent directory to all raw data
        sub_dirs (list, optional): potential existence of subdirs within parent (see Requirements). Defaults to [''].
        run_types (list, optional): run types within parent to include in data. Defaults to ['all'].
        flags (list, optional): flags indicating OUTCAR selection, must be file in OUTCAR folder. Defaults to ['OUTCAR'].

    Raises:
        Warning: LabeledSystem object was not able to be defined

    Returns:
        MultiSystem(): two MultiSystem() objects excluding and including virials
        dicts: total frame counts for systems with and without virials
    
    Requirements:
        parent_dir tree structure must follow structure:
             parent/system/run_type/runs/OUTCAR
             or
             parent/system/subdir/run_type/runs/OUTCAR
        e.g
            B4C/B12-CBC/temperature_hold/T5/OUTCAR
            or
            B4C/B12-CBC/small_strains/shear_strain/S25/OUTCAR
    """
    #define dpdata multisystem objects 
    ms_virial = MultiSystems()
    ms = MultiSystems()

    #keep count of number of frames in each run type
    counts_virial = {key: 0 for key in run_types}
    counts_novirial = {key: 0 for key in run_types}

    logs = []

    #scrub through defined parent directory
    for s in sub_dirs:
        #redefine parent dir 
        par = os.path.join(parent_dir,s)

        for r,d,f in os.walk(par):
            #extract information from path name 
            # print(r)
            system_info = r.split('/')[-4:]

            #define conditions for directory for extraction of OUTCAR
            hasflag = not set(flags).isdisjoint(f)
            isoutcar = 'OUTCAR' in f
            isdense = not 'X' in system_info[-2]

            if 'all' in run_types:
                inruntype=True
            else:
                inruntype = system_info[-3] in run_types
            
            if isoutcar and hasflag and inruntype and isdense:
                #get OUTCAR file path
                file_path = os.path.join(r,'OUTCAR')

                try:
                    #define labeled system with current OUTCAR
                    ls = LabeledSystem(file_path)

                    #separates data with and without virial stresses 
                    if 'virials' in ls.data.keys() and len(ls) > 0:
                        ms_virial.append(ls)
                        # print(counts_virial.keys())
                        counts_virial[system_info[-3]]+=ls.get_nframes()
                        logs.append(r)
                    else:
                        ms.append(ls)
                        counts_novirial[system_info[-3]]+=ls.get_nframes()
                        logs.append(r)
                except:
                    print(f'something is wrong with {r}')
                
                
    return ms,ms_virial,counts_novirial,counts_virial
#function for finding all non-training indices in splitting 
def find_missing(sub_list,full_list):
    """finds missing values from 0 to max value of full list, given a sublist

    Args:
        sub_list (list)): subset of list 
        full_list (int): max value of full list to compare against sub_list

    Returns:
        list: list of values in (0,full_list} that are not in sub_list
    """
    return [x for x in range(1, full_list) if x not in sub_list]

def train_test_split(destination_dir,ms_virial,ms,train_split=0.9):
    """scrubs destination dir for .npy files and splits into train-test sets

    Args:
        destination_dir (str): path to destination directory to write training and validation folders.
        train_split (float,optional): proportion of data to reserve for training. Defaults to 0.9.
        ms_virial (MultiSystem()): MultiSystem() object containing systems with virials.
        ms (MultiSystem()): MultiSystem() object containing systems without virials.
    """
    #requires that .npy are already in destination dir tree
    #walk destination_dir
    for r,d,f in os.walk(destination_dir):
        
        #check if we are in correct dir depth (with .npy files)
        if 'box.npy' in f:
            #makes training and validation dirs ./training/system/set.000/...
            ptrain = os.path.join(destination_dir,r.split('/')[-3],'training_data',*r.split('/')[-2:])
            ptest = os.path.join(destination_dir,r.split('/')[-3],'validation_data',*r.split('/')[-2:])
            
            #if train,val directories don't exist, make them
            if not os.path.isdir(ptrain):
                os.makedirs(ptrain)
            if not os.path.isdir(ptest):
                os.makedirs(ptest)
                
            #pick out all npy files for train-test splitting 
            files = [i for i in f if i.endswith('npy')]
            
            #define system 
            system = r.split('/')[-2]
            
            #find total number of frames 
            hasvirials = r.split('/')[-3] == 'virials'
            if hasvirials:
                n_frames = ms_virial[system].get_nframes()
            else:
                n_frames = ms[system].get_nframes()
            
            #find number of training frames 
            n_train = int(train_split * n_frames)
            
            #randomly choose training n_train indeces
            train_frames = random.sample(range(0,n_frames),n_train)

            #extract missing frames not chosen for training (for validation)
            test_frames = find_missing(train_frames,n_frames)
            
            #loops through all .npy's in r and adds them to dataframe 
            for file in files:
                #gets path to .npy file
                file_path = os.path.join(r,file)
                
                #load npy data
                with open(file_path):
                    data = np.load(file_path)
                
                #extract training and validation frames from data
                train_data = data[train_frames]
                test_data = data[test_frames]
                
                #define path for training and validation data file
                train_file = os.path.join(ptrain,file)
                test_file = os.path.join(ptest,file)
                
                #save split data into filename and path
                np.save(train_file,train_data)
                np.save(test_file,test_data)
            
            #copy type_map.raw and type.raw to train-test directories
            type_map = r+'/../type_map.raw'
            type_raw = r+'/../type.raw'
            
            shutil.copy(type_map,ptrain+'/..')
            shutil.copy(type_raw,ptrain+'/..')
            shutil.copy(type_map,ptest+'/..')
            shutil.copy(type_raw,ptest+'/..')

def OUTCAR_to_npy(parent_dir,destination_dir,sub_dirs=[''],run_types=['all'],flags=['OUTCAR'],train_split=0.9):
    """reads in data from OUTCAR output from VASP and converts to .npy files in training and validation folders.

    Args:
        parent_dir (str): path to parent directory to all raw data
        destination_dir (str): path to destination directory to write training and validation folders
        sub_dirs (list, optional): potential existence of subdirs within parent (see Notes). Defaults to [''].
        run_types (list, optional): run types within parent to include in data. Defaults to ['all'].
        flags (list, optional): flags indicating OUTCAR selection, must be file in OUTCAR folder. Defaults to ['OUTCAR'].
        train_split (float, optional): proportion of data to reserve for training. Defaults to 0.9.

    Raises:
        Exception: given parent directory does not exist
    """
    
    #check if destination directory exists
    if os.path.isdir(destination_dir):
        shutil.rmtree(destination_dir)

    os.mkdir(destination_dir)

    #check if parent dir exists
    if not os.path.isdir(parent_dir):
        raise Exception("parent directory does not exist.")
    
    ms,ms_virial,count_novirial,count_virial = OUTCAR_to_ms(parent_dir=parent_dir,sub_dirs=sub_dirs,run_types=run_types,flags=flags)

    #Convert systems to npys
    ms.to_deepmd_npy(os.path.join(destination_dir,'no_virials'),set_size = 1000000)
    ms_virial.to_deepmd_npy(os.path.join(destination_dir,'virials'),set_size = 1000000)

    train_test_split(destination_dir=destination_dir,train_split=train_split,ms_virial=ms_virial,ms=ms)
    return ms,ms_virial,count_novirial,count_virial
# #%%
# par = '/blue/subhash/kimia.gh/B4C_ML_Potential/data/devel/B4C'
# dest = '/blue/subhash/kimia.gh/python_course/test_02'
# sub = ['temperature_hold','small_strains']
# run = ['temperature_hold','shear_strain','volumetric_strain','uniaxial_strain']

# #%%
# a,b,c,d = OUTCAR_to_npy(parent_dir=par,destination_dir=dest,run_types=run)


# %%
