# import libs 
#%%
import numpy as np
import os 
import shutil 
from glob import glob

# get input args
# training_path = '/home/kimia.gh/blue2/python_course/test_03/virials/training_data'
# destination_path = '/home/kimia.gh/blue2/python_course/DeeperMD'
# k = 10

#splits data into k random sets 
def k_split(full_list:list,k:int):
    """split full_list into k random subsets

    Args:
        full_list (list): full list to be sliced into k subsets
        k (int): number of subsets to slice full_list into

    Returns:
        dict: dictionary with k keys pointing to k subsets of full_list
    """

    #get size of list
    size = np.array(full_list).shape[0]

    #generate random indices
    idx = list(np.random.choice(size,size,replace=False))

    #initialize start and stop for list slicing
    start = 0
    end = 0

    #initialize df
    df = {}

    #populate df for k-1 folds
    for i in range(k-1):
        #update end of slice 
        end += int(1/k * size)

        #populate set i with slice i of list
        df[i]=np.array(full_list)[idx[start:end]]

        #update start of next slice
        start += int(1/k * size)

    #populate final set of df with remaining data 
    df[k-1]=np.array(full_list)[idx[start:-1]]

    return df

#splits
def gen_data_dir(training_path:str,destination_path:str,k:int):
    """generates directory tree structure for cross-validation function (*see documentation for dir structure)

    Args:
        training_path (str): path to training dir with systems and .nyp files
        destination_path (str): path to desired destination dir for dir tree
        k (int): _description_
    """

    cval_dir_path = os.path.join(destination_path,'cval')

    if os.path.isdir(cval_dir_path):
        shutil.rmtree(cval_dir_path)

    os.mkdir(cval_dir_path)

    systems = os.listdir(training_path)

    for sys in systems:
        #define paths for dest dir tree 
        all_path = os.path.join(cval_dir_path,'all_data',sys)
        train_path = os.path.join(cval_dir_path,'train',sys)
        val_path = os.path.join(cval_dir_path,'valid',sys)
        
        #make directory tree
        os.makedirs(all_path)
        os.makedirs(train_path)
        os.makedirs(val_path) 

        #copy all type* files over to new folder structure
        type_files = glob(os.path.join(training_path,sys,'type*'))
        for f in type_files:
            shutil.copy(f,all_path)   
            shutil.copy(f,train_path)   
            shutil.copy(f,val_path)   
        
        #find data for system
        fs = glob(os.path.join(training_path,sys,'*','*.npy'))
        
        for f in fs:
            #find data name 
            data_type = f.split('/')[-1]

            #load data
            data = np.load(f)

            #split data into k sets 
            split_data = k_split(data,k)

            for set_num in split_data:
                
                #set name from number 
                set_name = 'set.'+str(set_num).zfill(3)

                #get path to set dir 
                set_path = os.path.join(all_path,set_name)

                #filename 
                file_name = os.path.join(set_path,data_type)
                
                #make set dir
                if not os.path.isdir(set_path):
                    os.mkdir(set_path)      

                np.save(file_name,split_data[set_num])
    print(f'cross-validation data directory tree generated at:\n{cval_dir_path}')


# %%
