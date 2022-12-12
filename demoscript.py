#%%
from training import hyperparam_train_test
from training import post_training_handling

#%%
trainpath='/home/kimia.gh/blue2/python_course/test_03/virials/training_data'
validpath='/home/kimia.gh/blue2/python_course/test_03/virials/validation_data'
params={"model descriptor rcut":[6.0,8.0],"model descriptor rcut_smth":[1.0,4.5]}
#params={"model descriptor rcut":[4.0,5.0,6.0,7.0],"model descriptor rcut_smth":[1.0,1.5,2.0,2.5],"model descriptor axis_neuron":[4,8,12,16],"model fitting_net neuron":[[10,10],[50,50],[50,50,50],[10,10,10],[100,100],[100,100,100]]}
#%%
hyperparam_train_test.json_dir_gen_1d(
        base_json='base.json', 
        param_dict=params,
        path_to_cval=None,
        gen_cval_data=False,
        training_path=trainpath,
        validation_path=validpath,
        crossval=False,
        d1_dir='1d_gridsearch_2')

#%%
hyperparam_train_test.hyperparam_train(
        directory='/home/kimia.gh/blue2/python_course/DeeperMD',
        d1_dir='1d_gridsearch_2',
        path_to_cval=None,
        gen_cval_data=False,
        compression=False,
        crossval=False,
        frozen_model='graph.pb',
        compressed_model='graph-compress.pb')

#%%
hyperparam_train_test.hyperparam_test(directory='/blue/subhash/michaelmacisaac/functions/deepmd/model1',d1_dir='1d_gridsearch_2'\
        ,param_dict=params,test_model='graph.pb',n=30,\
        path_to_cval=None,\
        multisystems=False,\
        test_path=validpath,\
        crossval=False)                                  
post_training_handling.lammps_lat_const_modifier(model='graph.pb',\
    data='/blue/subhash/michaelmacisaac/functions/deepmd/model1/data.b4c_cell',\
        lammps_script='/blue/subhash/michaelmacisaac/functions/deepmd/model1/in.lattice_constants')
hyperparam_train_test.hyperparam_lammps(directory='/blue/subhash/michaelmacisaac/functions/deepmd/model1',d1_dir='1d_gridsearch_2',\
            lammps_script='/blue/subhash/michaelmacisaac/functions/deepmd/model1/in.lattice_constants', ref_len=4.37956,\
                ref_coh=-7.5299,crossval=False)
