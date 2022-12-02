#%%
import json 
from functools import reduce
import operator

# %%
params = {
    'model fitting_net neuron':[[15,150,150],[10,10,10],[100,100]],
    'model descriptor type':['se_e2_a','se_e2_r']
    }

# %%
with open('/Users/kimiaghaffari/Desktop/Research/B4C_ML_Potential/training/input_scripts/input_029.json') as f:
    input = json.load(f)

# %%
# %%

import json 
from functools import reduce
import operator



def getFromDict(dataDict,mapList):
    return reduce(operator.getitem,mapList,dataDict)

def setInDict(dataDict,mapList,value):
    getFromDict(dataDict,mapList[:-1])[mapList[-1]] = value
# %%
for key in params.keys():
    key_list = key.split(' ')
    # try:
    #     print(getFromDict(input,key_list))
    # except:
    #     raise ValueError(f'key {key} in parameter list not valid')
    print(f'{key}\n')
    for value in params[key]:
        print(f'{value}\n')
        setInDict(input,key_list,value)
        print(f'{input}\n')
    #change value back to original 
    # print(getFromDict(input,key_list))


