import numpy as np
import json

def read(path):
    feature=[]
    IDS=[]
    try:
        with open(path,'r') as f:
            facedict=json.load(f)
        for key in facedict:
            feature.append(facedict[key])
            IDS.append(key)
        
    except:
        print('please give person features')

    return np.array(feature), IDS