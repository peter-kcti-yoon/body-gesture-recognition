import os
import numpy as np
import pdb
def read_data():
    path = './data/data1'
    npy_list = []
    for root,_,flist in os.walk(path):
        for ff in flist:
            if ff.endswith('npy'):
                npy = np.load(os.path.join(root,ff),allow_pickle=True)
                npy_list.append(npy)
    return npy_list
    # return np.vstack(npy_list)

a = read_data()
pdb.set_trace()