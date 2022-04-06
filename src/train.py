from model import Gesturer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from opt import actions
import numpy as np
import os


## POSE 

"""
POSE
x and y: Landmark coordinates normalized to [0.0, 1.0] by the image width and height
z: Represents the landmark depth with the depth at the midpoint of hips being the origin, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x.
visibility: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.


"""

def right_hand_only_bk(_raw):
    i = 33*4 + 468*3  + 21*3
    rh = _raw[:, i:] # TIMEWINDOW, 21*3
    rh = rh.reshape(30,21,3) # TIMEWINDOW, 21, 3
    xy =  rh[:,:,:2] # T, 21, 2
    vis = np.expand_dims(rh[:,:,2], axis=2)
    xy_norm = [ xyr[:,:]-xyr[0,:] for xyr in xy]
    xy_norm = np.array(xy_norm)
    rh = np.concatenate((xy_norm, vis),axis=2)
    
    return rh.reshape(30,-1)
    

def right_hand_only(_raw):
    i = 33*4 + 468*3  + 21*3
    rh = _raw[:, i:] # TIMEWINDOW, 21*3
    return rh.reshape(30,-1) # TIMEWINDOW, 21, 3
    



def train():
    DATA_ROOT_PATH ='../data'
    gesture = Gesturer()
    model = gesture.build()
    label_map = {label:num for num, label in enumerate(actions)}
    _X, _y = [], []
    person_dirs = ['data_dy','data_ys','data_yh']

    for dd in person_dirs:
        for action in actions:
            for f in os.listdir(os.path.join(DATA_ROOT_PATH,dd,action)):

                #### TODO normalize skeletons !!!!!!!
                tmp = np.load(open(os.path.join(DATA_ROOT_PATH,dd,action,f),'rb'))
                # tmp = right_hand_only(tmp)
                _X.append(tmp)
                _y.append(label_map[action])



    X = np.array(_X)
    y = to_categorical(_y).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # print(label_map)
    # print(X_train.shape, X_test.shape, len(y_train), len(y_test))

    # unique, counts = np.unique(y_train, return_counts=True)
    # print(unique, counts)
    # unique, counts = np.unique(y_test, return_counts=True)
    # print(unique, counts)
    # quit()
    model.fit(X_train, y_train, epochs=2000, callbacks=[gesture.tb_callback])
    model.save('../weights/action.h5')


if __name__ == '__main__':
    train()




