from model import Gesturer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from opt import arg_parser, actions_dict
import numpy as np
import os
from uilts import *

## POSE 

"""
POSE
x and y: Landmark coordinates normalized to [0.0, 1.0] by the image width and height
z: Represents the landmark depth with the depth at the midpoint of hips being the origin, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x.
visibility: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.


"""

def get_n_feature(m, c):
    if m=='rh':
        return 21*c
    else: # body
        return 21*c + 33*c


def normalize(args, tmp):
    res = []
    for t in tmp:
        pose, lh, rh = split_keypoints(t)
        
        if args.mode == 'rh':
            dd = normalize_xyz(rh, args.channels)
        else: # body
            dd = normalize_skeleton(pose,rh, args.channels)
        res.append(dd)
    return np.array(res)



def train():
    args = arg_parser()
    DATA_ROOT_PATH = os.path.join('../data',args.dataset)
    actions = actions_dict[args.actions]
    gesture = Gesturer(len(actions), get_n_feature(args.mode, args.channels))
    model = gesture.build()
    label_map = {label:num for num, label in enumerate(actions)}
    _X, _y = [], []


    for action in actions:
        for f in os.listdir(os.path.join(DATA_ROOT_PATH,action)):

            tmp = np.load(open(os.path.join(DATA_ROOT_PATH,action,f),'rb'))
            tmp = normalize(args, tmp)
            _X.append(tmp)
            _y.append(label_map[action])



    X = np.array(_X)
    y = to_categorical(_y).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model.fit(X_train, y_train, epochs=args.epoch, callbacks=[gesture.tb_callback])
    model.save('../weights/action.h5')


if __name__ == '__main__':
    train()




