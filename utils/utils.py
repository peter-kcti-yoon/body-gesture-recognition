import numpy as np
import os



def unpack_dataset(_X,_y):
    y = []
    for xx,yy in zip(_X,_y):
        y += [yy]* xx.shape[0]

    return _X.reshape(-1,258), y

def normalize(_raw, ch=2):
    # translate
    # scaling

    # body 33*4 x,y,z,v
    # hand 21*3 x,y,z
    _pose, _lh, _rh = split_keypoints(_raw)

    pose = _pose.reshape(-1,4)[:,:ch]
    lh = _lh.reshape(-1,3)[:,:ch]
    rh = _rh.reshape(-1,3)[:,:ch]

    return pose, lh, rh

def translate(raw):
    origin = raw[0] 
    return raw - origin

def scaling(pose, rh, lh=None):
    c1 = pose[12] - pose[11]
    c2 = pose[24] - pose[23]

    s = np.linalg.norm(c1-c2)

    if lh:
        return pose/s, lh/s, rh/s
    else:
        return pose/s, rh/s


def load_dataset(ver, actions):
    data_root = f'./data/data{ver}'

    X, y= [],[]
    for a in actions:
        apath = os.path.join(data_root,a)
        if not os.path.exists(apath):
            continue

        for f in os.listdir(apath):            
            xx = np.load(open(os.path.join(data_root,a,f),'rb'))
            X.append(xx)
            y.append(a)

    return np.array(X), np.array(y)


def load_testset(ver):
    data_root = f'./data/data{ver}'

    X = np.load(open(os.path.join(data_root,'testset1.npy'),'rb'))
    y = np.load(open(os.path.join(data_root,'labels.npy'),'rb'))
    
    return X, y


def get_n_feature(m, c):
    if m=='rh':
        return 21*c
    else: # body
        return 21*c + 33*c


def split_keypoints(_raw):
    """
    _raw: 285
    return 33*4, 21*3, 21*3
    """
    
    raw = np.array(_raw)
    idx = [ 33*4, 21*3, 21*3]
    body = raw[:idx[0]]
    lh = raw[idx[0]: idx[0]+idx[1]]
    rh = raw[idx[0]+idx[1]: ]

    return np.array(body), np.array(lh), np.array(rh)


def normalize_skeleton(_pose,_hand, val):
    pose = _pose.reshape(-1,4)[:,:3]
    hand = _hand.reshape(-1,3)

    body = np.vstack((pose,hand)) # (-1,3)

    return normalize_xyz(body, val)

def normalize_xyz(_hand, val):
    # if val=2 xy norm
    # o.w. val=3 xyz norm
    # hand: np (F,)

    hand = _hand.reshape(-1,3)

    xy =  hand[:,:val] # 21, 2
    origin = hand[0, :val] # 21,1
    xy_norm =  xy- origin

    return xy_norm.ravel()
    