import numpy as np
import os
def convert_old_data():
    SIGN_BACKGROUND ='background'
    SIGN_OKAY      ='Confirm'
    SIGN_CANCEL    ='Cancel'
    SIGN_POINT     ='Pointing'
    SIGN_GRIPP     ='Gripping'
    SIGN_HELLO     ='Hello'
    SIGN_TWO       ='Two'
    SIGN_BEST      ='Best'
    SIGN_LEFT      ='Left'
    SIGN_RIGHT     ='Right'

    ######################################
    ## Do not fix the order
    label_list = [SIGN_BACKGROUND, SIGN_OKAY, SIGN_CANCEL, SIGN_POINT,SIGN_GRIPP,
                SIGN_HELLO,SIGN_TWO, SIGN_BEST,SIGN_LEFT,SIGN_RIGHT]

    label2idx = { label:idx for idx,label in enumerate(label_list)}
    idx2label={idx:label2idx[idx] for idx in label2idx.keys()}
    npy_list_list = []
    for root,_,flist in os.walk('./data/data1'):
        for ff in flist:
            label_str = ff.split('_')[0]
            npy = np.load(os.path.join(root,ff), allow_pickle=True)
            labels = [label2idx[label_str]]*npy.shape[0]
            npy = np.append(npy, np.array(labels).reshape(-1,1), axis=1)        
            npy_list_list.append(npy)
    
    np.save(open('./data/data3/data_p01_s001.npy', 'wb'), 
                 np.vstack(npy_list_list))


def unpack_dataset(_X,_y):
    y = []
    for xx,yy in zip(_X,_y):
        y += [yy]* xx.shape[0]

    return _X.reshape(-1,258), y

def vectorform(self, _data):
    pass


def translate(raw):
    """
    shape shold be  (-1,n)
    """
    assert len(raw.shape) ==2, "shape shold be  (-1,n)"
    origin = raw[0] 
    return raw - origin

def scaling(_data, hand = True):
    
    p1 = _data[0]
    p2 = _data[5]
    s = np.linalg.norm(p1-p2)
    return _data/s
    

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
    _raw: 258
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
 
    
    # print(npy_list)
if __name__=='__main__':
    convert_old_data()