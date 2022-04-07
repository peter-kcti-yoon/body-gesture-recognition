import numpy as np





def split_keypoints(_raw):
    raw = np.array(_raw)
    idx = [ 33*4, 468*3, 21*3, 21*3]
    pose = raw[:idx[0]]
    lh = raw[idx[0]+idx[1]: idx[0]+idx[1]+idx[2]]
    rh = raw[idx[0]+idx[1]+idx[2]: ]

    return pose, lh, rh


def normalize_skeleton(_pose,_hand, val):
    pose = _pose.reshape(-1,4)[:,:3]
    hand = _hand.reshape(-1,3)

    body = np.vstack((pose,hand)) # (-1,3)

    return normalize_xyz(body, val)

def normalize_xyz(hand, val):
    # if val=2 xy norm
    # o.w. val=3 xyz norm
    # hand.shape is (-1, kp)

    xy =  hand[:,:val] # 21, 2
    origin = hand[:, val] # 21,1
    xy_norm =  xy- origin

    return xy_norm.ravel()
    