
import numpy as np
SIGN_OKAY      ='Confirm'
SIGN_CANCEL    ='Cancel'
SIGN_POINT     ='Pointing'
SIGN_GRIPP     ='Gripping'
SIGN_HELLO     ='Hello'
actions = [SIGN_CANCEL, SIGN_GRIPP, SIGN_OKAY, SIGN_POINT, SIGN_HELLO]


for ts1 in actions:
    # print(type(ts1))
    print(f'{ts1: >15}')
# actions_label = [ f'{ts1: <15 }' for ts1 in actions]