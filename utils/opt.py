# actions = ['gripping', 'open-hand', 'okidoki', 'handsuping','handuped','walking', 'pointing', 'cancel','fyou'] # action class

## HAND-ONLY
# actions = ['gripping', 'open-hand', 'okidoki', 'pointing', 'fyou'] # action class

# actions = ['gripping', 'open-hand', 'okidoki', 'pointing'] # action class


import argparse
SIGN_BACKGROUND ='background'
SIGN_OKAY      ='Confirm'
SIGN_CANCEL    ='Cancel'
SIGN_POINT     ='Pointing'
SIGN_GRIPP     ='Gripping'
SIGN_HELLO     ='Hello'
SIGN_TWO       ='Two'
SIGN_BEST      ='Best'
SIGN_LEFT      ='left'
SIGN_RIGHT     ='right'

label_list = [SIGN_BACKGROUND, SIGN_OKAY, SIGN_CANCEL, SIGN_POINT,SIGN_GRIPP,
             SIGN_HELLO,SIGN_TWO, SIGN_BEST,SIGN_LEFT,SIGN_RIGHT]

label2idx = { label:idx for idx,label in enumerate(label_list)}
idx2label={idx:label2idx[idx] for idx in label2idx.keys()}

actions_dict={
     1: [SIGN_CANCEL, SIGN_GRIPP, SIGN_OKAY, SIGN_POINT, SIGN_HELLO]
    ,2: [SIGN_CANCEL, SIGN_GRIPP, SIGN_OKAY, SIGN_HELLO, SIGN_TWO, SIGN_BEST]
    ,3: [SIGN_CANCEL, SIGN_GRIPP, SIGN_OKAY, SIGN_HELLO]
    ,4: [SIGN_CANCEL, SIGN_GRIPP, SIGN_OKAY, SIGN_HELLO, SIGN_BEST]
    ,5: [SIGN_CANCEL, SIGN_GRIPP, SIGN_OKAY, SIGN_BEST] #4
    ,5: [SIGN_GRIPP, SIGN_OKAY, SIGN_HELLO,SIGN_TWO, SIGN_BEST] #5

    ,42: [SIGN_CANCEL, SIGN_GRIPP, SIGN_OKAY, SIGN_HELLO, SIGN_TWO, SIGN_BEST]
}