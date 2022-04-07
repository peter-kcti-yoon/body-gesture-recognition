# actions = ['gripping', 'open-hand', 'okidoki', 'handsuping','handuped','walking', 'pointing', 'cancel','fyou'] # action class

## HAND-ONLY
# actions = ['gripping', 'open-hand', 'okidoki', 'pointing', 'fyou'] # action class

# actions = ['gripping', 'open-hand', 'okidoki', 'pointing'] # action class

import argparse

actions_dict={
    0: ['gripping', 'open-hand', 'okidoki', 'pointing'],
    1: ['gripping', 'open-hand', 'okidoki', 'pointing', 'fyou'],
    2: ['grip', 'okay', 'cancel', 'point']
    ,3: ['gripping', 'okidoki', 'open-hand']
}

def arg_parser():
    parse = argparse.ArgumentParser(description='Gesture Recognition')

    ## Training
    parse.add_argument('-e', '--epoch', default=500, type=int , help='Trainiing Epochs')
    parse.add_argument('-d', '--dataset', default=0 ,type=str, help='Index for dataset')
    parse.add_argument('-a', '--actions', default=2 , type=int, help='Index for action classes' )
    parse.add_argument('-c', '--channels', default=2 , type=int, help='Input data channels e.g.,) xyz=3, xy=2' )
    parse.add_argument('-m', '--mode', default='hand' , type=str, help='Body part to use' ,
    choices=['rh','body'])



    return parse.parse_args()