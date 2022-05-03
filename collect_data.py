from multiprocessing.connection import wait
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import argparse

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
SIGN_OKAY      ='Confirm'
SIGN_CANCEL    ='Cancel'
SIGN_POINT     ='Pointing'
SIGN_GRIPP     ='Gripping'
SIGN_HELLO     ='Hello'
actions = [SIGN_CANCEL, SIGN_GRIPP, SIGN_OKAY, SIGN_POINT, SIGN_HELLO]
actions_label = [f'{ts1: >10}' for ts1 in actions]

# actions_label = [ ss for ss in actions]
DATA_ROOT = os.path.join('../data/data1') 

def arg_parser():
    parse = argparse.ArgumentParser(description="Data collection")
    parse.add_argument('-p','--person', required=True, type=int, help='id of person')
    parse.add_argument('-r', '--repeat', default=2, type=int, help='repeating actions')
    parse.add_argument('-s', '--seq', default=30, type=int, help='duration of actions')
    # parse.add_argument('-n', '--num', default=10, type=int, help='number of samples')
    parse.add_argument('-d', '--data', default=1, help='dataset version')
    parse.add_argument('--test', action='store_true', help='data for test')
    
    return parse.parse_args()


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections   

def draw_styled_landmarks(image, results):

    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, lh, rh])

def init_data_dirs(DATA_ROOT):
    for action in actions:
        if not os.path.exists(os.path.join(DATA_ROOT,action)):
            try: 
                print(os.path.join(DATA_ROOT,action))
                os.makedirs(os.path.join(DATA_ROOT,action))
            except:
                print('makedirs exception')
            pass


colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, seq_len, input_frame):
    
    a = (120,200)
    b = (120+ int(400/seq_len*(seq_len-res)),250)
    cv2.rectangle(input_frame, a, b , (117,245,16), -1)
    cv2.putText(input_frame, 'Performing', (a[0],b[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return input_frame

def get_last_sample_id(DATA_ROOT, args):
    
    flist = os.listdir(DATA_ROOT)
    pp = 'p'+str(args.person).zfill(2)
    matching_list = []
    for ff in flist:
        if pp in ff:
            matching_list.append(ff)
    
    if len(matching_list):
        matching_list.sort()
        last = matching_list[-1]
        idx = last.index('s')
        return int(last.split('.')[0][idx+1:])
    else:
        return 0
    


def collect_test(args, DATA_ROOT):
    cap = cv2.VideoCapture(0)
    sample_id = get_last_sample_id(DATA_ROOT, args)
    this_sample_id = sample_id +1

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        kp_list = []
        wait_coutner = 0 # interval between action
        actions_index = 0 # index of order
        stacked_counter = 0 # how long action will take

        target_actions= [ [a]*args.repeat  for a in actions]
        target_actions = np.concatenate(target_actions)

        labels = []
        while 1:
            ret, frame = cap.read()
            if actions_index >= len(target_actions):
                break

            if ret is False:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)
            keypoints = extract_keypoints(results)
            curr_action = target_actions[actions_index]

            if wait_coutner < 50:
                cv2.putText(image, f'{curr_action} in {50 - wait_coutner}', (120,200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                wait_coutner += 1
                kp_list.append(keypoints)
                labels.append('Background')
            
            else:
                if stacked_counter < args.seq:
                    cv2.putText(image, f'{curr_action}', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255, 0), 4, cv2.LINE_AA)
                    kp_list.append(keypoints)
                    labels.append(curr_action)
                    stacked_counter += 1

                else: # done
                    wait_coutner = 0
                    stacked_counter = 0
                    actions_index += 1


            if cv2.waitKey(1) ==ord('q'):
                break
            cv2.imshow('OpenCV Feed', image)
            

        npy_name = f'data_p{str(args.person).zfill(2)}_s{str(this_sample_id).zfill(3)}.npy'
        npy_path = os.path.join(DATA_ROOT, npy_name)
        np.save(open(npy_path, 'wb'), np.array([labels, kp_list], dtype=object))

        cap.release()
        cv2.destroyAllWindows()

if __name__=='__main__':
    
    args = arg_parser()
    DATA_ROOT = f'./data/data{args.data}'
    print('Data root', DATA_ROOT)
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)

    assert args.person > 0, 'Please give the person id'
    collect_test(args, DATA_ROOT)
    
  

