from multiprocessing.connection import wait
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('../data') 

############################################
actions = np.array(['hello', 'thanks', 'iloveyou']) # action class
sequence_lengths = [40, 60, 70] # duration of each action
no_sequences = 2 # number of samples
############################################




def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections   

def draw_styled_landmarks(image, results):
    # Draw face connections
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
    #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
    #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    #                          ) 
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
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, face, lh, rh])

def init_data_dirs():
    DATA_ROOT = '../data'
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

init_data_dirs()


cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    for action, seq_len in zip(actions,sequence_lengths):
        # Loop through sequences aka videos
        for sequence in range(no_sequences):

            wait_count = 0
            wait_delay = 50            
            kp_list = []
            while 1:
                start_ckpt = time.time()            
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)
                print(f'Took: {(time.time() - start_ckpt):.2f}s', end='\r')

                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                if wait_count < wait_delay :
                    cv2.putText(image, f'STARTING COLLECTION in {wait_delay-wait_count}', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)

                    wait_count += 1
                    if cv2.waitKey(1) ==ord('q'):
                        quit()
                    cv2.imshow('OpenCV Feed', image)

                else:
                    keypoints = extract_keypoints(results)
                    kp_list.append(keypoints)
                    image = prob_viz(len(kp_list), seq_len,image)
                    if cv2.waitKey(1) ==ord('q'):
                        quit()
                    cv2.imshow('OpenCV Feed', image)

                    if seq_len < len(kp_list):
                        npy_path = os.path.join(DATA_PATH,action, f'{action}_{str(sequence).zfill(3)}')
                        np.save(npy_path, np.vstack(kp_list))
                        break


                    
    cap.release()
    cv2.destroyAllWindows()
  