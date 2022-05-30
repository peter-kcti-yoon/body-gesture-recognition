# ANN module
import sys
sys.path.append('..')
import torch
import yaml
import numpy as np
import argparse

from utils.utils import *
from utils.opt import *
import random
import mediapipe as mp
import cv2

from torch.nn import Softmax
import math

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic # Holistic model
colors = [(245,117,16), (117,245,16), (16,117,245),(16,117,245),(16,117,245),
    (245,117,16), (117,245,16), (16,117,245),(16,117,245),(16,117,245),
    (245,117,16), (117,245,16), (16,117,245),(16,117,245),(16,117,245)]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"seed : {seed}")

def prob_viz(image, actions, pred):
    output_frame = image.copy()
    for num, prob in enumerate(pred):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        # print((0,60+num*40), (int(prob*100), 90+num*40))
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
        
    return output_frame
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

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, lh, rh])


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def get_parser():
    parser = argparse.ArgumentParser(description='aaaa')

    ## Training
    parser.add_argument('-e', '--eval', action='store_true')
    parser.add_argument('-a', '--all', action='store_true')
    parser.add_argument('-ep', '--epoch', type=int, default=200)
    parser.add_argument('-p', '--part', type=int, help='index for body part')
    parser.add_argument('--train_batch_size', default=32, type=int, help='index for body part')
    parser.add_argument('--test_batch_size', default=64, type=int, help='index for body part')

    ## Model
    parser.add_argument('--model', default=None)
    parser.add_argument('--model-args',type=dict,default=dict(), help='the arguments of model')
    parser.add_argument('--weights',default=None,help='the weights for network initialization')


    ## data
    parser.add_argument('--translate', type=bool, default=True)
    parser.add_argument('--scale', type=bool, default=True)
    parser.add_argument('--type', type=str, default='body')

    ## Base
    parser.add_argument('--config', type=str, default='./config/hand/mlp2_hand.yaml')
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--actions', default=4, type=int , help='must be train or test')
    return parser


class Processor:
    def __init__(self, args):
        self.args = args
        self.load_model()
        self.run()
    
    def load_model(self):
        Model = import_class(self.args.model)
        print(Model)
        self.model = Model(**self.args.model_args)


    def run(self):
        cap = cv2.VideoCapture()
        cap.open(0)
        checkpoint = torch.load('./weights/mlp2_lr0.0001_ep200/model_weights.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        with torch.no_grad():
            self.model.eval()
        softmax = Softmax()
        actions = actions_dict[self.args.actions]

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while 1:
                ret, frame = cap.read()
                if ret is False:
                    break
                image, results = mediapipe_detection(frame, holistic)
                _kps = extract_keypoints(results)
                body, _, rh = split_keypoints(_kps)
                #####################################

                kp5 = rh.reshape(-1,3)[5]
                # x = kp5[0]
                # y = kp5[1]
                x = kp5[0]*frame.shape[1]
                y = kp5[1]*frame.shape[0]
                z= kp5[2]

                kp0= rh.reshape(-1,3)[0][:2]
                kp5 =rh.reshape(-1,3)[5][:2]
                kp0[0] *=frame.shape[1]
                kp0[1] *=frame.shape[0]
                kp5[0] *=frame.shape[1]
                kp5[1] *=frame.shape[0]
                dist = np.linalg.norm(kp0-kp5)
                

                cv2.putText(image, f'{int(x)},{int(y)},{z:.2f}', (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(image, f'{dist:.4f}', (300, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                # cv2.putText(image, f'{rh[5]}', (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)


                ####################################

                if self.args.translate:
                    body, rh = translate(body), translate(rh)
                if self.args.scale:
                    body, rh  = scaling(body,rh)
                
                if self.args.type =='body':
                    data = np.concatenate([body, rh ])
                else:
                    data = rh
                data =np.expand_dims(data,axis=0)
                x = torch.from_numpy(data).float()
                output = self.model(x)[0]
                # print([f'{int(v*100)}' for v in output.cpu().detach().numpy()])
                pred = softmax(output).cpu().detach().numpy()
                pred = [ 0 if math.isnan(t) else t for t in pred]
                
                # print([f'{v:.2f}' for v in tmp], sum(tmp))    
                # print([v for v in tmp])    
                # pred = pred.cpu().detach().numpy()
                # print(pred.cpu().detach().numpy())
                draw_landmarks(image, results)
                image = prob_viz(image, actions, pred)                
                
                if cv2.waitKey(1) ==ord('q'):
                    quit()
                cv2.imshow('OpenCV Feed', image)
                
            cap.release()
            cv2.destroyAllWindows()
    



if __name__=='__main__':

    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    set_seed(1)
    process = Processor(arg)
    

