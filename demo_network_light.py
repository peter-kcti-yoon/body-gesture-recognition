# ANN module
import sys
sys.path.append('..')
import torch
import yaml
import numpy as np
import argparse
import socket
import json
from utils.utils import *
from utils.opt import *
import random
import mediapipe as mp
import cv2

from torch.nn import Softmax
import math
import pickle

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
    for num in actions.keys():
        if num == pred:
            cv2.rectangle(output_frame, (0,60+num*40), (130, 90+num*40), colors[num], -1)
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
    # parser.add_argument('-e', '--eval',  help='evaluation mode')
    parser.add_argument('-ep', '--epoch', type=int, default=200)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train_batch_size', default=32, type=int, help='index for body part')
    parser.add_argument('--test_batch_size', default=64, type=int, help='index for body part')

    ## Eval
    parser.add_argument('--no_eval', default=False, type=bool, help='Only training w/o eval')
    parser.add_argument('--test', action='store_true',default=False, help='testing mode')


    ## Model
    parser.add_argument('--model', default=None)
    parser.add_argument('--model-args',type=dict,default=dict(), help='the arguments of model')
    parser.add_argument('--weights',default=None,help='the weights for network initialization')


    ## data
    parser.add_argument('--translate', type=bool, default=False)
    parser.add_argument('--scale', type=bool, default=False)
    parser.add_argument('--onlyxy', type=bool, default=False)
    parser.add_argument('--vector', type=bool, default=False)
    parser.add_argument('--type', type=str, default='body')

    ## Basework_dir: './work_dir/trip2'
    parser.add_argument('--config', type=str, default='./config/trip2_hand_data1.yaml')
    parser.add_argument('--work_dir', type=str, default='./work_dir/trip2')
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--actions', default=4, type=int , help='must be train or test')
    return parser


class Processor:
    def __init__(self, args):
        self.args = args


        ###### Initialize the server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('', 9999))
        self.server_socket.listen()

        ###

        self.load_model()
        self.run()
    
    def load_model(self):
        Model = import_class(self.args.model)
        print(Model)
        self.model = Model(**self.args.model_args)

    def transform(self, data):
        body, _, rh = split_keypoints(data)
        body = body.reshape(-1,4)
        rh  = rh.reshape(-1, 3)

        if self.args.onlyxy:
            body,rh =  body[:,:2], rh[:, :2]
        if self.args.translate:
            body, rh = translate(body), translate(rh)
        if self.args.scale:
            rh  = scaling(rh)
        if self.args.vector:
            body,rh = vectorform(body),vectorform(rh)

        
        data = rh.ravel()
        return torch.from_numpy(data).float()

    def run(self):
        cap = cv2.VideoCapture()
        cap.open(0)
        w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video_name = f'results.avi'
        video_path = os.path.join('.', video_name)        
        ## For linux
        # self.weight_path = os.path.join('./weights/',self.args.config.split('/')[2])
        # For Windows
        self.weight_path = os.path.join('./weights/trip2_hand_data1')
        checkpoint = torch.load(f'{self.weight_path}/model_weights.pt')
        # centroids = torch.load(f'{self.weight_path}/centroids.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        kmeans = pickle.load(open(f'{self.weight_path}/kmeans.pkl', 'rb'))
        writer = cv2.VideoWriter(video_path, fourcc, 15, (w, h))

        labels = {}
        with open(f'{self.weight_path}/labels.txt', 'r') as f:
            for pair in f.readlines():
                name, idx = pair.strip().split(' ')
                labels[int(idx)] = name
        print(labels)
        send_data = {}
        with torch.no_grad():
            self.model.eval()
        # softmax = Softmax()
        print('Waiting for client')
        while 1:
            server_socket, addr = self.server_socket.accept()
            send_data['Gesture'] = 'None'
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while 1:
                    ret, frame = cap.read()
                    if ret is False:
                        break
                    image, results = mediapipe_detection(frame, holistic)
                    _kps = extract_keypoints(results)
                    # print(_kps)
                    x = self.transform(_kps)

                    #####################################
                    # cv2.putText(image, f'{rh[5]}', (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)

                    ####################################

                    x = torch.unsqueeze(x, 0)
                    output = self.model(x)[0]
                    # print([f'{int(v*100)}' for v in output.cpu().detach().numpy()])
                    output = torch.nan_to_num(output, nan=99)
                    output = output.detach().numpy()
                    # print(pread)
                    dist = kmeans.transform(output.reshape(-1,2))[0]

                    min_idx = np.argmin(dist)

                    if dist[min_idx] < 0.5:
                        # print(dist[min_idx], labels[min_idx])
                        image = prob_viz(image, labels, min_idx)
                        send_data['Gesture'] = labels[min_idx]
                    else:
                        image = prob_viz(image, labels, -1)
                        send_data['Gesture'] = 'None'

                    # print([f'{v:.2f}' for v in tmp], sum(tmp))
                    # print([v for v in tmp])
                    # pred = pred.cpu().detach().numpy()
                    # print(pred.cpu().detach().numpy())
                    draw_landmarks(image, results)

                    json_data = json.dumps(send_data)
                    msg = json_data.encode('utf-8')
                    print(msg)

                    json_data = json.dumps(send_data)
                    msg = json_data.encode('utf-8')
                    print(msg)

                    ###### Send the encoded message to the Unity client
                    try:
                        server_socket.send(len(msg).to_bytes(4, 'little'))
                        server_socket.recv(1)
                        server_socket.send(msg)
                        server_socket.recv(1)
                    except Exception as e:
                        # print('exception:', e)
                        server_socket.shutdown(socket.SHUT_RDWR)
                        server_socket.close()
                        break

                    if cv2.waitKey(1) ==ord('q'):
                        quit()
                    cv2.imshow('OpenCV Feed', cv2.resize(image,(1920,1080)))
                    writer.write(image)
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
                assert (k in key) , k
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    set_seed(1)
    process = Processor(arg)
    

