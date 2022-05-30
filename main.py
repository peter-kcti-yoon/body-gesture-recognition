# ANN module
import sys
sys.path.append('..')
import torch
from torch import nn, optim                        
from torch.utils.data import DataLoader, Dataset   
import torch.nn.functional as F                    
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split
# Loss
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from utils.utils import *
from utils.opt import *
from sklearn.metrics import accuracy_score,classification_report
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


N_CLASSES = 'n_classes'
TEST_ACTIONS = 'test_actions'
TRAIN_ACTIONS='train_actions'
ACTIONS='actions'
ALL = 'all'
BODY ='body'
RH ='right_hand'
LH ='left_hand'
seed= 121

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

def _read_data():
    path = './data/data3'
    npy_list, label_list = [], []
    for root,_,flist in os.walk(path):
        for ff in flist:
            if ff.endswith('npy'):
                npy = np.load(os.path.join(root,ff),allow_pickle=True)
                npy_list.extend(npy[:,:-1])
                label_list.extend(npy[:, -1])

    label_npy = np.vstack(label_list).squeeze().astype(np.int64)
    
    return np.vstack(npy_list).squeeze(), label_npy

class TensorData(Dataset):

    def __init__(self, args, X, y):
        self.args = args
        self.x_data, self.y_data = X, y
        self.len = self.x_data.shape[0]


    def train_data(self, actions):
        x_data, y_data = load_dataset(1, actions)
        x_data, y_data = unpack_dataset(x_data, y_data)
        y_data = np.array([ actions.index(a) for a in y_data])
        return x_data, y_data

    def test_data(self,actions):
        x_data, y_data= load_testset(1)

        masking = []
        for i,la in enumerate(y_data):
            if la in actions:
                masking.append(i)

        masking = np.array(masking)
        x_data = x_data[masking]
        y_data = y_data[masking]
        y_data = np.array([ actions.index(a) for a in y_data])

        return x_data, y_data

    def process_data(self, _data):
        body, _, rh = split_keypoints(_data)

        if self.args.translate:
            body, rh = translate(body), translate(rh)
        if self.args.scale:
            body, rh  = scaling(body,rh)
        
        if self.args.type =='body':
            data = np.concatenate([body, rh ])
        else:
            data = rh

        return data

    def vectorform(self, _data):

        pass

    def __getitem__(self, index):
        data = self.x_data[index]
        data = self.process_data(data)

        
        x = torch.from_numpy(data).float()
        
        # y = np.array(label2idx[self.y_data[index]])
        y = np.array(self.y_data[index])
        # print(type(self.y_data[index]))
        # print(self.y_data[index])
        # quit()
        y = torch.from_numpy(y).long() 
        # y = torch.from_numpy(self.y_data[index]).long() 
        return x,y 

    def __len__(self):
        return self.len
    
class Regressor(nn.Module):
    def __init__(self, num_class, num_feature):
        super().__init__() # 모델 연산 정의
        self.fc1 = nn.Linear(num_feature, 330, bias=True) 
        self.fc2 = nn.Linear(330, 160, bias=True) 
        self.fc3 = nn.Linear(160, 80, bias=True) 
        self.fc4 = nn.Linear(80, 30, bias=True)         
        self.fc = nn.Linear(30, num_class, bias=True) 
        
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.xavier_uniform_(self.fc.weight)

        self.relu = torch.nn.ReLU()
        self.bn1 = nn.BatchNorm1d(330)
        self.bn2 = nn.BatchNorm1d(160)
        self.bn3 = nn.BatchNorm1d(80)
        self.bn4 = nn.BatchNorm1d(30)
        self.dropout = nn.Dropout(0.1) 

        self.model = nn.Sequential(self.fc1, self.bn1, self.relu,
                                   self.fc2, self.bn2, self.relu, self.dropout,
                                   self.fc3, self.bn3, self.relu, self.dropout,
                                   self.fc4 ,self.bn4, self.relu, self.dropout,
                                   self.fc)       

    def forward(self, x): # 모델 연산의 순서를 정의
        return self.model(x)


def print_results(pred, label, args):
    t1 = np.argmax(pred, axis=1)
    t2 = label
    acc = np.mean((t1==t2).astype(int))
    print(f'mAP: {acc*100:.2f}%')
    actions_list = actions_dict[args.actions]

    for i, a in enumerate(actions_list):
        indices = np.where(t2 ==i)
        pred_mask = t1[indices]
        label_mask = t2[indices]
        acc = np.mean((pred_mask==label_mask).astype(int))
        print(f'   {a:_>10} (#{pred_mask.shape[0]}): {acc:.2f}%')

def import_class(name):
    """
    dont forget to update __init__.py 
    """
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
    parser.add_argument('--test', default=False, type=bool, help='testing mode')


    ## Model
    parser.add_argument('--model', default=None)
    parser.add_argument('--model-args',type=dict,default=dict(), help='the arguments of model')
    parser.add_argument('--weights',default=None,help='the weights for network initialization')


    ## data
    parser.add_argument('--translate', type=bool, default=True)
    parser.add_argument('--scale', type=bool, default=True)
    parser.add_argument('--vector', type=bool, default=False)
    parser.add_argument('--type', type=str, default='body')

    ## Base
    parser.add_argument('--config', type=str, default='./config/hand/mlp1_hand.yaml')
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--actions', default=4, type=int , help='must be train or test')
    return parser

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Processor:
    def __init__(self, args):
        self.args = args
        model_name = args.model.split('.')[1]
        model_name =f'{model_name}_lr{args.learning_rate}_ep{args.epoch}'

        self.init_dirs(model_name)
        self.load_model()
                
        self.train_writer = SummaryWriter(f'./logs/{model_name}/train', 'train')
        self.valid_writer = SummaryWriter(f'./logs/{model_name}/valid', 'valid')
        self.avg = AverageMeter()
        
        if args.test:
            self.test()
        else:
            self.load_data()
            self.train()


    def init_dirs(self,model_name):
        self.weight_path = os.path.join('./weights',model_name)
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path)
        
        log_path = os.path.join('./logs',model_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)


    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path)
        with open('{}/config.yaml'.format(self.weight_path), 'w') as f:
            yaml.dump(arg_dict, f)


    def load_data(self):
        X, y= _read_data()
        print('Load data', X.shape, y.shape)
        trainx, valx , trainy, valy = train_test_split(X,y, test_size= 0.2, stratify=y,random_state=seed)

        trainsets = TensorData(self.args,  trainx, trainy)
        self.trainloader = DataLoader(trainsets, batch_size=self.args.train_batch_size, shuffle=True)
        
        validset = TensorData(self.args, valx, valy)
        self.validloader = DataLoader(validset, batch_size=self.args.test_batch_size, shuffle=False)        

        # TODO make testset loader
        # test_set = TensorData(self.args, valx, valy)
        # self.testloader = DataLoader(validset, batch_size=self.args.test_batch_size, shuffle=False)        

    def test(self):
        predictions = torch.tensor([], dtype=torch.float) # 예측값을 저장하는 텐서.
        actual = torch.tensor([], dtype=torch.float) # 실제값을 저장하는 텐서.
        target_names = []

        with torch.no_grad():
            self.model.eval() # 평가를 할 땐 반드시 eval()을 사용해야 한다.

        for data in self.validloader:
            inputs, values = data
            outputs = self.model(inputs)

            _target = actual.detach().numpy() # 넘파이 배열로 변경.
            target_names = [idx2label(t) for t in _target]

            predictions = torch.cat((predictions, outputs), 0) # cat함수를 통해 예측값을 누적.
            actual = torch.cat((actual, values), 0) # cat함수를 통해 실제값을 누적.

    def eval(self):
        
        with torch.no_grad():
            self.model.eval() # 평가를 할 땐 반드시 eval()을 사용해야 한다.

        for data in self.validloader:
            inputs, values = data
            outputs = self.model(inputs)
            outputs = self.softmax(outputs)
            outputs = torch.argmax(outputs, dim=1)
            # print(outputs.detach().numpy(),values.detach().numpy())

            acc = accuracy_score(outputs.detach().numpy(),values.detach().numpy())
            self.avg.update(acc , values.shape[0])


        
    
    def load_model(self):
        Model = import_class(self.args.model)
        print(Model)
        self.model = Model(**self.args.model_args)
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    def train(self):
        self.optimizer = optim.Adam(self.model.parameters(), 
                        lr=self.args.learning_rate, weight_decay=1e-8)

        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        best_epoch_acc = 0
        self.global_step =0
        name_desc = tqdm(range(self.args.epoch))
        for epoch in name_desc:
            running_loss = 0.0 # 한 에폭이 돌 때 그안에서 배치마다 loss가 나온다. 즉 한번 학습할 때 그렇게 쪼개지면서 loss가 다 나오니 MSE를 구하기 위해서 사용한다.
            for i, data in enumerate(self.trainloader, 0): # 무작위로 섞인 32개의 데이터가 담긴 배치가 하나씩 들어온다.
                self.global_step += 1
                inputs, values = data # data에는 X, Y가 들어있다.
                self.optimizer.zero_grad() # 최적화 초기화.

                outputs = self.model(inputs) # 모델에 입력값을 넣어 예측값을 산출한다.
                loss = self.loss(outputs, values) # 손실함수를 계산. error 계산.
                loss.backward() # 손실 함수를 기준으로 역전파를 설정한다.
                self.optimizer.step() # 역전파를 진행하고 가중치를 업데이트한다.

                running_loss += loss.item() # epoch 마다 평균 loss를 계산하기 위해 배치 loss를 더한다.
            
                
                if i %10 == 0:
                    
                    outputs = torch.argmax(self.softmax(outputs), dim=1)
                    train_acc = torch.mean((values==outputs).float())
                    self.eval()
                    self.train_writer.add_scalar('Loss', loss, self.global_step)
                    self.train_writer.add_scalar('Accuracy', train_acc, self.global_step)
                    self.valid_writer.add_scalar('Accuracy', self.avg.avg, self.global_step)
                        
                    msg = f'Epoch: {str(epoch).zfill(3)}, Step:{self.global_step}, Acc: {self.avg.avg:.4f}' 
                    name_desc.set_description(msg)
                    
            self.save_weight(epoch, False)
            if best_epoch_acc < self.avg.avg:
                best_epoch_acc = self.avg.avg
                self.save_weight(epoch, True)



        print('Model Best', best_epoch_acc)  
    def save_weight(self,epoch, is_best=False):
        if is_best:
            model_name = 'model_best'
        else:
            model_name = 'model_weights'

        torch.save({
                    'epoch': epoch,
                    'global_step':self.global_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.loss,
                    }, f'{self.weight_path}/{model_name}.pt')



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
    

