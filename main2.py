# ANN module
import sys
sys.path.append('..')
import torch
from torch import nn, optim                        
from torch.utils.data import DataLoader, Dataset   
import torch.nn.functional as F                    
from tqdm import tqdm
import yaml
# Loss
import matplotlib.pyplot as plt
import numpy as np
import argparse

from src.tools import *
from src.opt import *
import random

N_CLASSES = 'n_classes'
TEST_ACTIONS = 'test_actions'
TRAIN_ACTIONS='train_actions'
ACTIONS='actions'
ALL = 'all'
BODY ='body'
RH ='right_hand'
LH ='left_hand'


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


class TensorData(Dataset):

    def __init__(self, args):
        actions = actions_dict[args.actions]
        self.args = args
        if args.phase == 'train':
            self.x_data, self.y_data = self.train_data(actions)
        else:
            self.x_data, self.y_data = self.test_data(actions)
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

        data = np.concatenate([body, rh ])
        return data




    def __getitem__(self, index):

        data = self.x_data[index]
        data = self.process_data(data)

        # print(type(data), data.shape, data[0])
        x = torch.from_numpy(data).float()
        y = torch.from_numpy(np.array(self.y_data[index])).long() 
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



def train(configs, args):
    criterion = nn.CrossEntropyLoss()
    
    trainsets = TensorData(args.part)
    model = Regressor(configs[N_CLASSES], trainsets.n_feature)
    trainloader = torch.utils.data.DataLoader(trainsets, batch_size=32, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-8)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    loss_ = [] # loss를 저장할 리스트.
    n = len(trainloader)
    name_desc = tqdm(range(args.epoch))
    for epoch in name_desc:
        running_loss = 0.0 # 한 에폭이 돌 때 그안에서 배치마다 loss가 나온다. 즉 한번 학습할 때 그렇게 쪼개지면서 loss가 다 나오니 MSE를 구하기 위해서 사용한다.
        for i, data in enumerate(trainloader, 0): # 무작위로 섞인 32개의 데이터가 담긴 배치가 하나씩 들어온다.
            
            inputs, values = data # data에는 X, Y가 들어있다.
            optimizer.zero_grad() # 최적화 초기화.

            outputs = model(inputs) # 모델에 입력값을 넣어 예측값을 산출한다.
            loss = criterion(outputs, values) # 손실함수를 계산. error 계산.
            loss.backward() # 손실 함수를 기준으로 역전파를 설정한다.
            optimizer.step() # 역전파를 진행하고 가중치를 업데이트한다.

            running_loss += loss.item() # epoch 마다 평균 loss를 계산하기 위해 배치 loss를 더한다.
        
        loss_.append(running_loss/n) # MSE(Mean Squared Error) 계산
    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 'checkpoint.pt')
    

    evaluation(configs, args)
    plt.plot(loss_)
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.show()

def evaluation(configs, args):
    
    testset = TensorData(configs[ACTIONS],configs[N_CLASSES], args.part, is_train=False)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    
    model = Regressor(configs[N_CLASSES],testset.n_feature)
    checkpoint = torch.load('checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    predictions = torch.tensor([], dtype=torch.float) # 예측값을 저장하는 텐서.
    actual = torch.tensor([], dtype=torch.float) # 실제값을 저장하는 텐서.

    with torch.no_grad():
        model.eval() # 평가를 할 땐 반드시 eval()을 사용해야 한다.

    for data in dataloader:
        inputs, values = data
        outputs = model(inputs)

        predictions = torch.cat((predictions, outputs), 0) # cat함수를 통해 예측값을 누적.
        actual = torch.cat((actual, values), 0) # cat함수를 통해 실제값을 누적.

    predictions = predictions.detach().numpy() # 넘파이 배열로 변경.
    actual = actual.detach().numpy() # 넘파이 배열로 변경.
    print_results(predictions, actual, configs[ACTIONS])



def print_results(pred, label, actions):
    t1 = np.argmax(pred, axis=1)
    t2 = label
    acc = np.mean((t1==t2).astype(int))
    print(f'mAP: {acc*100:.2f}%')

    for i, a in enumerate(actions):
        indices = np.where(t2 ==i)
        pred_mask = t1[indices]
        label_mask = t2[indices]
        acc = np.mean((pred_mask==label_mask).astype(int))
        print(f'\t {a:_>10}: {acc:.2f}%, #Samples: {pred_mask.shape}')

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

    ## Model
    parser.add_argument('--model', default=None)
    parser.add_argument('--model-args',type=dict,default=dict(), help='the arguments of model')
    parser.add_argument('--weights',default=None,help='the weights for network initialization')


    ## data
    parser.add_argument('--translate', type=bool, default=True)
    parser.add_argument('--scale', type=bool, default=True)

    ## Base
    parser.add_argument('--config', type=str, default='./configs/train_small_body.yaml')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--actions', default=4, type=int , help='must be train or test')
    return parser


class Processor:
    def __init__(self, args):
        self.args = args
        self.load_model()

        if args.phase =='train':
            self.train()



    def load_model(self):
        Model = import_class(self.args.model)
        print(Model)
        self.model = Model(**self.args.model_args)
        print(self.model)
        self.loss = nn.CrossEntropyLoss()

    def train(self):
        
        trainsets = TensorData(self.args)
        trainloader = torch.utils.data.DataLoader(trainsets, batch_size=32, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=0.00001, weight_decay=1e-8)
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        loss_ = [] # loss를 저장할 리스트.
        n = len(trainloader)
        name_desc = tqdm(range(self.args.epoch))
        for epoch in name_desc:
            running_loss = 0.0 # 한 에폭이 돌 때 그안에서 배치마다 loss가 나온다. 즉 한번 학습할 때 그렇게 쪼개지면서 loss가 다 나오니 MSE를 구하기 위해서 사용한다.
            for i, data in enumerate(trainloader, 0): # 무작위로 섞인 32개의 데이터가 담긴 배치가 하나씩 들어온다.
                
                inputs, values = data # data에는 X, Y가 들어있다.
                optimizer.zero_grad() # 최적화 초기화.

                outputs = self.model(inputs) # 모델에 입력값을 넣어 예측값을 산출한다.
                loss = self.loss(outputs, values) # 손실함수를 계산. error 계산.
                loss.backward() # 손실 함수를 기준으로 역전파를 설정한다.
                optimizer.step() # 역전파를 진행하고 가중치를 업데이트한다.

                running_loss += loss.item() # epoch 마다 평균 loss를 계산하기 위해 배치 loss를 더한다.
            
            loss_.append(running_loss/n) # MSE(Mean Squared Error) 계산
        
        torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, 'checkpoint.pt')
        

        
        plt.plot(loss_)
        plt.title('Loss')
        plt.xlabel('epoch')
        plt.show()        


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
    

