# ANN module
import sys
sys.path.append('..')
import torch
from torch import nn, optim                           # torch 에서 제공하는 신경망 기술, 손실함수, 최적화를 할 수 있는 함수들을 불러온다.
from torch.utils.data import DataLoader, Dataset      # 데이터를 모델에 사용할 수 있게 정리해주는 라이브러리.
import torch.nn.functional as F                       # torch 내의 세부적인 기능을 불러옴.
from tqdm import tqdm
# Loss
from sklearn.metrics import mean_squared_error        # regression 문제의 모델 성능 측정을 위해서 MSE를 불러온다.

import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import argparse

from src.tools import *
from src.opt import *
# torch의 Dataset 을 상속.

N_CLASSES = 'n_classes'
TEST_ACTIONS = 'test_actions'
TRAIN_ACTIONS='train_actions'
class TensorData(Dataset):

    def __init__(self, actions, n_classes, is_train=True):
        self.n_classes = n_classes
        print('TENSOR DATASET', actions)
        if is_train:
            self.x_data, self.y_data = self.train_data(actions)
        else:
            self.x_data, self.y_data = self.test_data(actions)
        self.len = self.y_data.shape[0]


    def train_data(self, actions):
        x_data, y_data = load_dataset(1, actions)
        x_data, y_data = unpack_dataset(x_data, y_data)
        y_data = [ actions.index(a) for a in y_data]
        # y_data = F.one_hot(torch.Tensor(y_data), num_classes = self.n_classes)
        y_data = F.one_hot(torch.Tensor(y_data).long(), num_classes = len(actions))
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
        y_data = [ actions.index(a) for a in y_data]
        # y_data = F.one_hot(torch.Tensor(y_data), num_classes = self.n_classes)
        y_data = F.one_hot(torch.Tensor(y_data).long(), num_classes = len(actions))
        return x_data, y_data

    def __getitem__(self, index):
        x = torch.from_numpy(self.x_data[index]).float()
        y = self.y_data[index].float() 
        # torch.from_numpy(np.array(self.y_data[index])).float()
        return x,y 

    def __len__(self):
        return self.len

    
class Regressor(nn.Module):
    def __init__(self, n_class):
        super().__init__() # 모델 연산 정의
        self.fc1 = nn.Linear(258, 50, bias=True) # 입력층(13) -> 은닉층1(50)으로 가는 연산
        self.fc2 = nn.Linear(50, 30, bias=True) # 은닉층1(50) -> 은닉층2(30)으로 가는 연산
        self.fc3 = nn.Linear(30, n_class, bias=True) # 은닉층2(30) -> 출력층(1)으로 가는 연산
        self.dropout = nn.Dropout(0.2) # 연산이 될 때마다 20%의 비율로 랜덤하게 노드를 없앤다.

    def forward(self, x): # 모델 연산의 순서를 정의
        x = F.relu(self.fc1(x)) # Linear 계산 후 활성화 함수 ReLU를 적용한다.  
        x = self.dropout(F.relu(self.fc2(x))) # 은닉층2에서 드랍아웃을 적용한다.(즉, 30개의 20%인 6개의 노드가 계산에서 제외된다.)
        # x = F.relu(self.fc3(x)) # Linear 계산 후 활성화 함수 ReLU를 적용한다.  
        x = self.fc3(x) # Linear 계산 후 활성화 함수 ReLU를 적용한다.  
        return x




def train(configs):
    criterion = nn.MSELoss()
    model = Regressor(configs[N_CLASSES])
    trainsets = TensorData(configs[TRAIN_ACTIONS], configs[N_CLASSES], is_train=True)
    trainloader = torch.utils.data.DataLoader(trainsets, batch_size=32, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-7)
    loss_ = [] # loss를 저장할 리스트.
    n = len(trainloader)
    name_desc = tqdm(range(200))
    for epoch in range(200):
        name_desc.update(1)
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
    
    plt.plot(loss_)
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.show()

def evaluation(configs):
    model = Regressor(configs[N_CLASSES])
    testset = TensorData(configs[TEST_ACTIONS],configs[N_CLASSES], is_train=False)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

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
    rmse = np.sqrt(mean_squared_error(predictions, actual)) # sklearn을 이용해 RMSE를 계산.

    acc = np.mean((actual==predictions).astype(int))
    print('rmse', rmse)
    
    t1 = np.argmax(actual, axis=1)
    t2 = np.argmax(predictions, axis=1)
    acc = np.mean((t1==t2).astype(int))
    print('Accuracy', acc,'%')
    print_results(predictions, actual, configs[TEST_ACTIONS])

    return rmse


def print_results(pred, label, actions):
    t1 = np.argmax(pred, axis=1)
    t2 = np.argmax(label, axis=1)
    acc = np.mean((t1==t2).astype(int))
    print(f'mAP: {acc:.2f}%')

    for i, a in enumerate(actions):
        indices = np.where(t2 ==i)
        pred_mask = t1[indices]
        label_mask = t2[indices]
        acc = np.mean((pred_mask==label_mask).astype(int))
        print(f'\t {a}: {acc:.2f}%')


def parse_args():
    arg = argparse.ArgumentParser(description='aaaa')
    arg.add_argument('-e', '--eval', action='store_true')
    arg.add_argument('-a', '--all', action='store_true')
    return arg.parse_args()

if __name__=='__main__':
    args = parse_args()
    data_config_dict={
        1: {
            TEST_ACTIONS: actions_dict[2],
            TRAIN_ACTIONS: actions_dict[2],
            N_CLASSES : 6
        },

        2: {
            TEST_ACTIONS: actions_dict[3],
            TRAIN_ACTIONS: actions_dict[3],
            N_CLASSES : 4
        }
    }

    config_index = 2
    
    if args.eval:
        evaluation(data_config_dict[config_index])
    elif args.all:
        train(data_config_dict[config_index])
        evaluation(data_config_dict[config_index])
    else:
        train(data_config_dict[config_index])