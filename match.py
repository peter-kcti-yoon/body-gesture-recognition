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
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

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


def _read_data3(npy_name_list):

    X,y = [],[]
    for npy_name in npy_name_list:
        npy = np.load(os.path.join('./data/data3',npy_name))
        _X = npy[:,:258]
        _y = npy[:,-1]

        _X = _X[np.where(_y!=0)]
        _y = _y[np.where(_y!=0)]
        X.extend(_X)
        y.extend(_y)
        # X.append(_X)
        # y.append(_y)
        # return np.vstack(X).squeeze(), np.vstack(y).squeeze()

    return np.array(X), np.array(y)




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

    def __init__(self, args, X, y, train=False):
        self.args = args
        self.is_train = train
        self.le = LabelEncoder()
        self.le.fit(y)
        self.y_data = self.le.transform(y)
        self.x_data = X
        self.len = self.x_data.shape[0]
        

        

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


    def __getitem__(self, index): 
        anchor_data = self.x_data[index]
        anchor_label = self.y_data[index]
        if self.is_train:    
            _x_data = np.delete(self.x_data, index, axis=0)
            _y_data = np.delete(self.y_data, index, axis=0)
            positive_list = np.where(_y_data == anchor_label)[0]
            negative_list = np.where(_y_data != anchor_label)[0]
            pos_idx = np.random.choice(positive_list,1)[0]
            neg_idx = np.random.choice(negative_list,1)[0]
            # print('>>>>>>>>>>>>', pos_idx, neg_idx)
            pos_data ,neg_data =  _x_data[pos_idx], _x_data[neg_idx]
            # print('>>>>>>>>>>>>', pos_data.shape, neg_data.shape)

            anchor_data = self.transform(anchor_data)
            pos_data = self.transform(pos_data)
            neg_data = self.transform(neg_data)
            anchor_label = torch.from_numpy(np.array(anchor_label)).long()
     
            return anchor_data, pos_data, neg_data, anchor_label
        else:
            anchor_data = self.transform(anchor_data)
            anchor_label = torch.from_numpy(np.array(anchor_label)).long()
            return anchor_data, anchor_label

    def __len__(self):
        return self.len
 

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
        self.load_data()
        if args.test:
            self.test2()
        else:
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
        trainx,trainy = _read_data3(['data_p01_s001.npy','data_p01_s002.npy','data_p01_s003.npy'])
        
        valx, valy    = _read_data3(['data_p02_s001.npy','data_p02_s002.npy','data_p02_s003.npy'])
        self.trainsets = TensorData(self.args,  trainx, trainy, train=True)
        self.trainloader = DataLoader(self.trainsets, batch_size=self.args.train_batch_size, shuffle=True)
        
        self.validset = TensorData(self.args, valx, valy)
        self.validloader = DataLoader(self.validset, batch_size=self.args.test_batch_size, shuffle=False)        

        # TODO make testset loader
        # test_set = TensorData(self.args, valx, valy)
        # self.testloader = DataLoader(validset, batch_size=self.args.test_batch_size, shuffle=False)        

    def test2(self):
        k = 10
        # tmpset =TensorData(self.args, None,None)
        # trainx,trainy = _read_data3(['data_p01_s001.npy','data_p01_s002.npy','data_p01_s003.npy'])

        checkpoint = torch.load('./weights/trip1_lr0.0001_ep400/model_weights.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        train_results,labels  = [], []
        with torch.no_grad():
            self.model.eval() # 평가를 할 땐 반드시 eval()을 사용해야 한다.

        for img, label in self.validloader:
            res = self.model(img).detach().numpy()
            train_results.append(res)
            labels.append(label.detach().numpy())

        train_results = np.concatenate(train_results)
        labels = np.concatenate(labels)
        tsne = TSNE(n_components=2, learning_rate='auto',init='random')
        transformed = tsne.fit_transform(train_results)
        plt.figure(figsize=(15, 10), facecolor="azure")
        for label in np.unique(labels):
            print(label, type(label))
            l = self.validset.le.inverse_transform([label])
            str_label = idx2label[l[0]]
            tmp = transformed[labels==label]
            plt.scatter(tmp[:, 0], tmp[:, 1], label=str_label)

        plt.legend()
        plt.show()

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
        # self.loss = nn.CrossEntropyLoss()
        self.loss = nn.TripletMarginLoss(margin=1.0, p=2)
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
                anchor_data, pos_data, neg_data, anchor_label = data # data에는 X, Y가 들어있다.
                self.optimizer.zero_grad() # 최적화 초기화.

                anchor_out = self.model(anchor_data) 
                pos_out = self.model(pos_data) 
                neg_out = self.model(neg_data) 
                loss = self.loss(anchor_out, pos_out, neg_out) # 손실함수를 계산. error 계산.
                loss.backward() # 손실 함수를 기준으로 역전파를 설정한다.
                self.optimizer.step() # 역전파를 진행하고 가중치를 업데이트한다.

                running_loss += loss.item() # epoch 마다 평균 loss를 계산하기 위해 배치 loss를 더한다.
            
                
                if i %20 == 0:
                    dist_pos = torch.norm(anchor_out-pos_out, 2).mean()
                    dist_neg = torch.norm(anchor_out-neg_out, 2).mean()    
                    # outputs = torch.argmax(self.softmax(outputs), dim=1)
                    # train_acc = torch.mean((values==outputs).float())
                    # self.eval()
                    self.train_writer.add_scalar('Loss', loss, self.global_step)
                    self.train_writer.add_scalar('dist_pos', dist_pos, self.global_step)
                    self.train_writer.add_scalar('dist_neg', dist_neg, self.global_step)
                    # self.train_writer.add_scalar('Accuracy', train_acc, self.global_step)
                    # self.valid_writer.add_scalar('Accuracy', self.avg.avg, self.global_step)
                        
                    # msg = f'Epoch: {str(epoch).zfill(3)}, Step:{self.global_step}, Acc: {self.avg.avg:.4f}' 
                    msg = f'Epoch: {str(epoch).zfill(3)}, Step:{self.global_step} Pos: {dist_pos.detach():.3f}, Neg: {dist_neg.detach():.3f}'

                    name_desc.set_description(msg)
                    
            self.save_weight(epoch, False)
            # if best_epoch_acc < self.avg.avg:
                # best_epoch_acc = self.avg.avg
                # self.save_weight(epoch, True)



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
    

