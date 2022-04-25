from torch import nn


class Model(nn.Module):
    def __init__(self, num_class, num_feature):
        super().__init__() # 모델 연산 정의
        self.fc1 = nn.Linear(num_feature, 330, bias=True) 
        self.fc2 = nn.Linear(330, 160, bias=True) 
        self.fc3 = nn.Linear(160, 80, bias=True) 
        self.fc4 = nn.Linear(80, 30, bias=True)         
        self.fc = nn.Linear(30, num_class, bias=True) 
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc.weight)

        self.relu = nn.ReLU()
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
