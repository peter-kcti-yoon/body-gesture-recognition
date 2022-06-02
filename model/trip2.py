from torch import nn

def fc_init(fc):
    nn.init.kaiming_normal_(fc.weight, mode='fan_out')
    # nn.init.xavier_uniform_(fc.weight)
    nn.init.constant_(fc.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class layer(nn.Module):
    def __init__(self,in_channel, out_channel, is_dropout=False  ) -> None:
        super().__init__()

        self.fc = nn.Linear(in_channel, out_channel, bias=True) 
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1) 
        self.is_dropout = is_dropout
        fc_init(self.fc)
        bn_init(self.bn, 1)

    
    def forward(self, x):
        x = self.bn(self.fc(x)) 
        x = self.relu(x)
        if self.is_dropout:
            x = self.dropout(x)
        return x

    

class Model(nn.Module):
    def __init__(self, num_class, num_feature):
        super().__init__()

        self.layer1 = layer(num_feature, 48)
        self.layer2 = layer(48, 48*2)
        self.layer3 = layer(48*2, 48*2, is_dropout=True)
        self.layer4 = layer(48*2, 48, is_dropout=True)
        self.layer5 = layer(48, 24, is_dropout=True)
        self.layer6 = layer(24, 24, is_dropout=True)
        self.fc = nn.Linear(24, 2)
    
        self.model = nn.Sequential(self.layer1,
                                    self.layer2,
                                    self.layer3,
                                    self.layer4,
                                    self.layer5,
                                    self.layer6
                                    )       



    def forward(self, x): # 모델 연산의 순서를 정의
        x = self.model(x)
        return self.fc(x)
        # return self.model(x)
