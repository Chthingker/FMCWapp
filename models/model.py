import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.ops as ops

def gen_iter(x=0):
    """
    选取第几个chirp作为连续长时间序列上的观测对象 0-5 x取值
    :param x:
    :return:
    """
    data = np.load('models\data\data.npz\data.npz',allow_pickle=True)

    breath_phase = F.normalize(torch.tensor(data['arr_0']),p=2,dim=-1,)
#     breath_phase = torch.tensor(data['arr_0'])
    
    heart_phase =F.normalize(torch.tensor(data['arr_1']),p=2,dim=-1,)

    breath_refer = torch.tensor(data['arr_2'])
    heart_refer = torch.tensor(data['arr_3'])

    i=x
    while i<180 :
        yield breath_phase[i],heart_phase[i],breath_refer[i],heart_refer[i]
        i+=6


        

def get_X_y(fold):
    for breath_phase,heart_phase,breath_refer,heart_refer in gen_iter():
        breath_phase = torch.unsqueeze(breath_phase,dim=1)
        heart_phase = torch.unsqueeze(heart_phase,dim=1)
        breath_refer = torch.unsqueeze(breath_refer,dim=-1)
        heart_refer = torch.unsqueeze(heart_refer,dim=-1)
        if not torch.any(torch.isnan(heart_refer)):
            yield breath_phase[:fold,:],heart_phase[:fold,:],breath_refer,heart_refer
    
def get_X_y_test(fold):
    for breath_phase,heart_phase,breath_refer,heart_refer in gen_iter():
        breath_phase = torch.unsqueeze(breath_phase,dim=1)
        heart_phase = torch.unsqueeze(heart_phase,dim=1)
        breath_refer = torch.unsqueeze(breath_refer,dim=-1)
        heart_refer = torch.unsqueeze(heart_refer,dim=-1)
        if not torch.any(torch.isnan(heart_refer)):
            yield breath_phase[fold:,:],heart_phase[fold:,:],breath_refer,heart_refer

# for breath_phase,heart_phase,breath_refer,heart_refer in get_X_y(7):
#     print(breath_phase.shape)
#     print(breath_refer.shape)
def get_freqdomain_data(n:int):
    for breath_phase,heart_phase,breath_refer,heart_refer in get_X_y(n):
        breath_phase = F.normalize(torch.abs(torch.fft.fft(breath_phase,dim=-1)) , dim=-1)
#         breath_phase = torch.unsqueeze(breath_phase,dim=1)
        heart_phase = F.normalize(torch.abs(torch.fft.fft(heart_phase,dim=-1)) , dim=-1)
#         heart_phase = torch.unsqueeze(heart_phase,dim=1)
        breath_refer = torch.unsqueeze(breath_refer,dim=-1)
        heart_refer = torch.unsqueeze(heart_refer,dim=-1)
        
        yield breath_phase[:n,:],heart_phase[:n,:],breath_refer,heart_refer  #未归一化数据




class HeartTCNBlock(nn.Module):
    def __init__(self, input_channel, output_channel, dilation=1, dropout=0.02):
        super().__init__()
        self.res = nn.Conv2d(input_channel, output_channel, 1, padding=0)
        self.relu = nn.SiLU()

        self.conv1 = nn.Conv2d(input_channel, 8, 1, padding=0)
        self.bn1 = nn.BatchNorm2d(8)

        # Dilated Causal Convolution
        dila_ker_size = 9
        self.conv2 = nn.Conv2d(8, 8, dila_ker_size, padding=(dila_ker_size - 1) * dilation // 2, dilation=dilation)

        self.bn2 = nn.BatchNorm2d(8)

        # SE Block
        self.se = ops.SqueezeExcitation(8, 8)

        self.conv3 = nn.Conv2d(8, output_channel, 1, padding=0)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.se(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)
        return self.relu(out + self.res(x))


class HeartTCN(nn.Module):
    def __init__(self, phase_length=127, input_channel=1, output_channel=32, dropout=0.02, model_type='hr'):
        super().__init__()
        self.block1 = HeartTCNBlock(input_channel, output_channel, 1, dropout)
        self.block2 = HeartTCNBlock(output_channel, output_channel, 2, dropout)
        self.block3 = HeartTCNBlock(output_channel, output_channel, 4, dropout)
        self.block4 = HeartTCNBlock(output_channel, output_channel, 8, dropout)
        self.block5 = HeartTCNBlock(output_channel, 8, 16, dropout)
        self.conv = nn.Conv2d(8, 1, 1, padding=0)

        self.model_type = model_type
        if model_type == 'hr':
            self.ave_pool = nn.AdaptiveAvgPool2d((1, 1))
        if model_type == 'pulse':
            self.fc = nn.Linear(1 * phase_length * 5, phase_length + 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.conv(x)

        x = (self.ave_pool(x).flatten(1) if self.model_type == 'hr'
             else self.fc(torch.flatten(x, 1)))

        return x

# 训练模型：
# model = HeartTCN()
# l=[]
# lt=[]

# def train():
#     num_epoches = 30
#     optimizer = torch.optim.SGD(model.parameters(),lr=0.0003)
#     Lossf = nn.MSELoss()
    
#     for epoch in range(num_epoches):
#         ls=0
#         for breath_phase,heart_phase,breath_refer,heart_refer in get_X_y(7):
#             X = torch.unsqueeze(breath_phase,dim=1).to(torch.float32)
#             y = breath_refer.repeat(7,1).to(torch.float32)
#             optimizer.zero_grad()
#             Y = model(X)
#             loss = Lossf(y,Y)
#             ls+=loss.item()
#             loss.backward()
#             optimizer.step()
#         l.append(ls)

#         print(epoch,":",l[-1])

# train()

# PATH = 'models/net_BR_save.pt'
# torch.save(model.state_dict(), PATH)
            
