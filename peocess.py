import torch
from models.model import *
from getdata import *
BR_model = r"models/net_BR_save.pt"
HR_model = r"models/net_save.pt"


# net2 = HeartTCN()
# HR_model = 
# my_model = net2.load_state_dict(torch.load(BR_model, map_location=torch.device('cpu')))

# 然后就可以分析数据了

def analysis(model:str , data): #data 是单条数据 本函数必须返回分析后的numpy
    net = HeartTCN()
    if model == 'HR' :
        net.load_state_dict(torch.load(HR_model, map_location=torch.device('cpu')))
    else:
        net.load_state_dict(torch.load(BR_model, map_location=torch.device('cpu')))
    # print(net)
    return net(data)


def GetFile(x:int,s:str):
    data = get_data(x,s)
    for i in range(60):
        X = torch.unsqueeze(data[i],dim=1).to(torch.float32)
        # print(analysis('HR',X)) #([1, 12, 1200])
        # print(data[i].shape)
        # break
        yield analysis(s,X)


    