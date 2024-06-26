import torch
import numpy as np
import torch.nn.functional as F
import os

path = os.path.join('models/data/data.npz') #果然，部署是Linux，所以要用/而非\  --windows
# print(path)

def get_data(x:int,s:str): #选定第几个数据进行 分为三段：每段60  x:{0,1,2}
    data = np.load(path,allow_pickle=True)

    breath_phase = F.normalize(torch.tensor(data['arr_0']),p=2,dim=-1,)
#     breath_phase = torch.tensor(data['arr_0'])
    
    heart_phase =F.normalize(torch.tensor(data['arr_1']),p=2,dim=-1,)

    breath_refer = torch.tensor(data['arr_2'])
    heart_refer = torch.tensor(data['arr_3'])


    breath_phase = torch.unsqueeze(breath_phase,dim=1)
    heart_phase = torch.unsqueeze(heart_phase,dim=1)
    l=x*60 #确定索引：
    r=60+l
    if s=='HR':
        return heart_phase[l:r:6] #每隔六步一个
    else :
        return breath_phase[l:r:6]

# data = get_data(0,'HR')
# print(len(data))