import streamlit as st
import torch
from peocess import *
import numpy as np
import plotly.figure_factory as ff
import pandas as pd

st.title(
    """
    # 欢迎来到FMCW app
    """
)



# 选择数据：
option = st.selectbox(
    "请选择一个样例文件进行监测",
    ("0", "1", "2"))

# st.write("You selected:", option)  option返回选择的数
selected_file_num = int(option) 

ls=[]
for data in GetFile(selected_file_num,'HR'):
    ls.append(data.detach().numpy())
ls=np.array(ls)
ls2=[]
for data in GetFile(selected_file_num,'BR'):
    ls2.append(data.detach().numpy())
ls2=np.array(ls2)
x=[ls,ls2]
labels=['心率','呼吸']



fig = ff.create_distplot(
    x,labels
)

st.plotly_chart(fig,use_container_width=True)

# st.write(option)


chart_data = pd.DataFrame(np.array(x).T,columns=labels)
st.area_chart(chart_data)
