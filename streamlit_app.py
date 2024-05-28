import streamlit as st
import torch
from peocess import *
import numpy as np
import plotly.figure_factory as ff

st.write(
    """
    # welcome **world!**
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


fig = ff.create_distplot(
    ls
)

st.plotly_chart(fig,use_container_width=True)

# st.write(option)


