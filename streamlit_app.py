import streamlit as st
import torch
from peocess import *
import numpy as np

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


# st.write(option)


