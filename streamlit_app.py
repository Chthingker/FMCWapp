import streamlit as st
import torch
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

st.write(option.type)