import streamlit as st
import pandas as pd
import numpy as np
from inference import predict
import os
import pickle
from typing import Tuple


import base64


def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.

    Returns
    -------
    The background.
    '''
    main_bg_ext = "png"

    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )


set_bg_hack('bg.png')


st.markdown(
    """<h1 style='text-align: end;font-family : ui-rounded;color: rgb(214, 122, 127);font-size:70px;margin-top:-50px;'>AUDIO GENRE CLASSIFIER</h1>""",
    unsafe_allow_html=True)


file = st.sidebar.file_uploader("Upload Audio To Classify", type=["wav"])
rad_test = 'wav'
if rad_test == 'wav':
    if file is not None:
        st.markdown(
            """<h1 style='color:gray;text-align:end; font-family:Georgia, serif;>Audio </h1>""",
            unsafe_allow_html=True)
        st.audio(file)
        col1, col2, col3 = st.columns(3)
        if col2.button("Classify Audio"):
            prediction = predict(file)
            st.markdown(
                f"""<h1 style='color:#778899;text-align:center;font-family:Georgia;'>Woah you chose  <span style='color:rgb(214, 122, 127); font-style:Apple Chancery'>{prediction}!!</span> </h1>""",

                unsafe_allow_html=True)
