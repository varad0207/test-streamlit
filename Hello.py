import streamlit as st
import numpy as np
import pandas as pd

from models.gan import GAN

st.title('Generative Adversarial Networks')
st.divider()
st.header('Using GAN for Anomaly Detection of various datasets')

uploaded_file = st.file_uploader('Upload a dataset')
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    
    X_data = dataframe.iloc[:, :-1]
    df_shape = len(X_data.columns)
    y_true = dataframe.iloc[:, -1]

    tmp = len(X_data)
    df = X_data.astype('float32')
    df = np.array(df)

    train_size = int(tmp * 0.7)
    test_size = tmp
    test_data = df
    train_data = []

    for i in range(train_size):
        train_data.append(df[i])
    train_data = np.array(train_data)


    st.divider()

    gan_model = GAN(df_shape, train_data, test_data, y_true)
    gan_model.train(epochs=50, batch_size=32)
    gan_model.test()
    gan_model.visualize()
