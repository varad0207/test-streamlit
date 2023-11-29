import streamlit as st
import numpy as np
import pandas as pd

from models.gan import GAN
from models.lof import lof

def load_data_for_train_test(df):
    X_data = df.iloc[:, :-1]
    df_shape = len(X_data.columns)
    y_true = df.iloc[:, -1]

    tmp = len(X_data)
    df = X_data.astype('float32')
    df = np.array(df)

    train_size = int(tmp * 0.7)
    test_data = df
    train_data = []

    for i in range(train_size):
        train_data.append(df[i])
    train_data = np.array(train_data)

    return train_data, test_data, y_true, df_shape

def visualize_res(selected_model, uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        train_data, test_data, y_true, df_shape = load_data_for_train_test(df)
        with st.expander('See Dataset'):
            st.caption('Uploaded Dataset')
            st.table(df)

        if selected_model == 'GAN':
            gan_model = GAN(df_shape, train_data, test_data, y_true)
            gan_model.train(epochs=50, batch_size=32)
            gan_model.test()
            gan_model.visualize()

        elif selected_model == 'LOF':
            lof_model = lof(train_data, test_data, y_true)
            lof_model.visualize()


st.title('Hello Scientists')
st.divider()
st.header('Explore the World of Anomaly Detection')


# dropdown
selected_model = st.selectbox('Choose a Model', 
                              ['GAN', 'LOF'],
                              index=None,
                                placeholder='Select a model',)
uploaded_file = st.file_uploader('Upload a dataset')

if st.button('Visualize'):
    visualize_res(selected_model, uploaded_file)
