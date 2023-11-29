import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics

from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers.legacy import Adam

# from main import df_shape

class GAN():
    def __init__(self, df_shape, train_data, test_data, y_true) -> None:
        self.ip_shape = df_shape
        self.train_data = train_data
        self.test_data = test_data
        self.y_true = y_true

        optimizer = Adam(learning_rate=0.00001, beta_1=0.5)

        # build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # build generator
        self.generator = self.build_generator()

        # generator generates noise
        z = Input(shape=(self.ip_shape,))
        genData = self.generator(z)

        # train only the generator
        self.discriminator.trainable = False

        # discriminator takes generated noise as ip and determines validity
        validity = self.discriminator(genData)

        # combined model
        self.gan = Model(z, validity)
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    # defining generator
    def build_generator(self):
        model = Sequential()
        
        model.add(Dense(64, input_dim=self.ip_shape))
        model.add(Activation('tanh'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(128))
        model.add(Activation('tanh'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256))
        model.add(Activation('tanh'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(Activation('tanh'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.ip_shape, activation='tanh'))

        # model.summary()

        noise = Input(shape=(self.ip_shape,))
        genData = model(noise)

        return Model(noise, genData)

    # defining generator
    def build_discriminator(self):
        model = Sequential()

        model.add(Dense(512))
        model.add(Activation('tanh'))
        model.add(Dense(256))
        model.add(Activation('tanh'))
        model.add(Dense(256))
        model.add(Activation('tanh'))
        model.add(Dense(128))
        model.add(Activation('tanh'))
        model.add(Dense(1, activation='sigmoid'))

        # model.summary()

        data = Input(self.ip_shape)
        validity = model(data)

        return Model(data, validity)
    
    # training models
    def train(self, epochs, batch_size = 128):
        # 
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # -------------------
            # train Discriminator
            # -------------------

            # selecting random batch of data
            i = np.random.randint(0, self.train_data.shape[0], batch_size)
            imgs = self.train_data[i]
            
            noise = np.random.normal(0, 1, (batch_size, self.ip_shape))
            # generate new batch of data
            gen_imgs = self.generator.predict(noise)

            # training discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------
            # train Generator
            # ---------------

            noise = np.random.normal(0, 1, (batch_size, self.ip_shape))

            g_loss = self.gan.train_on_batch(noise, valid)

            print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

    # testing on discriminator
    def test(self):
        preds = self.discriminator.predict(self.test_data)
        self.y_pred = np.array(preds)

        per = np.percentile(preds, 10)
        inds = (self.y_pred > per)
        inds_comp = (self.y_pred <= per)

        self.y_pred[inds] = 0
        self.y_pred[inds_comp] = 1

        self.precision_gan, self.recall_gan, self.f1_score_gan, _ = metrics.precision_recall_fscore_support(self.y_true, self.y_pred, average='binary')
        self.fpr, self.tpr, _ = metrics.roc_curve(self.y_true, self.y_pred)
        self.auc_roc_gan = metrics.auc(self.fpr, self.tpr)

        print("Metrics: ")
        print('Precision: {:.4f}'.format(self.precision_gan))
        print('Recall: {:.4f}'.format(self.recall_gan))
        print('F1 score: {:.4f}'.format(self.f1_score_gan))
        print('AUC-ROC socre: {:.4f}'.format(self.auc_roc_gan))

    # visualize results
    def visualize(self):
        # Plot ROC curve
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.fpr, self.tpr, label=f'AUC = {self.auc_roc_gan:.4f}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        st.pyplot(fig)

        # true v predicted labels
        st.divider()
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        ax.plot(self.y_true, label='True Labels', color='blue', linestyle='-.')
        ax.plot(self.y_pred, label='Predicted Labels', color='yellow')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Label (0: Normal, 1: Anomalous)')
        ax.set_title('True vs. Predicted Labels')
        ax.legend(loc=(1.05, 0.5))
        st.pyplot(fig)
        
        # confusion matrix
        st.divider()
        conf_matrix = metrics.confusion_matrix(self.y_true, self.y_pred)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax)
        plt.title('Confusion Matrix')
        st.pyplot(fig)

# if __name__ == '__main__':
#     # df = pd.read_csv(filename, usecols=lambda col:col != 'Class')
#     X_data = dataframe[:, :-1]
#     df_shape = len(X_data.columns)
#     y_true = dataframe[:, -1]

#     tmp = len(X_data)
#     df = X_data.astype('float32')
#     df = np.array(df)

#     train_size = int(tmp * 0.7)
#     test_size = tmp
#     test_data = df
#     train_data = []

#     for i in range(train_size):
#         train_data.append(df[i])
#     train_data = np.array(train_data)

#     gan = GAN()
#     gan.train(epochs=50, batch_size=32)
#     gan.test()
#     gan.visualize()
