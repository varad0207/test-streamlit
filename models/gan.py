import numpy as np

from sklearn import metrics

from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers.legacy import Adam

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

            # print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

    # testing on discriminator
    def test(self):
        preds = self.discriminator.predict(self.test_data)
        y_pred = np.array(preds)

        per = np.percentile(preds, 10)
        inds = (y_pred > per)
        inds_comp = (y_pred <= per)

        y_pred[inds] = 0
        y_pred[inds_comp] = 1

        precision_gan, recall_gan, f1_score_gan, _ = metrics.precision_recall_fscore_support(self.y_true, y_pred, average='binary')
        fpr, tpr, _ = metrics.roc_curve(self.y_true, y_pred)
        auc_roc_gan = metrics.auc(fpr, tpr)

        print("Metrics: ")
        print('Precision: {:.4f}'.format(precision_gan))
        print('Recall: {:.4f}'.format(recall_gan))
        print('F1 score: {:.4f}'.format(f1_score_gan))
        print('AUC-ROC socre: {:.4f}'.format(auc_roc_gan))

        return self.y_true, y_pred, fpr, tpr, auc_roc_gan
