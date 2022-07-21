from keras.preprocessing.image import ImageDataGenerator
import numpy as np

class Utilities():

    @staticmethod
    def da_generator(X_train, y_train, batch_size):
        return ImageDataGenerator(rotation_range=20,
                                  horizontal_flip=True,
                                  height_shift_range=.2,
                                  width_shift_range=.2,
                                  zoom_range=.2,
                                  channel_shift_range=.2).flow(X_train, y_train, batch_size)

    @staticmethod
    def da_generator_strong(X_train, y_train, batch_size):
        return ImageDataGenerator(rotation_range=20,
                                  horizontal_flip=True,
                                  height_shift_range=.3,
                                  width_shift_range=.3,
                                  zoom_range=.3,
                                  channel_shift_range=.3).flow(X_train, y_train, batch_size)

    @staticmethod
    def step_decay(epoch):
        lr = 0.001

        if(epoch >= 100):
            lr/=5

        if(epoch >= 140):
            lr/=2

        return lr

    @staticmethod
    def tta(model, test_size, generator, batch_size, epochs = 10):

        pred = np.zeros(shape=(test_size, 10), dtype=float)
        step_per_epoch = test_size//batch_size

        for epoch in range(epochs):
            for step in range(step_per_epoch):
                sta = batch_size * step
                end = sta + batch_size
                tmp_x = generator.__next__()
                pred[sta:end] += model.predict(tmp_x)

        return pred/epochs

    @staticmethod
    def tta_generator(X_test, batch_size):
        return ImageDataGenerator(rotation_range=20, horizontal_flip=True,
                                  height_shift_range=.2, width_shift_range=.2, zoom_range=.2,
                                  channel_shift_range=.2).flow(X_test, batch_size=batch_size, shuffle=False)

