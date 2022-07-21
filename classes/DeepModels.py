from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D, BatchNormalization

class ModelCreater():

    @staticmethod
    def createBenchModel():

        inputs = Input(shape=(32, 32, 3))

        x = Conv2D(64, (3, 3), padding="SAME", activation="relu")(inputs)
        x = Conv2D(64, (3, 3), padding="SAME", activation="relu")(x)
        x = Dropout(0.25)(x)
        x = MaxPooling2D()(x)

        x = Conv2D(128, (3, 3), padding="SAME", activation="relu")(x)
        x = Conv2D(128, (3, 3), padding="SAME", activation="relu")(x)
        x = Dropout(0.25)(x)
        x = MaxPooling2D()(x)

        x = Conv2D(256, (3, 3), padding="SAME", activation="relu")(x)
        x = Conv2D(256, (3, 3), padding="SAME", activation="relu")(x)
        x = GlobalAveragePooling2D()(x)

        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.25)(x)
        y = Dense(10, activation="softmax")(x)

        return Model(inputs, y)

    @staticmethod
    def createDeepModel():

        inputs = Input(shape=(32, 32, 3))

        x = Conv2D(64, (3, 3), padding="SAME", activation="relu")(inputs)
        x = Conv2D(64, (3, 3), padding="SAME", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), padding="SAME", activation="relu")(x)
        x = MaxPooling2D()(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding="SAME", activation="relu")(x)
        x = Conv2D(128, (3, 3), padding="SAME", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), padding="SAME", activation="relu")(x)
        x = MaxPooling2D()(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3, 3), padding="SAME", activation="relu")(x)
        x = Conv2D(256, (3, 3), padding="SAME", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), padding="SAME", activation="relu")(x)
        x = Conv2D(256, (3, 3), padding="SAME", activation="relu")(x)
        x = Conv2D(256, (3, 3), padding="SAME", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), padding="SAME", activation="relu")(x)
        x = Conv2D(512, (3, 3), padding="SAME", activation="relu")(x)
        x = GlobalAveragePooling2D()(x)

        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)
        y = Dense(10, activation="softmax")(x)

        return Model(inputs, y)



