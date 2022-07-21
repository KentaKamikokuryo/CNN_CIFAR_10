import numpy as np
import keras
from keras.optimizers import adam_v2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pickle

from classes.DeepModels import *
from classes.Result import *
from classes.Utilities import *
from classes.Info import *

(x_train_raw, t_train_raw), (x_test_raw, t_test_raw) = cifar10.load_data()
t_train = to_categorical(t_train_raw)
t_test = to_categorical(t_test_raw)
x_train = x_train_raw / 255
x_test = x_test_raw / 255

batch_size = 500
epochs = 20
steps_per_epoch = x_train.shape[0] // batch_size
validation_steps = x_test.shape[0] // batch_size

pathInfo = PathInfo()



"""Bench mark"""
model = ModelCreater.createBenchModel()
model.compile(loss="categorical_crossentropy", optimizer=adam_v2.Adam(), metrics=["accuracy"])
train_gen = ImageDataGenerator().flow(x_train, t_train, batch_size=batch_size)
val_gen = ImageDataGenerator().flow(x_test, t_test, batch_size=batch_size)
history = model.fit_generator(train_gen, epochs=epochs, steps_per_epoch=steps_per_epoch,
                              validation_data=val_gen, validation_steps=validation_steps)

Result.print_evaluation(model, x_test, t_test)
Result.plot_loss_and_acc(pathInfo=pathInfo, history=history, file_name="bench")
model.save(pathInfo.models + "bench.hdf5")

"""Data Augmentation"""
model = ModelCreater.createBenchModel()
model.compile(loss="categorical_crossentropy", optimizer=adam_v2.Adam(), metrics=["accuracy"])
val_gen = ImageDataGenerator().flow(x_test, t_test, batch_size=batch_size)
history = model.fit_generator(Utilities.da_generator(X_train=x_train, y_train=t_train, batch_size=batch_size),
                              epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_gen,
                              validation_steps=validation_steps)

Result.print_evaluation(model, x_test, t_test)
Result.plot_loss_and_acc(pathInfo=pathInfo, history=history, file_name="DA")
bench_history = Result.load_history(pathInfo=pathInfo, file_name="bench")
Result.compare(pathInfo=pathInfo, his1=bench_history, his1_name="no_DA", his2=history, his2_name="DA",
               file_name="no_DA_vs_DA")
model.save(pathInfo.models + "DA.hdf5")

"""deep layer"""
model = ModelCreater.createDeepModel()
model.compile(loss="categorical_crossentropy", optimizer=adam_v2.Adam(), metrics=["accuracy"])
val_gen = ImageDataGenerator().flow(x_test, t_test, batch_size)
history = model.fit_generator(Utilities.da_generator(X_train=x_train, y_train=t_train, batch_size=batch_size),
                              epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_gen,
                              validation_steps=validation_steps)

Result.print_evaluation(model, x_test, t_test)
Result.plot_loss_and_acc(pathInfo=pathInfo, history=history, file_name="deep")
bench_history = Result.load_history(pathInfo=pathInfo, file_name="DA")
Result.compare(pathInfo=pathInfo, his1=bench_history, his1_name="bench", his2=history, his2_name="deep",
               file_name="bench_vs_deep")
model.save(pathInfo.models + "deep.hdf5")

"""learning rate decay"""
model = ModelCreater.createDeepModel()
model.compile(loss="categorical_crossentropy", optimizer=adam_v2.Adam(), metrics=["accuracy"])
val_gen = ImageDataGenerator().flow(x_test, t_test, batch_size)
lr_decay = LearningRateScheduler(Utilities.step_decay)
history = model.fit_generator(Utilities.da_generator(X_train=x_train, y_train=t_train, batch_size=batch_size),
                              epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_gen,
                              validation_steps=validation_steps, callbacks=[lr_decay])

Result.print_evaluation(model, x_test, t_test)
Result.plot_loss_and_acc(pathInfo=pathInfo, history=history, file_name="lr_decay")
no_decay_history = Result.load_history(pathInfo=pathInfo, file_name="deep")
Result.compare(pathInfo=pathInfo, his1=no_decay_history, his1_name="no_decay", his2=history, his2_name="decay",
               file_name="no_decay_vs_decay")
model.save(pathInfo.models + "lr_decay.hdf5")

"""Ensemble"""
ens_epochs = 2
tta_epochs = 50

for i in range(ens_epochs):
    model = ModelCreater.createDeepModel()
    model.compile(loss="categorical_crossentropy", optimizer=adam_v2.Adam(), metrics=["accuracy"])
    val_gen = ImageDataGenerator().flow(x_test, t_test, batch_size)
    lr_decay = LearningRateScheduler(Utilities.step_decay)

    if(i < ens_epochs/2):
        train_gen = Utilities.da_generator(X_train=x_train, y_train=t_train, batch_size=batch_size)
    else:
        train_gen = Utilities.da_generator_strong(X_train=x_train, y_train=t_train, batch_size=batch_size)

    his = model.fit_generator(train_gen, epochs=epochs, steps_per_epoch=steps_per_epoch,
                              validation_data=val_gen, validation_steps=validation_steps, verbose=0,
                              callbacks=[lr_decay])

    pred = Utilities.tta(model=model, test_size=x_test.shape[0],
                         generator=Utilities.tta_generator(X_test=x_test, batch_size=batch_size),
                         batch_size=batch_size, epochs=tta_epochs)

    np.save(pathInfo.predictions + "pred_" + str(i), pred)
    model.save(pathInfo.models + "ensemble_" + str(i) + ".hdf5")

acc_list = []
final_pred = np.zeros_like(t_test)
for i in range(ens_epochs):
    pred = np.load(pathInfo.predictions + "pred_" + str(i) + ".npy")
    acc_list.append(accuracy_score(np.argmax(pred, axis=1), np.argmax(t_test, axis=1)))
    final_pred += pred

final_pred /= ens_epochs
np.save(pathInfo.predictions + "final_pred", final_pred)
print("acc_mean: ",end = "")
print( np.mean(acc_list))
print("final_acc: " ,end = "")
print( accuracy_score(np.argmax(final_pred,axis = 1), np.argmax(t_test,axis = 1)))

