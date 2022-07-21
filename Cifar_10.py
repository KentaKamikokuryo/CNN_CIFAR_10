import keras
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import seaborn as sns

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

fig = plt.figure(figsize=(13.0, 13.0))

label_list = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "flog", "horse", "ship", "truck"]

sns.set(style='ticks', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=1.5)
color = sns.color_palette("Set2", 6)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

for i in range(25):
  plt.subplot(5, 5, i+1)
  plt.subplots_adjust(wspace=0.4, hspace=0.6)
  plt.imshow(x_train[i])
  plt.title(label_list[y_train[i][0]])
  plt.axis("off")