import matplotlib.pyplot as plt
from classes.Info import PathInfo
import pickle

class Result():

    @staticmethod
    def print_evaluation(model, x, t):
        """

        :param model: Modl that you want evaluate
        :param x: predicting figure: shape = (batch, 32, 32, 3)
        :param t: one-hot label
        :return:
        """

        ev = model.evaluate(x, t)
        print("loss:", end=" ")
        print(ev[0])
        print("accuracy:", end=" ")
        print(ev[1])

    @staticmethod
    def plot_loss_and_acc(pathInfo: PathInfo, history, file_name = None):

        fig, ax = plt.subplot(1, 2, figsize=(10, 5))
        epochs = len(history.history["loss"])
        ax[0].plot(range(epochs), history.history["loss"], label="train_loss", c="tomato")
        ax[0].plot(range(epochs), history.history["val_loss"], label="valid_loss", c="c")
        ax[0].set_xlabel("epochs", fontsize=14)
        ax[0].set_ylabel("loss", fontsize=14)
        ax[0].legend(fontsize=14)

        ax[1].plot(range(epochs), history.history["acc"], label="train_acc", c="tomato")
        ax[1].plot(range(epochs), history.history["val_acc"], label="valid_acc", c="c")
        ax[1].set_xlabel("epochs", fontsize=14)
        ax[1].set_ylabel("acc", fontsize=14)
        ax[1].legend(fontsize=14)

        with open(pathInfo.histories + file_name + ".binaryfile", mode="wb") as f:
            pickle.dump(history, f)

        fig.savefig(pathInfo.loss_and_acc + file_name + "_acc")

    @staticmethod
    def load_history(pathInfo: PathInfo, file_name):
        with open(pathInfo.histories + file_name + ".binaryfile", mode="rb") as f:
            res = pickle.load(f)
        return res

    @staticmethod
    def compare(pathInfo: PathInfo, his1, his1_name, his2, his2_name, file_name = None):

        """

        :param his1: history1 that you want compare
        :param his1_name: his1 label
        :param his2: history2 that you want compare
        :param his2_name: his2 label
        :param file_name: name for saving data
        :return:
        """

        keys = ["loss", "val_loss", "accuracy", "val_accuracy"]
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        epochs = min([len(his1.history["loss"]), len(his2.history["loss"])])

        ind = 0
        for i in range(2):
            for j in range(2):

                ax[i, j].plot(range(epochs), his1.history[keys[ind]][:epochs], label=his1_name)
                ax[i, j].plot(range(epochs), his2.history[keys[ind]][:epochs], label=his2_name)
                ax[i, j].set_xlabel("epochs", fontsize=14)
                ax[i, j].set_ylabel(keys[ind], fontsize=14)
                ax[i, j].legend(fontsize=14)

                ind += 1

            fig.savefig(pathInfo.comparisons + file_name + "_comp")

