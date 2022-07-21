import os, itertools

class PathInfo():

    def __init__(self):

        self._cwd = os.getcwd()
        self._path_parent_project = os.path.abspath(os.path.join(self._cwd, os.pardir))

        self._comparisons = self._path_parent_project + "\\comparisons\\"
        self._histories = self._path_parent_project + "\\histories\\"
        self._loss_and_acc = self._path_parent_project + "\\loss_and_acc\\"
        self._models = self._path_parent_project + "\\models\\"
        self._predictions = self._path_parent_project + "\\predictions\\"

        self._create_folder(self._comparisons)
        self._create_folder(self._histories)
        self._create_folder(self._loss_and_acc)
        self._create_folder(self._models)
        self._create_folder(self._predictions)

    def _create_folder(self, path):

        if not (os.path.exists(path)):
            os.makedirs(path)

    @property
    def comparisons(self):
        return self._comparisons

    @property
    def histories(self):
        return self._histories

    @property
    def loss_and_acc(self):
        return self._loss_and_acc

    @property
    def models(self):
        return self._models

    @property
    def predictions(self):
        return self._predictions