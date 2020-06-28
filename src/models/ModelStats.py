class ModelStats(object):
    def __init__(self, model, history, evaluation):
        self._model = model
        self._history = history
        self._evaluation = evaluation

    def getModel(self):
        return self._model

    def getHistory(self):
        return self._history

    def getEvaluation(self):
        return self._evaluation
