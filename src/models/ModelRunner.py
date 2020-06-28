import os
import shutil
from builtins import staticmethod

from src.models.ModelStats import ModelStats
from src.MuraLoader import MuraLoader
from src.seeded import tf
from src.models.tf_utils import metrics, callbacks, createModelPath
from os import path


class ModelRunner(object):
    def __init__(self, muraLoader: MuraLoader):
        self._muraLoader = muraLoader

    # def clean_up(model):
    #     K.clear_session()
    #     del model
    #     gc.collect()

    def run(self, name: str, model: tf.keras.models.Model, epochs=5, verbose=1, overwrite=False):
        modelPath = createModelPath(name)
        if path.exists(modelPath):
            if overwrite:
                shutil.rmtree(modelPath)
            else:
                raise FileExistsError
        os.mkdir(modelPath)

        model = self.__compile(model)
        self.save_model_summary(name=name, model=model)
        self.save_model_graph(name=name, model=model)

        if verbose == 1:
            print(model.summary())

        train_set, validation_set, test_set = self._muraLoader.get_sets(0.2)
        train_generator, validation_generator, test_generator = self._muraLoader.get_generators(train_set, validation_set, test_set, batch_size=32)

        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks(model=name),
            verbose=verbose
        )

        self.save_model_history(name=name, history=history.history)

        evaluation = model.evaluate(test_generator)

        return ModelStats(model, history, evaluation)

    def __compile(self, model: tf.keras.models.Model):
        adam = tf.optimizers.Adam()
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        model.compile(adam, loss=loss, metrics=[metrics()])

        return model

    @staticmethod
    def save_model_summary(name, model):
        from contextlib import redirect_stdout
        print("Saving Model Summary")

        with open(createModelPath(name, sub="model_summary.txt"), 'w') as f:
            with redirect_stdout(f):
                model.summary()

    @staticmethod
    def save_model_graph(name, model):
        from tensorflow.keras.utils import plot_model
        print("Saving Model Plot")
        plot_model(model, to_file=createModelPath(name, sub="model_plot.png"), show_shapes=True, show_layer_names=True, expand_nested=True)

    @staticmethod
    def save_model_history(name, history):
        import pandas as pd
        pd.DataFrame.from_dict(history).to_csv(createModelPath(name, sub="model_history.txt"), index=False, sep="\t")
