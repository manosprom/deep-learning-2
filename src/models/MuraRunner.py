import os
import shutil

from src.MuraLoader import MuraLoader
from src.seeded import tf
from src.models.tf_utils import metrics, callbacks, createModelPath
from os import path


class MuraRunner(object):
    def __init__(self, muraLoader: MuraLoader):
        self._muraLoader = muraLoader

    # def clean_up(model):
    #     K.clear_session()
    #     del model
    #     gc.collect()

    def run(self, model: tf.keras.models.Model, epochs=5, verbose=1, overwrite=True):
        modelPath = createModelPath(model.name)
        if path.exists(modelPath):
            if overwrite:
                shutil.rmtree(modelPath)
            else:
                raise FileExistsError
        os.mkdir(modelPath)

        model = self.__compile(model)
        self.save_model_summary(model)
        self.save_model_grah(model)

        train_set, validation_set, test_set = self._muraLoader.get_sets(0.2)
        train_generator, validation_generator, test_generator = self._muraLoader.get_generators(train_set, validation_set, test_set, batch_size=32)

        model, history = self.__fit(model, train_generator, validation_generator, epochs=epochs, verbose=verbose)
        evaluation = model.evaluate(test_generator)
        return model, history, evaluation

    def __compile(self, model: tf.keras.models.Model):
        adam = tf.optimizers.Adam()
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        model.compile(adam, loss=loss, metrics=[metrics()])

        return model

    def __fit(self, model, train_generator, validation_generator, epochs=5, verbose=1):

        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks(model=model.name),
            verbose=verbose
        )
        return model, history

    def save_model_summary(self, model):
        from contextlib import redirect_stdout
        print("Saving Model Summary")

        with open(createModelPath(model.name, sub="model_summary.txt"), 'w') as f:
            with redirect_stdout(f):
                model.summary()

    def save_model_grah(self, model):
        from tensorflow.keras.utils import plot_model
        print("Saving Model Plot")
        plot_model(model, to_file=createModelPath(model.name, sub="model_plot.png"), show_shapes=True, show_layer_names=True, expand_nested=True)
