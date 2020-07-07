import os
import shutil
from builtins import staticmethod

from src.models.ModelStats import ModelStats
from src.MuraLoader import MuraLoader
from src.seeded import tf
from tensorflow.keras import backend as K
import gc
from src.models.tf_utils import metrics, callbacks, createModelPath
from os import path


class ModelRunner(object):
    def __init__(self, muraLoader: MuraLoader):
        self._muraLoader = muraLoader

    def clean_up(model):
        K.clear_session()
        del model
        gc.collect()

    def run(self, model: tf.keras.models.Model, name=None, epochs=100, verbose=1, overwrite=False, weight_classes=False,
            batch_size=32, augment=False, body_part=None):

        if not name:
            name = model.name

        run_name = []

        if weight_classes:
            run_name.append("wc")

        if augment:
            run_name.append("aug")

        if body_part:
            run_name.append(body_part)

        if len(run_name) > 0:
            name = "__".join([name, "_".join(run_name)])

        modelPath = createModelPath(name)

        print(name)
        print(modelPath)

        if path.exists(modelPath):
            if overwrite:
                shutil.rmtree(modelPath)
            else:
                best_model_name = ModelRunner.load_model(name)
                best_model_path = createModelPath(name, sub=best_model_name)
                best_model = tf.keras.models.load_model(best_model_path)
                history = ModelRunner.load_model_history(name)
                evaluation = ModelRunner.load_evaluation(name)
                return ModelStats(best_model, history, evaluation)
        os.makedirs(modelPath)

        model = self.__compile(model)
        ModelRunner.save_model_summary(name=name, model=model)
        ModelRunner.save_model_graph(name=name, model=model)

        if verbose == 1:
            print(model.summary())

        train_set, validation_set, test_set = self._muraLoader.get_sets(0.2, body_part=body_part)
        train_generator, validation_generator, test_generator = self._muraLoader.get_generators(
            train_set,
            validation_set,
            test_set,
            batch_size=batch_size,
            augment=augment
        )

        class_weights = None
        if weight_classes:
            from sklearn.utils import class_weight
            import numpy as np
            class_weights = dict(enumerate(class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_set['StudyLabel']), y=train_set['StudyLabel'])))
            print(f"fitting with class_weights: {class_weights}")

        history = model.fit(train_generator, validation_data=validation_generator, epochs=epochs, callbacks=callbacks(model=name), verbose=verbose, class_weight=class_weights)

        ModelRunner.save_training_plots(name=name, history=history.history)
        ModelRunner.save_model_history(name=name, history=history.history)

        evaluation = model.evaluate(test_generator)
        ModelRunner.save_evaluation(name=name, metric_names=model.metrics_names, evaluation=evaluation)
        ModelRunner.clean_up(model)

        evaluation = ModelRunner.load_evaluation(name)
        history = ModelRunner.load_model_history(name)
        return ModelStats(model, history, evaluation)

    def __compile(self, model: tf.keras.models.Model):
        adam = tf.optimizers.Adam()
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        model.compile(adam, loss=loss, metrics=[metrics()])

        return model

    @staticmethod
    def save_evaluation(name, metric_names, evaluation):
        import pandas as pd
        eval_df = pd.DataFrame(dict(zip(metric_names, evaluation)), index=[0])
        eval_df.to_csv(createModelPath(name, "evaluation.tsv"), index=False, sep="\t")

    @staticmethod
    def load_evaluation(name):
        import pandas as pd
        return pd.read_csv(createModelPath(name, "evaluation.tsv"), sep="\t")

    @staticmethod
    def save_training_plots(name, history):
        from src.visualize import plot_history
        fig = plot_history({"model": history})
        fig.savefig(createModelPath(name, sub="training_graph.png"))

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
        historyDf = pd.DataFrame.from_dict(history)
        historyDf.to_csv(createModelPath(name, sub="model_history.tsv"), index=False, sep="\t")
        return historyDf

    @staticmethod
    def load_model_history(name):
        import pandas as pd
        df = pd.read_csv(createModelPath(name, sub="model_history.tsv"), sep="\t")
        return df

    @staticmethod
    def load_model(name):
        saved_models = [f for f in os.listdir(createModelPath(name)) if f.endswith("h5")]
        saved_models.sort(reverse=True)
        # print(saved_models)
        best_model = saved_models[0]
        return best_model

    @staticmethod
    def plot_histories(models={}, figsize=(20, 15)):
        import pandas as pd
        from src.visualize import plot_history

        histories = {}

        for plot_name, model_name in models.items():
            print(plot_name, model_name)
            df = pd.read_csv(createModelPath(model_name, sub="model_history.tsv"), sep="\t")
            history = df.to_dict(orient="list")
            histories[plot_name] = history

        return plot_history(histories, figsize=figsize)

    @staticmethod
    def fetch_best_epochs(models={}):
        import pandas as pd
        histories = {}

        for plot_name, model_name in models.items():
            print(plot_name, model_name)
            df = pd.read_csv(createModelPath(model_name, sub="model_history.tsv"), sep="\t")
            histories[plot_name] = df[df['val_loss'] == min(df['val_loss'])]\
                .reset_index()\
                .rename(columns={'index': 'epoch'})\
                .to_dict(orient="list")

        return pd.DataFrame.from_dict(histories, orient="index")

    @staticmethod
    def fetch_evaluations(models={}):
        import pandas as pd

        evaluations = {}

        for stat_name, model_name in models.items():
            print(stat_name, model_name)
            df = pd.read_csv(createModelPath(model_name, sub="evaluation.tsv"), sep="\t")
            evaluation = df.to_dict(orient="list")
            evaluations[stat_name] = evaluation

        return pd.DataFrame.from_dict(evaluations, orient="index")
