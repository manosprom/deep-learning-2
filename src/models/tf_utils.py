import os

from src.seeded import tf, tfa


# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
def earlyStopping(patience=10, verbose=0):
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_kappa",
        patience=patience,
        mode="max",
        restore_best_weights=True,
        verbose=verbose
    )


# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau
def reduceLROnPlateau(patience=5, verbose=0):
    return tf.keras.callbacks.ReduceLROnPlateau(
        patience=patience,
        cooldown=2,
        factor=0.2,
        min_lr=0,
        verbose=verbose
    )


# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
def checkPointer(model, verbose=0):
    checkPointPath = createModelPath(model, sub="weights.{epoch:02d}-{val_loss:.4f}-{val_auc:.4f}-{val_accuracy:.4f}-{val_kappa:.4f}.h5")
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkPointPath,
        save_weights_only=False,
        monitor='val_kappa',
        mode='max',
        save_best_only=True,
        verbose=verbose
    )

    return checkpointer


# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
def tensorBoard(model):
    tensorBoardPath = createModelPath(model, sub="tensorboard")
    tensorBoard = tf.keras.callbacks.TensorBoard(
        log_dir=tensorBoardPath,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        update_freq='epoch',
        profile_batch=2,
        embeddings_freq=0,
        embeddings_metadata=None
    )
    return tensorBoard


def csvLogger(model):
    return tf.keras.callbacks.CSVLogger(
        filename=createModelPath(model, sub="logs.csv"),
        separator="\t",
        append=True
    )


def createModelPath(model, sub= None):
    path = f"/../../models/{model}/"
    if sub:
        path += f"{sub}"
    return os.path.abspath(os.path.dirname(__file__) + path)


def metrics():
    return [
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryAccuracy("accuracy"),
        tfa.metrics.CohenKappa(name="kappa", num_classes=2)
    ]


def callbacks(model="model", verbose=1):
    return [
        earlyStopping(verbose=verbose),
        reduceLROnPlateau(verbose=verbose),
        tensorBoard(model=model),
        csvLogger(model=model),
        checkPointer(model=model, verbose=verbose),
    ]
