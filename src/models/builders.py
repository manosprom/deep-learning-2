from src.seeded import tf


def custom_cnn_builder(
        starting_filters=32,
        layers=1,
        convs_per_layer=1,
        batch_norm=False,
        shape=(256, 256, 3),
        pooling="max",
        dropout=None,
        pool_size=(2, 2),
        pool_strides=(2, 2)
):
    inputs = tf.keras.layers.Input(shape=shape, name="input")

    layer = inputs

    for conv_level in range(layers):
        current_filters = starting_filters * (2 ** conv_level)
        for conv_number in range(convs_per_layer):
            layer = tf.keras.layers.Conv2D(
                filters=current_filters,
                kernel_size=(3, 3),
                activation='relu',
                name=f"conv_{conv_level}_{conv_number}",
                padding='same'
            )(layer)
            if batch_norm:
                layer = tf.keras.layers.BatchNormalization(name=f"bn_{conv_level}_{conv_number}")(layer)

        if pooling == 'avg':
            layer = tf.keras.layers.AvgPool2D(
                pool_size=pool_size,
                strides=pool_strides,
                name=f"mp_{conv_level}",
                padding='same'
            )(layer)
        else:
            layer = tf.keras.layers.MaxPool2D(
                pool_size=pool_size,
                strides=pool_strides,
                name=f"mp_{conv_level}",
                padding='same'
            )(layer)

        if dropout:
            layer = tf.keras.layers.Dropout(dropout, name=f"dropout_{conv_level}")(layer)

    outputs = tf.keras.layers.Flatten(name="flatten")(layer)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(outputs)
    return tf.keras.models.Model(inputs, outputs, name="custom_cnn")


def resnet152v2_builder(pooling="max", shape=(256, 256)):
    base_model = tf.keras.applications.ResNet152V2(
        include_top=False,
        weights='imagenet',
        input_shape=shape,
        pooling=pooling
    )
    base_model.trainable = False

    prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="resnet_output_softmax_1")

    model = tf.keras.Sequential(
        layers=[
            base_model,
            prediction_layer
        ],
        name="resnet152v2"
    )

    return model


def densenet201_builder(pooling="max", shape=(256, 256, 3)):
    base_model = tf.keras.applications.DenseNet201(
        include_top=False,
        weights='imagenet',
        input_shape=shape,
        pooling=pooling
    )

    base_model.trainable = False

    prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="densenet_output_sigmoid_1")

    model = tf.keras.Sequential(layers=[
        base_model,
        prediction_layer
    ], name="densenet201")

    return model
