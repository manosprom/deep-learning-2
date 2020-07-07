from src.seeded import tf


def custom_cnn_builder(
        shape=(256, 256, 3),
        starting_filters=32,
        conv_layers=1,
        conv_strides=(1, 1),
        conv_kernel=(3, 3),
        convs_per_layer=1,
        batch_norm=False,
        pooling="max",
        dropout=None,
        pool_size=(2, 2),
        pool_strides=(2, 2),
        last_pooling=None,
        spatial_dropout=None,
        last_dropout=None,
):
    name = []

    inputs = tf.keras.layers.Input(shape=shape, name="input")

    layer = inputs

    name.append("cnn")
    name.append("in." + ".".join(map(str, shape)))
    name.append(f"cl.{str(conv_layers)}.{str(convs_per_layer)}")
    name.append(f"f.{str(starting_filters)}")
    name.append("ck." + ".".join(map(str, conv_kernel)))

    if batch_norm:
        name.append("bn")

    if pooling:
        name.append("p." + str(pooling))
        name.append("ps." + ".".join(map(str, pool_size)))
        name.append("pstr." + ".".join(map(str, pool_strides)))

    if dropout:
        name.append("dp." + str(dropout))

    if spatial_dropout:
        name.append("spdp." + str(spatial_dropout))

    if last_pooling:
        name.append(f"lp.{last_pooling}")

    if last_dropout:
        name.append(f"ld.{last_dropout}")

    for conv_level in range(conv_layers):
        current_filters = starting_filters * (2 ** conv_level)
        for conv_number in range(convs_per_layer):
            layer = tf.keras.layers.Conv2D(filters=current_filters, kernel_size=conv_kernel, strides=conv_strides, name=f"conv_{conv_level}_{conv_number}", padding='same')(layer)
            if batch_norm:
                layer = tf.keras.layers.BatchNormalization(name=f"bn_{conv_level}_{conv_number}")(layer)
            layer = tf.keras.layers.Activation("relu", name=f"conv_{conv_level}_{conv_number}_relu")(layer)

        if spatial_dropout:
            layer = tf.keras.layers.SpatialDropout2D(spatial_dropout, name=f"sp_dropout_{conv_level}")(layer)

        if pooling == 'avg':
            layer = tf.keras.layers.AvgPool2D(pool_size=pool_size, strides=pool_strides, name=f"mp_{conv_level}", padding='same')(layer)
        elif pooling == 'max':
            layer = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_strides, name=f"mp_{conv_level}", padding='same')(layer)

        if dropout:
            layer = tf.keras.layers.Dropout(dropout, name=f"dropout_{conv_level}")(layer)

    if last_pooling == "avg":
        layer = tf.keras.layers.GlobalAveragePooling2D(name=f"lp_{last_pooling}")(layer)
    elif last_pooling == "max":
        layer = tf.keras.layers.GlobalMaxPooling2D(name=f"lp_{last_pooling}")(layer)

    layer = tf.keras.layers.Flatten(name="flatten")(layer)

    if last_dropout:
        layer = tf.keras.layers.Dropout(last_dropout, name="last_dp")(layer)

    layer = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(layer)

    name = "custom/" + "_".join(name)
    return tf.keras.models.Model(inputs, layer, name=name)


def resnet152v2_builder(pooling="max", shape=(256, 256, 3), trainable_layers_after=None):
    name = []

    name.append("resnet152v2")
    name.append("in." + ".".join(map(str, shape)))
    name.append(f"p.{pooling}")

    if trainable_layers_after:
        name.append(f"tla.{trainable_layers_after}")

    resNet = tf.keras.applications.ResNet152V2(
        include_top=False,
        weights='imagenet',
        input_shape=shape,
        pooling=pooling
    )

    network_layers = len(resNet.layers)
    print(f"Number of layers in network {network_layers}")

    if trainable_layers_after:
        for layer in resNet.layers[:trainable_layers_after]:
            layer.trainable = False
    else:
        resNet.trainable = False

    prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="resnet_output_sigmoid")

    model = tf.keras.Sequential(
        layers=[
            resNet,
            prediction_layer
        ],
        name="resnet152v2"
    )

    return model


def densenet201_builder(
        pooling="avg",
        shape=(256, 256, 3),
        trainable_layers_after=None,
        mlp=None,
        mlp_dropout=0.25
):
    name = []

    name.append("densenet201")
    name.append("in." + ".".join(map(str, shape)))
    name.append(f"p.{pooling}")

    if trainable_layers_after:
        name.append(f"tla.{trainable_layers_after}")

    if mlp:
        name.append(f"m.{'.'.join(map(str, mlp))}")
        name.append(f"mdp.{mlp_dropout}")

    denseNet = tf.keras.applications.DenseNet201(
        include_top=False,
        weights='imagenet',
        input_shape=shape,
        pooling=pooling
    )

    network_layers = len(denseNet.layers)
    print(f"Number of layers in network {network_layers}")

    if trainable_layers_after:
        for layer in denseNet.layers[:trainable_layers_after]:
            layer.trainable = False
    else:
        denseNet.trainable = False

    output = denseNet.output

    for index, mlp_neurons in enumerate(mlp):
        print(index, mlp_neurons, mlp_dropout)
        output = tf.keras.layers.Dense(mlp_neurons, activation="relu", name=f"m.{index}.{mlp_neurons}")(output)
        if mlp_dropout:
            output = tf.keras.layers.Dropout(mlp_dropout, name=f"mdp.{index}.{mlp_neurons}")(output)

    output = tf.keras.layers.Dense(1, activation="sigmoid", name="densenet_output_sigmoid")(output)

    name = "densenet201/" + "_".join(name)

    print(denseNet.input)
    print(output)

    model = tf.keras.models.Model(denseNet.input, output, name=name)

    # model = tf.keras.Sequential(layers=[
    #     denseNet,
    #     prediction_layer
    # ], name=name)

    return model

def custom_cnn_feed_forward_builder(
        shape=(256, 256, 3),
        starting_filters=32
):
    inputs = tf.keras.layers.Input(shape=shape, name="input")

    input_reduction = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(inputs)
    input_reduction = tf.keras.layers.BatchNormalization()(input_reduction)
    input_reduction = tf.keras.layers.Activation("relu")(input_reduction)
    input_reduction = tf.keras.layers.MaxPool2D(padding="same")(input_reduction)

    block1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(input_reduction)
    block1 = tf.keras.layers.BatchNormalization()(block1)
    block1 = tf.keras.layers.Activation("relu")(block1)
    block1 = tf.keras.layers.MaxPool2D(padding="same")(block1)

    block2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(block1)
    block2 = tf.keras.layers.BatchNormalization()(block2)
    block2 = tf.keras.layers.Activation("relu")(block2)
    block2 = tf.keras.layers.MaxPool2D(padding="same")(block2)

    block_concatenation = tf.keras.layers.Concatenate([block1, block2])

    block_concatenation = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(block_concatenation)

    name = "ff/test"
    return tf.keras.models.Model(inputs, block_concatenation, name=name)