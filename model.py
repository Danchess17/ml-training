from keras import Input, Model
from keras.layers import *


def MyUnet(img_size=(500, 500, 3), num_classes=7):
    inputs = Input(shape=img_size)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = Conv2D(32, 3, strides=2, padding="same", name="entry_conv2d")(inputs)
    x = BatchNormalization(name="entry_bn")(x)
    x = Activation("relu", name="entry_act")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = Activation("relu", name=str(filters) + "_act_1")(x)
        x = SeparableConv2D(filters, 3, padding="same", name=str(filters) + "_sep_1")(x)
        x = BatchNormalization(name=str(filters) + "_bn_1")(x)

        x = Activation("relu", name=str(filters) + "_act_2")(x)
        x = SeparableConv2D(filters, 3, padding="same", name=str(filters) + "_sep_2")(x)
        x = BatchNormalization(name=str(filters) + "_bn_2")(x)

        x = MaxPooling2D(3, strides=2, padding="same", name=str(filters) + "_pool")(x)

        # Project residual
        residual = Conv2D(filters, 1, strides=2, padding="same", name=str(filters) + "_conv2d")(
            previous_block_activation
        )
        x = add([x, residual], name=str(filters) + "_add")  # Add back residual
        previous_block_activation = x  # Set aside next residual


    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = Activation("relu", name=str(filters) + "_act_3")(x)
        x = Conv2DTranspose(filters, 3, padding="same", name=str(filters) + "_trans_1")(x)
        x = BatchNormalization(name=str(filters) + "_bn_3")(x)

        x = Activation("relu", name=str(filters) + "_act_4")(x)
        x = Conv2DTranspose(filters, 3, padding="same", name=str(filters) + "_trans_2")(x)
        x = BatchNormalization(name=str(filters) + "_bn_4")(x)

        x = UpSampling2D(2, name=str(filters) + "_up")(x)
        # Project residual
        residual = UpSampling2D(2, name=str(filters) + "_up_res")(previous_block_activation)
        residual = Conv2D(filters, 1, padding="same", name=str(filters) + "_conv2d_res")(residual)
        x = add([x, residual], name=str(filters) + "_add_res")  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = Conv2D(num_classes, 3, activation="softmax", padding="same", name="second_conv2d")(x)
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    return model
