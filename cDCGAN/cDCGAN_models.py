import tensorflow as tf


# Architecture of generator and discriminator were inspired by the proposed DCGAN from the Progressive Growing of
# GANs https://research.nvidia.com/publication/2017-10_Progressive-Growing-of
# Model were build based on tensorflows Keras functional API
# https://www.tensorflow.org/guide/keras/functional
def generator_model():
    # Prepare noise input z
    input_z = tf.keras.layers.Input(shape=(100,))
    dense_z_1 = tf.keras.layers.Dense(256 * 8 * 8)(input_z)
    act_z_1 = tf.keras.layers.LeakyReLU(alpha=0.2)(dense_z_1)
    bn_z_1 = tf.keras.layers.BatchNormalization(momentum=0.9)(act_z_1)
    reshape_z = tf.keras.layers.Reshape(target_shape=(8, 8, 256), input_shape=(4 * 4 * 1024,))(bn_z_1)

    # prepare conditional (label) input c
    input_c = tf.keras.layers.Input(shape=(3,))
    dense_c_1 = tf.keras.layers.Dense(8 * 8 * 1)(input_c)
    act_c_1 = tf.keras.layers.LeakyReLU(alpha=0.2)(dense_c_1)
    bn_c_1 = tf.keras.layers.BatchNormalization(momentum=0.9)(act_c_1)
    reshape_c = tf.keras.layers.Reshape(target_shape=(8, 8, 1), input_shape=(4 * 4 * 1,))(bn_c_1)

    # concatenating noise z and label c
    concat_z_c = tf.keras.layers.Concatenate()([reshape_z, reshape_c])

    # Image generation
    # Upsampling to 8x8
    conv2D_1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(
        concat_z_c)
    act_conv2D_1 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2D_1)
    bn_conv2D_1 = tf.keras.layers.BatchNormalization(momentum=0.9)(act_conv2D_1)

    # Upsampling to 16x16
    conv2D_2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(
        bn_conv2D_1)
    act_conv2D_2 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2D_2)
    bn_conv2D_2 = tf.keras.layers.BatchNormalization(momentum=0.9)(act_conv2D_2)

    # Upsampling to 32x32
    conv2D_3 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(
        bn_conv2D_2)
    act_conv2D_3 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2D_3)
    bn_conv2D_3 = tf.keras.layers.BatchNormalization(momentum=0.9)(act_conv2D_3)

    # Upsampling to 64x64
    conv2D_4 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same')(
        bn_conv2D_3)
    act_conv2D_4 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2D_4)
    bn_conv2D_4 = tf.keras.layers.BatchNormalization(momentum=0.9)(act_conv2D_4)

    # Upsampling to 128x128
    conv2D_5 = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=(2, 2), padding='same')(
        bn_conv2D_4)
    act_conv2D_5 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2D_5)
    bn_conv2D_5 = tf.keras.layers.BatchNormalization(momentum=0.9)(act_conv2D_5)

    # Output layer
    conv2D_6 = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), activation='tanh', padding='same')(
        bn_conv2D_5)

    # Model output
    model = tf.keras.models.Model(inputs=[input_z, input_c], outputs=conv2D_6)
    return model


def discriminator_model():
    # prepare conditional (label) input c
    input_c = tf.keras.layers.Input(shape=(3,))
    dense_c_1 = tf.keras.layers.Dense(256 * 256 * 1)(input_c)
    act_c_1 = tf.keras.layers.LeakyReLU(alpha=0.2)(dense_c_1)
    reshape_c = tf.keras.layers.Reshape(target_shape=(256, 256, 1), input_shape=(256 * 256 * 1,))(act_c_1)

    # Get input images x: real p(x_r) or fake p(x_z)
    input_x = tf.keras.layers.Input(shape=(256, 256, 3))

    # Concatenate input c and image x
    concat_x_c = tf.keras.layers.Concatenate()([input_x, reshape_c])

    # Feature extraction for discriminating real from fake images
    # Begin feature extraction process
    # Downsampling: 16 feature maps
    conv2d_1 = tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', name='conv_512x512')(concat_x_c)
    #bn_conv2d_1 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv2d_1)
    act_conv2d_1 = tf.keras.layers.LeakyReLU(alpha=0.2, name='lReLU_512x512')(conv2d_1)
    dp_conv2d_1 = tf.keras.layers.Dropout(0.5, name='Dropout_512x512')(act_conv2d_1)

    # Downsampling: 32 feature maps
    conv2d_2 = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', name='conv_256x256')(dp_conv2d_1)
    #bn_conv2d_2 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv2d_2)
    act_conv2d_2 = tf.keras.layers.LeakyReLU(alpha=0.2, name='lReLU_256x256')(conv2d_2)
    dp_conv2d_2 = tf.keras.layers.Dropout(0.5, name='Dropout_256x256')(act_conv2d_2)

    # Downsampling: 64 feature maps
    conv2d_3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', name='conv_128x128')(dp_conv2d_2)
    #bn_conv2d_3 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv2d_3)
    act_conv2d_3 = tf.keras.layers.LeakyReLU(alpha=0.2, name='lReLU_128x128')(conv2d_3)
    dp_conv2d_3 = tf.keras.layers.Dropout(0.5, name='Dropout_128x128')(act_conv2d_3)

    # Downsampling: 128 feature maps
    conv2d_4 = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', name='conv_64x64')(dp_conv2d_3)
    #bn_conv2d_4 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv2d_4)
    act_conv2d_4 = tf.keras.layers.LeakyReLU(alpha=0.2, name='lReLU_64x64')(conv2d_4)
    dp_conv2d_4 = tf.keras.layers.Dropout(0.5, name='Dropout_64x64')(act_conv2d_4)

    # Downsampling: 256 feature maps
    conv2d_5 = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', name='conv_32x32')(dp_conv2d_4)
    #bn_conv2d_5 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv2d_5)
    act_conv2d_5 = tf.keras.layers.LeakyReLU(alpha=0.2, name='lReLU_32x32')(conv2d_5)
    dp_conv2d_5 = tf.keras.layers.Dropout(0.5, name='Dropout_32x32')(act_conv2d_5)

    # Downsampling: 256 feature maps
    conv2d_6 = tf.keras.layers.Conv2D(256, (5, 5), strides=(1, 1), padding='valid', name='conv_4x4')(dp_conv2d_5)
    #bn_conv2d_6 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv2d_6)
    act_conv2d_6 = tf.keras.layers.LeakyReLU(alpha=0.2, name='lReLU_4x4')(conv2d_6)
    dp_conv2d_6 = tf.keras.layers.Dropout(0.5, name='Dropout_4x4')(act_conv2d_6)

    # Downsampling: 256 feature maps
    conv2d_7 = tf.keras.layers.Conv2D(256, (4, 4), strides=(1, 1), padding='valid', name='conv_1x1')(dp_conv2d_6)
    #bn_conv2d_7 = tf.keras.layers.BatchNormalization(momentum=0.9)(conv2d_7)
    act_conv2d_7 = tf.keras.layers.LeakyReLU(alpha=0.2, name='lReLU_1x1')(conv2d_7)
    dp_conv2d_7 = tf.keras.layers.Dropout(0.5, name='Dropout_1x1')(act_conv2d_7)

    flat_output = tf.keras.layers.Flatten()(dp_conv2d_7)
    final_output = tf.keras.layers.Dense(units=1, activation='linear', name='final_output')(flat_output)
    #bn_final_output = tf.keras.layers.BatchNormalization(momentum=0.9)(final_output)

    model = tf.keras.models.Model(inputs=[input_x, input_c], outputs=final_output, name="Discriminator")
    return model


# define the combined generator and discriminator model, for updating the generator
# Source: https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
def gan_model(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input
    # get image output from the generator model
    gen_output = g_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = d_model([gen_output, gen_label])
    # define gan model as taking noise and label and outputting a classification
    model = tf.keras.models.Model([gen_noise, gen_label], gan_output)
    # compile model
    opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=opt, metrics=['acc'])
    return model
