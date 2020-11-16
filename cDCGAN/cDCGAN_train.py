import tensorflow as tf
import numpy as np
import time
import os
import sys
from cDCGAN_models import generator_model, discriminator_model, gan_model
from cDCGAN_data_prep import preprocess_data
from cDCGAN_utils import generate_and_save_images, save_diagnostics_to_file

"""
argv[0] script_name.py
argv[1] number of epochs
argv[2] batch size
argv[3] optimizer used in discriminator: Stochastic Gradient Descent ("sgd") and Adam ("adam")
argv[4] prefix you want to name all output files (losses, models, etc.)
"""


def main(argv=None):
    # Load Mammography (images) dataset including their respective labels (one-hot-encoded)
    # Images are represented as greyscale using RGB-values
    # Make sure directories are correct: Directory paths can be changed in script "cDCGAN_data_prep.py"
    X_train, X_test, y_train, y_test = preprocess_data()
    # Set Hyper-parameters for training the cDCGAN
    buffer_size = len(X_train) + 1  # Shuffle training data, adding 1 enables uniform shuffle
    print(len(X_train))             # (every random permutation is equally likely to occur)
    EPOCHS = argv[1]                # Number of epochs of training)
    batch_size = argv[2]            # Split training set (real images and respective labels) into batches
    disc_optimizer = argv[3]        # Optimizer for discriminator: 'adam' or 'sgd'
    labels_state = argv[4]          # State of the labels: 'hard' or 'soft'
    name = argv[5]                  # Give output files (models, diagnostic text file with losses / accuracy) a name
    dim_noise_z = 100               # Size of latent space (noise z) used to map fake mammography images

    # Create target Directory if don't exist
    if not os.path.exists(name):
        os.mkdir(name)
        print("Directory ", name, " Created ")
    else:
        print("Directory ", name, " already exists")

    # Use tf.data.Dataset.from_tensor_slices to shuffle data (uniformly) and create an tensor object which holds all
    # batches containing the image data and their respective labels for given batch_size
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=buffer_size).batch(
        batch_size=batch_size)
    """
    # The following for-loop print the data (RGB-values) for each batch and their respective labels.
    for image_batch, label_batch in train_data.take(5):
        #print(image_batch)
        tf.print(label_batch)
    """

    # Define cDCGAN: Generator and Discriminator
    # More information about the architectures can be found in python script cDCGAN_models.py
    generator = generator_model()
    # print(generator.summary())
    discriminator = discriminator_model()
    # print(discriminator.summary())


    # Set up optimizer for analysis:
    # Using SGD may tune down the discriminator in detecting fake labels
    if disc_optimizer == 'adam':
        discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                              metrics=['acc'])
    elif disc_optimizer == 'sgd':
        discriminator.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                              metrics=['acc'])

    # Defining the gan model, which trains the generator. Trainable parameters of discriminator are not trained.
    gan = gan_model(g_model=generator, d_model=discriminator)

    # List to store discriminator and generator losses, accuracies, and respective time stamps.
    diagnostic_info = []

    # Training the conditional DCGAN
    # Loop through number of epochs
    for epoch in range(EPOCHS):
        print("Current epoch: {}".format(epoch+1))
        start = time.time()
        train_step = 1

        # Random input and labels to create images for current epoch
        # Used to exam the current performance of the model and training progress
        seed = tf.random.normal([batch_size, dim_noise_z])
        seed_ints = np.random.randint(0, 3, batch_size)
        # generate one_hot_encoding
        seed_labels = tf.one_hot(indices=seed_ints, depth=3, dtype=tf.float32)

        # List to store losses, accuracies, time stamps per epoch.
        diagnostics_per_epoch = []

        # For-loop to train the images
        for image_batch, label_batch in train_data:
            print('Current training step: ', train_step)

            # Generate tensor holding specific number of (batch_size) latent vectors of certain dimension (dim_noise_z) for
            # fake image generation
            noise_z = tf.random.normal([batch_size, dim_noise_z])
            # Generate randomly integers to be used for one-hot-encoding of fake labels as input for generator
            # Classes are as following: Normal --> 0 --> [1,0,0]; Benign --> 1 --> [0,1,0]; Malignant --> 2 --> [0,0,1]
            fake_labels_as_int = np.random.randint(low=0, high=3, size=batch_size)
            fake_labels = tf.one_hot(indices=fake_labels_as_int, depth=3, dtype=tf.float32)

            # Generating a set of fake images
            print("Generate fake images")
            fake_images = generator.predict([noise_z, fake_labels])

            # generate labels to mark real images as real
            print("Real and Fake loss")

            # Model tuning: hard labels and soft labels --> soft labels provide more uncertainty to discriminator
            if labels_state == 'soft':
                y_real = tf.random.uniform(shape=[batch_size], minval=0.7, maxval=1.2)
                y_fake = tf.random.uniform(shape=[batch_size], minval=0, maxval=0.3)
            elif labels_state == 'hard':
                y_real = tf.ones([batch_size], dtype=tf.float32)
                y_fake = tf.zeros([batch_size], dtype=tf.float32)

            # Compute the losses of the discriminator for
            # batch with real images and real labels
            discriminator.trainable = True
            disc_real_loss = discriminator.train_on_batch([image_batch, label_batch], y_real)
            # batch with fake images and fake labels
            disc_fake_loss = discriminator.train_on_batch([fake_images, fake_labels], y_fake)

            print("Generator loss")
            y_real_gen = tf.ones([batch_size], dtype=tf.float32)
            gan_loss = gan.train_on_batch([noise_z, fake_labels], y_real_gen)

            # Storing batch metrics in list
            time_stamp = time.time() - start
            diagnostics_per_batch = [disc_real_loss, disc_fake_loss, gan_loss, time_stamp]
            # appending batch metrics to epoch metrics
            diagnostics_per_epoch.append(diagnostics_per_batch)

            train_step += 1

        if epoch % 10 == 0:
            # Call function to generate random images with noise z and fake labels to check training evolution of the cDCGAN
            generate_and_save_images(generator,
                                     epoch + 1,
                                     seed,
                                     seed_labels,
                                     name)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        print()
        diagnostic_info.append(diagnostics_per_epoch)

    # Calls function to write diagnostic information in text file in working directory
    save_diagnostics_to_file(name+'cDCGAN_diagnostics', diagnostic_info)

    # Saving models for reproduction in working directory
    generator.save(name+'cDCGAN_generator', save_format='h5')
    discriminator.save(name+'cDCGAN_discriminator', save_format='h5')
    gan.save(name+'cDCGAN_gan', save_format='h5')


if __name__ == '__main__':
    main(sys.argv)
