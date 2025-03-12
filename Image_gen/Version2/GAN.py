import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]

# One-hot encode labels
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)


# Training parameters
BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 100  # You can adjust this
noise_dim = 100  # You can adjust this
num_examples_to_generate = 16
gp_weight = 10.0
num_classes = 10  # MNIST has 10 classes (digits 0-9)
d_steps = 3 #Number of steps to train discriminator.


# Create training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE) #Drop remainder is important!


# Create TensorBoard logger
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# Generator model
def build_conditional_generator(noise_dim, num_classes):
    noise_input = tf.keras.layers.Input(shape=(noise_dim,))
    label_input = tf.keras.layers.Input(shape=(num_classes,))

    # Process label (e.g., with an Embedding layer)
    label_embedding = tf.keras.layers.Dense(7*7)(label_input)  # Match initial dense layer
    label_reshape = tf.keras.layers.Reshape((7,7,1))(label_embedding)

    # Process noise
    noise_dense = tf.keras.layers.Dense(7*7*256, use_bias=False)(noise_input)
    noise_reshape = tf.keras.layers.Reshape((7, 7, 256))(noise_dense)
    noise_bn = tf.keras.layers.BatchNormalization()(noise_reshape)
    noise_relu = tf.keras.layers.ReLU()(noise_bn)

    # Concatenate noise and label embedding
    merged_input = tf.keras.layers.Concatenate()([noise_relu, label_reshape]) # shape (7,7,257)

    # ... rest of the generator ...
    x = tf.keras.layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False)(merged_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    output = tf.keras.layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh')(x)

    return tf.keras.Model([noise_input, label_input], output)


# Discriminator model
def build_conditional_discriminator(num_classes):
    image_input = tf.keras.layers.Input(shape=(28, 28, 1))
    label_input = tf.keras.layers.Input(shape=(num_classes,))

    # Process label 
    label_embedding = tf.keras.layers.Dense(28*28)(label_input)
    label_reshape = tf.keras.layers.Reshape((28, 28, 1))(label_embedding)


    # Concatenate image and label
    merged_input = tf.keras.layers.Concatenate()([image_input, label_reshape]) # (28,28,2)

    # Process image (Conv2D layers, as in the original)
    x = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(merged_input)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(256, (5,5), strides=(2,2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
        
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model([image_input, label_input], output)


# Initialize models and optimizers
generator = build_conditional_generator(noise_dim, num_classes)
discriminator = build_conditional_discriminator(num_classes)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.9)

# Training metrics
gen_loss_metric = tf.keras.metrics.Mean(name='gen_loss')
disc_loss_metric = tf.keras.metrics.Mean(name='disc_loss')
gp_metric = tf.keras.metrics.Mean(name='gradient_penalty')

# Gradient penalty implementation
def gradient_penalty(batch_size, real_images, fake_images, labels):
    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * real_images + (1 - alpha) * fake_images

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator([interpolated, labels], training=True)

    gradients = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    return tf.reduce_mean((norm - 1.0) ** 2)

@tf.function
def train_step(images, labels):
    batch_size = tf.shape(images)[0]

    # Train discriminator multiple times
    for _ in range(d_steps):
        noise = tf.random.normal([batch_size, noise_dim])
        with tf.GradientTape() as d_tape:
            fake_images = generator([noise, labels], training=True)
            real_output = discriminator([images, labels], training=True)
            fake_output = discriminator([fake_images, labels], training=True)

            # Calculate losses
            d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
            gp = gradient_penalty(batch_size, images, fake_images, labels)
            d_total_loss = d_loss + gp_weight * gp

        d_gradients = d_tape.gradient(d_total_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

        # Update metrics
        disc_loss_metric.update_state(d_total_loss)
        gp_metric.update_state(gp)

    # Train generator
    noise = tf.random.normal([batch_size, noise_dim])
    # Generate random labels for the generator during training.
    gen_labels = tf.one_hot(tf.random.uniform([batch_size], minval=0, maxval=num_classes, dtype=tf.int32), depth=num_classes)

    with tf.GradientTape() as g_tape:
        fake_images = generator([noise, gen_labels], training=True)
        fake_output = discriminator([fake_images, gen_labels], training=True)
        g_loss = -tf.reduce_mean(fake_output)

    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    gen_loss_metric.update_state(g_loss)


# Image generation function
def generate_and_save_images(model, epoch, noise, labels, save_dir='generated_images'):
    predictions = model([noise, labels], training=False)
    plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.title(f"Label: {tf.argmax(labels[i]).numpy()}")  # Display the label
        plt.axis('off')

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, f'image_epoch_{epoch:04d}.png'))
    plt.show()


# Training loop
def train(dataset, epochs):
    # Generate consistent noise and labels for visualization
    seed_noise = tf.random.normal([num_examples_to_generate, noise_dim])
    seed_labels = tf.one_hot(np.arange(num_examples_to_generate) % num_classes, depth=num_classes)  # Cycle through labels

    for epoch in range(epochs):
        # Reset metrics each epoch
        gen_loss_metric.reset_states()
        disc_loss_metric.reset_states()
        gp_metric.reset_states()

        for image_batch, label_batch in dataset:
            train_step(image_batch, label_batch)

        # Print metrics
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Generator Loss: {gen_loss_metric.result():.4f}')
        print(f'Discriminator Loss: {disc_loss_metric.result():.4f}')
        print(f'Gradient Penalty: {gp_metric.result():.4f}\n')

        # TensorBoard logging
        with train_summary_writer.as_default():
            tf.summary.scalar('gen_loss', gen_loss_metric.result(), step=epoch)
            tf.summary.scalar('disc_loss', disc_loss_metric.result(), step=epoch)
            tf.summary.scalar('gradient_penalty', gp_metric.result(), step=epoch)

        # Save images every 10 epochs
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, seed_noise, seed_labels)

# Start training
train(train_dataset, EPOCHS)

# Generate and save images after training is complete using specific labels:
final_noise = tf.random.normal([num_examples_to_generate, noise_dim])
# Example:  Generate two of each digit (0-9), total 16, repeating last 4 digits.
desired_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])

final_labels = tf.one_hot(desired_labels, depth=num_classes)
generate_and_save_images(generator, EPOCHS, final_noise, final_labels, save_dir='final_generated_images') #Save to different folder.
