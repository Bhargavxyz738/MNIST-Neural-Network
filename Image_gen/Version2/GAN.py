import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Load and preprocess data
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]

# Training parameters
BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 100
noise_dim = 128
num_examples_to_generate = 16
gp_weight = 10.0

# Create training dataset
train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Create TensorBoard logger
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# Generator model
def build_generator():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(7*7*512, use_bias=False, input_shape=(noise_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Reshape((7, 7, 512)),
        
        tf.keras.layers.Conv2DTranspose(256, (5,5), strides=(1,1), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        
        tf.keras.layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        
        tf.keras.layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        
        tf.keras.layers.Conv2DTranspose(1, (5,5), strides=(1,1), padding='same', 
                                      use_bias=False, activation='tanh')
    ])

# Discriminator model
def build_discriminator():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[28,28,1]),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv2D(256, (5,5), strides=(2,2), padding='same'),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])

# Initialize models and optimizers
generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.9)

# Training metrics
gen_loss_metric = tf.keras.metrics.Mean(name='gen_loss')
disc_loss_metric = tf.keras.metrics.Mean(name='disc_loss')
gp_metric = tf.keras.metrics.Mean(name='gradient_penalty')

# Gradient penalty implementation
def gradient_penalty(batch_size, real_images, fake_images):
    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * real_images + (1 - alpha) * fake_images

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated)
    
    gradients = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
    return tf.reduce_mean((norm - 1.0) ** 2)

@tf.function
def train_step(images):
    batch_size = tf.shape(images)[0]
    
    # Train discriminator multiple times
    for _ in range(3):
        noise = tf.random.normal([batch_size, noise_dim])
        with tf.GradientTape() as d_tape:
            real_output = discriminator(images)
            fake_images = generator(noise, training=True)
            fake_output = discriminator(fake_images)
            
            # Calculate losses
            d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
            gp = gradient_penalty(batch_size, images, fake_images)
            d_total_loss = d_loss + gp_weight * gp
            
        d_gradients = d_tape.gradient(d_total_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
        
        # Update metrics
        disc_loss_metric.update_state(d_total_loss)
        gp_metric.update_state(gp)
    
    # Train generator
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as g_tape:
        fake_images = generator(noise, training=True)
        fake_output = discriminator(fake_images)
        g_loss = -tf.reduce_mean(fake_output)
        
    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    gen_loss_metric.update_state(g_loss)

# Image generation function
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig(f'image_epoch_{epoch:04d}.png')
    plt.show()

# Training loop
def train(dataset, epochs):
    # Generate random seed for consistent image comparison
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    
    for epoch in range(epochs):
        # Reset metrics each epoch
        gen_loss_metric.reset_states()
        disc_loss_metric.reset_states()
        gp_metric.reset_states()
        
        for image_batch in dataset:
            train_step(image_batch)
        
        # Print metrics
        print(f'Epoch {epoch+1}/{EPOCHS}')
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
            generate_and_save_images(generator, epoch + 1, seed)

# Start training
train(train_dataset, EPOCHS)
