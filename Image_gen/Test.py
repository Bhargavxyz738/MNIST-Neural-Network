import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the trained generator model
generator = tf.keras.models.load_model('generator_model.keras')

# Define the noise dimension (must match the training configuration)
noise_dim = 100

# Create a random noise vector for image generation
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Generate images using the loaded generator
generated_images = generator(seed, training=False)

# Display the generated images
fig = plt.figure(figsize=(4, 4))
for i in range(generated_images.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')
plt.show()
