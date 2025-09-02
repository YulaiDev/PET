import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
import dp_accounting

# Hyperparameters
batch_size = 250
microbatches = 250
learning_rate = 0.15
noise_multiplier = 0.1
l2_norm_clip = 1.0
epochs = 10

def compute_epsilon(steps):
    if noise_multiplier == 0.0:
        return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    accountant = dp_accounting.rdp.RdpAccountant(orders)
    sampling_probability = batch_size / 60000
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            sampling_probability,
            dp_accounting.GaussianDpEvent(noise_multiplier)), steps)
    accountant.compose(event)
    return accountant.get_epsilon(target_delta=1e-5)

# Load and preprocess data
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()
train_data = train_data.astype(np.float32) / 255
test_data = test_data.astype(np.float32) / 255
train_data = train_data.reshape((-1, 28, 28, 1))
test_data = test_data.reshape((-1, 28, 28, 1))
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 8, strides=2, padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 1),
    tf.keras.layers.Conv2D(32, 4, strides=2, padding='valid', activation='relu'),
    tf.keras.layers.MaxPool2D(2, 1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10)
])

# DP optimizer
optimizer = DPKerasSGDOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=microbatches,
    learning_rate=learning_rate
)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(test_data, test_labels))

# Report epsilon
eps = compute_epsilon(epochs * 60000 // batch_size)
print(f"For delta=1e-5, the current epsilon is: {eps:.2f}")
