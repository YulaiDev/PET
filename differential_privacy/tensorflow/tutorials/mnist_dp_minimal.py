import numpy as np
import tensorflow as tf
import dp_accounting
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

# Hyperparameters
batch_size = 250
learning_rate = 0.15
noise_multiplier = 0.5
l2_norm_clip = 1.0
epochs = 5
microbatches = 250  #  Deze staat nu op de juiste plek

# Load and preprocess MNIST
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()
train_data = train_data.astype(np.float32) / 255.
test_data = test_data.astype(np.float32) / 255.
train_data = train_data[..., tf.newaxis]
test_data = test_data[..., tf.newaxis]
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 8, strides=2, padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 1),
    tf.keras.layers.Conv2D(32, 4, strides=2, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Define optimizer and loss
optimizer = DPKerasSGDOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=microbatches,
    learning_rate=learning_rate
)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Train model
model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_data, test_labels))

# Compute privacy budget ε
def compute_epsilon(steps):
    if noise_multiplier == 0.0:
        return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    accountant = dp_accounting.rdp.RdpAccountant(orders)
    sampling_prob = batch_size / 60000
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            sampling_prob,
            dp_accounting.GaussianDpEvent(noise_multiplier)
        ), steps
    )
    accountant.compose(event)
    return accountant.get_epsilon(1e-5)

steps = (60000 // batch_size) * epochs
epsilon = compute_epsilon(steps)
print(f'Privacy budget: ε = {epsilon:.2f} for δ = 1e-5')
