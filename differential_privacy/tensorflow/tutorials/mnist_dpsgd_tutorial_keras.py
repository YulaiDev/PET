# Copyright 2019, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training a CNN on MNIST with Keras and the DP SGD optimizer.""" #Train een CNN op MNIST met Keras en DP-SGD

from absl import app
from absl import flags
from absl import logging
import dp_accounting #voor berekenen van privacyverlies (epsilon)
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer #DP-SGD optimizer voor Keras

#HYPERPARAMETERS EN FLAGS
flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, ' #gebruik dp of niet?
    'train with vanilla SGD.') # gebruik DP of niet? True = met DP-SGD, False = gewone SGD
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training') #leer-snelheid
flags.DEFINE_float('noise_multiplier', 0.5, #hoeveel ruis je toevoegt per GRADIENT UPDATE
                   'Ratio of the standard deviation to the clipping norm')
# Clippen = begrenzen van de invloed van een datapunt. 
# voorkomt dat een outlier (extreem datapunt) een te grote impact heeft. Essentieel voor DP.

flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm') #wanneer je met dp traint, wil je voorkomen dat een enkel datapunt een te grote invloed heeft op de modelupdate. daarom clippen = grenzen hoeveel kracht of invloed een enkele training sample op het model heeft. Als je niet clipt kunnen data points met extreme waarden grote invloed hebben op het model.
flags.DEFINE_integer('batch_size', 250, 'Batch size') #hoeveel voorbeelden tegelijk
flags.DEFINE_integer('epochs', 20, 'Number of epochs') #hoe vaak de dataset volledig gezien wordt
flags.DEFINE_integer(
    'microbatches', 250, 'Number of microbatches '
    '(must evenly divide batch_size)') #batch wordt verdeeld in microbatches, moet exact deelbaar zijn
flags.DEFINE_string('model_dir', None, 'Model directory') #waar resultaten/checkpoints worden opgeslagen


FLAGS = flags.FLAGS #container met alle hyperparameters

#Privacy berekenen 
def compute_epsilon(steps):
  """Computes epsilon value for given hyperparameters."""
  if FLAGS.noise_multiplier == 0.0:
    return float('inf') #geen ruis = oneindig privacyverlies
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  accountant = dp_accounting.rdp.RdpAccountant(orders) #accountant = berekent privacyverlies
#berekenen van de privacy verlies waarde 
  sampling_probability = FLAGS.batch_size / 60000 #steekproefkans (MNIST heeft 60k voorbeelden)
  event = dp_accounting.SelfComposedDpEvent(
      dp_accounting.PoissonSampledDpEvent(#Poisson sampling = theoretische garantie voor DP
          dp_accounting.GaussianDpEvent(FLAGS.noise_multiplier)), steps)

  accountant.compose(event)

  # Delta = 1e-5 omdat MNIST 60.000 trainingpunten heeft
  return accountant.get_epsilon(target_delta=1e-5) #epsilon teruggeven

#data laden 
def load_mnist():
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.mnist.load_data() #laad mnist train en test
  train_data, train_labels = train
  test_data, test_labels = test
  # normaliseren van data tussen 0 en 1
  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255
  # reshapen zodat het vorm (28,28,1) heeft (grijswaardenbeeldjes)

  train_data = train_data.reshape((train_data.shape[0], 28, 28, 1))
  test_data = test_data.reshape((test_data.shape[0], 28, 28, 1))
  # labels converteren naar one-hot vectors (10 klassen)

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
  test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)
  # check of data correct geschaald is

  assert train_data.min() == 0.
  assert train_data.max() == 1.
  assert test_data.min() == 0.
  assert test_data.max() == 1.

  return train_data, train_labels, test_data, test_labels
#MAIN FUNCTIE

def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')
    #als dp=True, check dat batch_size deelbaar is door microbatches

  # Load training and test data.
  train_data, train_labels, test_data, test_labels = load_mnist()
#model gedifeneerd een kleine cnn 
  # Define a sequential Keras model
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(
          16,
          8,
          strides=2,
          padding='same',
          activation='relu',
          input_shape=(28, 28, 1)),# eerste conv-laag
      tf.keras.layers.MaxPool2D(2, 1),#pooling
      tf.keras.layers.Conv2D(
          32, 4, strides=2, padding='valid', activation='relu'),# tweede conv-laag
      tf.keras.layers.MaxPool2D(2, 1),#pooling
      tf.keras.layers.Flatten(),#vlak maken
      tf.keras.layers.Dense(32, activation='relu'),#dense laag
      tf.keras.layers.Dense(10)#output = 10 klassen (digits 0-9)
  ])
#Kies optimizer en loss afhankelijk van dp
  if FLAGS.dpsgd:
    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=FLAGS.l2_norm_clip, #clipnorm per voorbeeld
        noise_multiplier=FLAGS.noise_multiplier,#hoeveel ruis
        num_microbatches=FLAGS.microbatches, #splitsing batch
        learning_rate=FLAGS.learning_rate)
    # Compute vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)
    #        #reduction=NONE: per-voorbeeld loss nodig voor DP
  else:
    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate) #gewoon SGD
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True) #gemiddelde loss
  # Compile model with Keras
  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

  # Train model with Keras
  model.fit(
      train_data,
      train_labels,
      epochs=FLAGS.epochs,
      validation_data=(test_data, test_labels),
      batch_size=FLAGS.batch_size)

  # Compute the privacy budget expended.
  if FLAGS.dpsgd:
    eps = compute_epsilon(FLAGS.epochs * 60000 // FLAGS.batch_size)
    print('For delta=1e-5, the current epsilon is: %.2f' % eps)
  else:
    print('Trained with vanilla non-private SGD optimizer')


if __name__ == '__main__':
  app.run(main)
