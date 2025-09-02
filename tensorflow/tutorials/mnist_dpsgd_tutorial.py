# Copyright 2020, The TensorFlow Authors.
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
"""Train a CNN on MNIST with differentially private SGD optimizer."""

import time # Om epoch- tijden te meten 

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from tensorflow import estimator as tf_estimator #het trainings-/model-framework en de Estimator-AP
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib #rekent de actuele epsilon (ε) uit voor DP-SGD
from tensorflow_privacy.privacy.optimizers import dp_optimizer #bevat de DP-SGD optimizer
import mnist_dpsgd_tutorial_common as common #lokale hulpfuncties voor dit tutorial (o.a. het CNN-model en input-pipelines)

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.') #schakel om DP-SGD aan/uit te zetten (anders: gewone SGD)
flags.DEFINE_float('learning_rate', .15, 'Learning rate for training') #leer-snelheid (voor zowel DP-SGD als vanilla SGD)
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm') #verhouding σ/clipnorm — bepaalt de hoeveelheid ruis die voor DP wordt toegevoegd
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')#gradient clipping-norm (per voorbeeld/microbatch) voor DP-SGD
flags.DEFINE_integer('batch_size', 256, 'Batch size') #trainingsgrootte en aantal passes door data
flags.DEFINE_integer('epochs', 30, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 256, 'Number of microbatches '
    '(must evenly divide batch_size)') #aantal microbatches waar een batch in wordt gesplitst (moet batch_size exact delen) nodig voor per-microbatch loss/gradient berekening in DP
flags.DEFINE_string('model_dir', None, 'Model directory') #directory waar Estimator checkpoints en samenvattingen wegschrijft

FLAGS = flags.FLAGS #container die de geparste vlaggen bevat


def cnn_model_fn(features, labels, mode, params):  # pylint: disable=unused-argument standaard Estimator model_fn signatuur
  """Model function for a CNN."""

  # Define CNN architecture.
  logits = common.get_cnn_model(features) #bouwt en past een CNN toe op de input features; geeft logits terug

  # Calculate loss as a vector (to support microbatches in DP-SGD).
  vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits) #per-voorbeeld cross-entropy (nodig voor microbatch-DP)
  # Define mean of loss across minibatch (for reporting through tf.Estimator).
  scalar_loss = tf.reduce_mean(input_tensor=vector_loss) #gemiddelde over de batch (Estimator verwacht een scalar loss voor logging/eval)

  # Configure the training op (for TRAIN mode).
  if mode == tf_estimator.ModeKeys.TRAIN:
    if FLAGS.dpsgd:
      # Use DP version of GradientDescentOptimizer. Other optimizers are
      # available in dp_optimizer. Most optimizers inheriting from
      # tf.compat.v1.train.Optimizer should be wrappable in differentially
      # private counterparts by calling dp_optimizer.optimizer_from_args().
      optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
          l2_norm_clip=FLAGS.l2_norm_clip, #maximale L2-norm van de per-voorbeeld (of microbatch) gradient.
          noise_multiplier=FLAGS.noise_multiplier, #bepaalt de variantie van de Gaussische ruis die aan de som van geclipte gradients wordt toegevoegd
          num_microbatches=FLAGS.microbatches, #splitst je batch in kleine stukjes (microbatches)
          learning_rate=FLAGS.learning_rate) #stapgrootte van de GD-update (zoals bij gewone SGD)
      opt_loss = vector_loss
    else:
      optimizer = tf.compat.v1.train.GradientDescentOptimizer(
          learning_rate=FLAGS.learning_rate)
      opt_loss = scalar_loss # dit wordt gebruikt de gewone GradientDescentOptimizer die de gemiddelde batch-loss (scalar_loss) minimaliseert, dus standaard SGD zonder clipping en zonder ruis (geen differentiële privacy).

    global_step = tf.compat.v1.train.get_global_step()
    train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)

    # In the following, we pass the mean of the loss (scalar_loss) rather than
    # the vector_loss because tf.estimator requires a scalar loss. This is only
    # used for evaluation and debugging by tf.estimator. The actual loss being
    # minimized is opt_loss defined above and passed to optimizer.minimize().
    return tf_estimator.EstimatorSpec(
        mode=mode, loss=scalar_loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode).
  elif mode == tf_estimator.ModeKeys.EVAL: # Dit stuk wordt uitgevoerd als de Estimator in evaluatiemodus draait
    eval_metric_ops = {
        'accuracy':
            tf.compat.v1.metrics.accuracy(
                labels=labels, predictions=tf.argmax(input=logits, axis=1))
    }
    return tf_estimator.EstimatorSpec( #Gemiddelde loss van de batch
        mode=mode, loss=scalar_loss, eval_metric_ops=eval_metric_ops) #definieert welke metrics gerapporteerd worden
# Dit stuk zorgt dus dat je tijdens evaluatie loss + accuracy terugkrijgt

def main(unused_argv): #Dit is de hoofdfunctie die door app.run(main) wordt aangeroepen
  logging.set_verbosity(logging.INFO) #Zet het logniveau naar INFO zodat trainings- en evaluatie-informatie zichtbaar wordt.
  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size') # Als DP-SGD aanstaat, controleer dat batch_size exact deelbaar is door microbatches 

  # Instantiate the tf.Estimator.
  mnist_classifier = tf_estimator.Estimator( 
      model_fn=cnn_model_fn, model_dir=FLAGS.model_dir) #dit is de modeldefinitie die bepaalt hoe training, evaluatie en loss berekend worden.

  # Training loop.
  steps_per_epoch = 60000 // FLAGS.batch_size
  for epoch in range(1, FLAGS.epochs + 1):
    start_time = time.time()
    # Train the model for one epoch.
    mnist_classifier.train(
        input_fn=common.make_input_fn('train', FLAGS.batch_size),
        steps=steps_per_epoch)
    end_time = time.time()
    logging.info('Epoch %d time in seconds: %.2f', epoch, end_time - start_time)

    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(
        input_fn=common.make_input_fn('test', FLAGS.batch_size, 1))
    test_accuracy = eval_results['accuracy']
    print('Test accuracy after %d epochs is: %.3f' % (epoch, test_accuracy))

    # Compute the privacy budget expended.
    if FLAGS.dpsgd:
      if FLAGS.noise_multiplier > 0.0:
        eps, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
            60000, FLAGS.batch_size, FLAGS.noise_multiplier, epoch, 1e-5)
        print('For delta=1e-5, the current epsilon is: %.2f' % eps)
      else:
        print('Trained with DP-SGD but with zero noise.')
    else:
      print('Trained with vanilla non-private SGD optimizer')


if __name__ == '__main__':
  app.run(main)
