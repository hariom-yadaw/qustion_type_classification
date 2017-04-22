#training

import tensorflow as tf
import numpy as np
import os
import datetime
import data_utils
from sentence_classification_cnn import Sentence_CNN
from tensorflow.contrib import learn
from tensorflow.contrib.tensorboard.plugins import projector
import ast

try:
    from ConfigParser import SafeConfigParser
except:
    from configparser import SafeConfigParser

# Parameters
# ==================================================

def get_config(config_file='config.ini'):
    parser = SafeConfigParser()
    parser.read(config_file)
    # get the ints, floats and strings
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    _conf_floats = [(key, float(value)) for key, value in parser.items('floats')]
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
    _conf_booleans = [(key, ast.literal_eval(value)) for key, value in parser.items('booleans')]
    return dict(_conf_ints + _conf_floats + _conf_strings + _conf_booleans)


gConfig = get_config(config_file='config.ini')
if gConfig['isdebug']:
    print('Applying Parameters: %d' % len(gConfig))
    # print(self.gConfig['data_dir'])
    for k, v in gConfig.iteritems():
        print('%s: %s' % (k, str(v)))
    print("")

# Data Preparation
# ==================================================
print("Loading data...")
x_text, y = data_utils.load_data_and_labels(gConfig['train_data_file'])

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
dev_sample_index = -1 * int(gConfig['dev_sample_percentage'] * float(len(y)))
# x_train = x_shuffled
# y_train = y_shuffled
# x_dev = x_shuffled[dev_sample_index:]
# y_dev = y_shuffled[dev_sample_index:]
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
print("sequence length: {:d}".format(x_train.shape[1]))
print("num classes: {:d}".format(y_train.shape[1]))


# Training
# ==================================================
# Output directory for models and summaries
# timestamp = str(int(time.time()))
# out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
out_dir = gConfig['model_dir']
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=gConfig['allow_soft_placement'],
      log_device_placement=gConfig['log_device_placement'])
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = Sentence_CNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=gConfig['embedding_dim'],
            filter_sizes=list(map(int, gConfig['filter_sizes'].split(","))),
            num_filters=gConfig['num_filters'], model_dir=out_dir,
            l2_reg_lambda=gConfig['l2_reg_lambda'])

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        # timestamp = str(int(time.time()))
        # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=gConfig['num_checkpoints'])

        #adding metadat for word embeddings
        # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
        config = projector.ProjectorConfig()

        # You can add multiple embeddings. Here we add only one.
        embedding = config.embeddings.add()
        embedding.tensor_name = cnn.embedded_chars_expanded.name
        # Link this tensor to its metadata file (e.g. labels).
        # embedding.metadata_path = os.path.join(checkpoint_dir, 'metadata.tsv')
        embedding.metadata_path = checkpoint_dir

        # Use the same LOG_DIR where you stored your checkpoint.
        embed_summary_dir = tf.summary.FileWriter(checkpoint_dir)

        # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
        # read this file during startup.
        projector.visualize_embeddings(embed_summary_dir, config)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))
        print("saving vocabulary...")

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: gConfig['dropout_keep_prob']
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_utils.batch_iter(
            list(zip(x_train, y_train)), gConfig['batch_size'], gConfig['num_epochs'])
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % gConfig['evaluate_every'] == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % gConfig['checkpoint_every'] == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
