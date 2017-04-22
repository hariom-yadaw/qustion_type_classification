# predict question type

import tensorflow as tf
import numpy as np
import os
import math

from tensorflow.contrib import learn

import ast
import sys

sentence_labels = ["what", "who", "when", "affirmation", "unknown"]

try:
    from ConfigParser import SafeConfigParser
except:
    from configparser import SafeConfigParser

class query_NLU(object):

    def __init__(self):
        self.config = self.get_config(config_file='config.ini')

        if self.config['isdebug']:
            print('Applying Parameters: %d' %len(self.config))
            # print(self.gConfig['data_dir'])
            for k, v in self.config.iteritems():
                print('%s: %s' % (k, str(v)))
            print("")

        self.graph = tf.Graph()
        with self.graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=self.config['allow_soft_placement'],
                log_device_placement=self.config['log_device_placement'])

            self.sess = tf.Session(config=session_conf)

        self.load_model()


    def get_config(self, config_file='config.ini'):
        parser = SafeConfigParser()
        parser.read(config_file)
        # get the ints, floats and strings
        _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
        _conf_floats = [(key, float(value)) for key, value in parser.items('floats')]
        _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
        _conf_booleans = [(key, ast.literal_eval(value)) for key, value in parser.items('booleans')]
        return dict(_conf_ints + _conf_floats + _conf_strings + _conf_booleans)

    def load_model(self):
        vocab_path = os.path.join(self.config['checkpoint_dir'], "..", "vocab")
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

        checkpoint_file = tf.train.latest_checkpoint(self.config['checkpoint_dir'])

        with self.graph.as_default():
            with self.sess.as_default():
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)
                # Get the placeholders from the graph by name
                self.input_x = self.graph.get_operation_by_name("input_x").outputs[0]
                self.dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                self.scores = self.graph.get_operation_by_name("output/scores").outputs[0]
                self.predictions = self.graph.get_operation_by_name("output/predictions").outputs[0]

    def analyze_sentence(self, sentence):
        x_raw = [sentence]
        x_test = np.array(list(self.vocab_processor.transform(x_raw)))

        scores = self.sess.run(self.scores, {self.input_x: x_test, self.dropout_keep_prob: 1.0})
        # print(scores[0])
        label = self.sess.run(self.predictions, {self.input_x: x_test, self.dropout_keep_prob: 1.0})
        # print(label[0])
        prob = self.calc_prob(scores[0], label[0])
        # print(prob)

        if prob < self.config['probability_thresh']:
            return sentence_labels[4], prob
        else:
            return sentence_labels[label[0]], prob


    def calc_prob(self, score, index):
        total = math.exp(score[0]) + math.exp(score[1]) + math.exp(score[2]) + math.exp(score[3]) + math.exp(score[4])
        return math.exp(score[index])/total

if __name__ == '__main__':
    objTest = query_NLU()
    print("Please enter sentences indicating questions of type: what/who/when/affirmative/unknown")
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()

    while sentence:
        # objTest.analyze_sentence(sentence)
        pred_label, prob = objTest.analyze_sentence(sentence)
        if objTest.config['isdebug']:
            print(pred_label + " (" + str(round(prob * 100, 2)) + ")")
        else:
            print(pred_label)

        print(" ")
        # print("> ",)
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

