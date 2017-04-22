import numpy as np
import re
from tensorflow.python.platform import gfile

sentence_labels = ["what", "who", "when", "affirmation", "unknown"]

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(labelled_data_file):
    what_examples = list()
    who_examples = list()
    when_examples = list()
    affirmation_examples = list()
    unknown_examples = list()

    with gfile.GFile(labelled_data_file, mode="r") as f:
        for line in f:
            query_and_label = line.split(",,,")

            if query_and_label[1].strip() == sentence_labels[0]:
                what_examples.append(query_and_label[0].strip())

            if query_and_label[1].strip() == sentence_labels[1]:
                who_examples.append(query_and_label[0].strip())

            if query_and_label[1].strip() == sentence_labels[2]:
                when_examples.append(query_and_label[0].strip())

            if query_and_label[1].strip() == sentence_labels[3]:
                affirmation_examples.append(query_and_label[0].strip())

            if query_and_label[1].strip() == sentence_labels[4]:
                unknown_examples.append(query_and_label[0].strip())

    x_text = what_examples + who_examples + when_examples + affirmation_examples + unknown_examples
    x_text = [clean_str(sent) for sent in x_text]

    what_labels = [[1, 0, 0, 0, 0] for _ in what_examples]
    who_labels = [[0, 1, 0, 0, 0] for _ in who_examples]
    when_labels = [[0, 0, 1, 0, 0] for _ in when_examples]
    affirmation_labels = [[0, 0, 0, 1, 0] for _ in affirmation_examples]
    unknown_labels = [[0, 0, 0, 0, 1] for _ in unknown_examples]

    y = np.concatenate([what_labels, who_labels, when_labels, affirmation_labels, unknown_labels], 0)

    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
