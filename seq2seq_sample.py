from __future__ import print_function
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, TimeDistributed, RepeatVector
from keras.layers import LSTM


def shift(seq, n):
    n = n % len(seq)
    return seq[n:] + seq[:n]


def decode(prediction, target_dict):
    outputs = []
    for i in range(0, prediction.shape[0]):
        sample = []
        e = np.exp(prediction[i])
        s = np.sum(e, axis=1)
        for j in range(s.shape[0]):
            e[j] = e[j] / s[j]
        max_args = np.argmax(e, axis=1)
        for i in max_args:
            if i != 0:
                sample.append(target_dict[i])
        outputs.append(sample)
    return outputs


def encode(text, vocab=None, max_len=10):
    tk = Tokenizer(nb_words=2000, lower=True, split=" ")
    tk.fit_on_texts(text)
    x = tk.texts_to_sequences(text)

    enc_data = np.zeros(
        (len(x), max_len, len(tk.word_index) + 1), dtype=np.bool)
    for i in range(0, len(x)):
        sample = x[i]
        for j in range(0, len(sample)):
            k = sample[j] - 1
            enc_data[i][j][k] = 1
    return [enc_data, tk]


def preprocess_data(text, vocab):
    data = encode(text, vocab)
    return data


def train(src_txt, tar_txt, model_file, num_layers=4):

    src_data, src_tk = encode(src_txt)
    tar_data, tar_tk = encode(tar_txt)

    X = src_data
    Y = tar_data

    # padding sequence
    max_len = 10
    input_dim = len(src_tk.word_counts) + 1
    # embedding_dim = 128
    # hidden_dim = 10
    output_dim = len(tar_tk.word_counts) + 1

    # lstm model
    model = Sequential()

    # model.add(Embedding(input_dim, embedding_dim, dropout = 0.2))

    RNNCell = LSTM

    model.add(RNNCell(output_dim, dropout_W=0.2,
                      dropout_U=0.2, input_shape=(max_len, input_dim)))
    model.add(RepeatVector(max_len))
    for i in range(num_layers):
        model.add(RNNCell(output_dim, return_sequences=True,
                          dropout_W=0.2, dropout_U=0.2))
    model.add(TimeDistributed(Dense(output_dim)))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.summary()
    model.fit(X, Y, batch_size=20, nb_epoch=10)
    return [model, src_tk, tar_tk]


def test(txt, target_dict, model, vocab=None):
    print('-' * 50 + 'testing' + '-' * 50)
    test_data, test_tk = encode(txt)

    X = test_data

    pred = model.predict(X, batch_size=20)

    print(target_dict)
    outputs = decode(pred, target_dict)
    print(outputs)
    print('-' * 100)


if __name__ == '__main__':
    src_txt = []
    for i in range(100):
        src_txt.append("Hi, this is Pooojitha and Pranav.")
    tar_txt = []
    for i in range(100):
        tar_txt.append("gi, ghis gis Granav and Goojitha.")

    # vocab = Vocab('raw_data/vocab')
    model, src_tk, tar_tk = train(src_txt, tar_txt, 'model')

    tar_index_to_word = {}
    for k, v in tar_tk.word_index.items():
        tar_index_to_word[v] = k

    test_txt = []
    for i in range(1):
        test_txt.append("Hi, this is Pooojitha and Pranav.")
    test(test_txt, tar_index_to_word, model)
