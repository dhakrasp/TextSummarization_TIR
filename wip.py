from __future__ import print_function
# os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
import codecs
import json
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Dropout, Embedding, TimeDistributed, RepeatVector
from keras.layers import LSTM, GRU, SimpleRNN
from keras.callbacks import ModelCheckpoint
from timeit import default_timer as timer
from Tokenizer import Tokenizer
from data import Vocab


MAX_SRC_LEN = 60
MAX_TAR_LEN = 20


def decode(prediction, vocab):
    outputs = []
    for i in range(0, prediction.shape[0]):
        sample = []
        e = prediction[i]
        for j in range(e.shape[0]):
            k = e[j]
            if k != 0:
                sample.append(vocab.IdToWord(k))
        outputs.append(sample)
    return outputs


def encode_input(text, tokenizer, MAX_SRC_LEN):
    assert tokenizer is not None
    assert MAX_SRC_LEN > 0
    return tokenizer.texts_to_sequences(text)


def encode_output(text, tokenizer, MAX_TAR_LEN):
    assert tokenizer is not None
    assert MAX_TAR_LEN > 0
    x = tokenizer.texts_to_sequences(text)
    enc_data = np.zeros(
        (len(x), MAX_TAR_LEN, tokenizer.vocab.NumIds() + 1), dtype=np.bool)
    for i in range(0, len(x)):
        sample = x[i]
        for j in range(0, len(sample)):
            k = sample[j]
            enc_data[i][j][k] = 1
        for j in range(len(sample), MAX_TAR_LEN):
            k = 0
            enc_data[i][j][k] = 1
    return enc_data


def preprocess_data(src_text, src_tokenizer, tar_text=None, tar_tokenizer=None):
    assert src_tokenizer is not None
    assert src_txt is not None
    src_data = encode_input(src_text, src_tokenizer, MAX_SRC_LEN)
    Y = None
    if tar_text is not None:
        assert tar_tokenizer is not None
        Y = encode_output(tar_text, tar_tokenizer, MAX_TAR_LEN)
    X = pad_sequences(src_data, maxlen=MAX_SRC_LEN, padding='post')
    return [X, Y]


def build_graph(hyper_params, model_file):

    input_dim = hyper_params['input_dim']
    embedding_dim = hyper_params['embedding_dim']
    hidden_dim = hyper_params['hidden_dim']
    output_dim = hyper_params['output_dim']
    # batch_size = hyper_params['batch_size']
    num_layers = hyper_params['num_layers']
    max_src_len = hyper_params['max_src_len']
    max_tar_len = hyper_params['max_tar_len']
    str_cell_type = hyper_params['cell_type']

    # LSTM is the default type
    if str_cell_type == 'SimpleRNN':
        cell_type = SimpleRNN
    elif str_cell_type == 'GRU':
        cell_type = GRU
    else:
        cell_type = LSTM

    # lstm model
    model = Sequential()
    model.add(Embedding(input_dim + 1, embedding_dim, init='lecun_uniform', dropout=0.2,
                        input_shape=(max_src_len, input_dim), mask_zero=True))
    RNNCell = cell_type

    # This is the encoder
    model.add(RNNCell(hidden_dim, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(hidden_dim))
    # Feed the Hidden state to each time step of decoder
    model.add(RepeatVector(max_tar_len))

    # This is the decoder
    for i in range(num_layers):
        model.add(RNNCell(hidden_dim, return_sequences=True,
                          dropout_W=0.2, dropout_U=0.2))
    model.add(TimeDistributed(Dense(output_dim)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model_json = model.to_json()
    print(model_json)
    with open(model_file, mode='w') as f:
        json.dump(model_json, f)
    model.summary()
    return model


def train(X, Y, hyper_params, epochs, model_file, weights_file):
    model = build_graph(hyper_params, model_file)
    batch_size = hyper_params['batch_size']
    checkpoint_callback = ModelCheckpoint(
        weights_file, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    start = timer()
    model.fit(X, Y, batch_size=batch_size, callbacks=[
              checkpoint_callback], nb_epoch=epochs)
    end = timer()
    print('\n\n\nTime to train\t{}\t{}s'.format('-' * 50 + '>', end - start))
    return [model, src_tokenizer, tar_tokenizer]


def test(txt, tokenizer, target_vocab, model):
    X, _ = preprocess_data(txt, tokenizer, tar_text=None, tar_tokenizer=None)
    batch_size = 20

    pred = model.predict_classes(X, batch_size=batch_size)
    # print(pred)

    outputs = decode(pred, target_vocab)
    print('\n\n\n')
    print('-' * 50 + '\t\tTesting\t\t' + '-' * 50)
    print(outputs)


def get_file_content(file_name):
    with codecs.open(file_name, mode='r', encoding='utf-8') as f:
        return f.read()


if __name__ == '__main__':
    src_txt = []
    src_file_name = "source/50"
    tar_file_name = "target/10"
    for i in range(100):
        src_txt.append(get_file_content(src_file_name))
    tar_txt = []
    for i in range(100):
        tar_txt.append(get_file_content(tar_file_name))

    src_tokenizer = Tokenizer(Vocab('vocab/src_vocab', 10000))
    tar_tokenizer = Tokenizer(Vocab('vocab/tar_vocab', 10000))

    X, Y = preprocess_data(src_txt, src_tokenizer, tar_txt, tar_tokenizer)

    input_dim = src_tokenizer.vocab.NumIds() + 1
    embedding_dim = 100
    hidden_dim = 50
    output_dim = tar_tokenizer.vocab.NumIds() + 1
    batch_size = 20
    num_layers = 2
    epochs = 20
    cell_type = 'LSTM'

    hyper_params = {}
    hyper_params['input_dim'] = input_dim
    hyper_params['embedding_dim'] = embedding_dim
    hyper_params['hidden_dim'] = hidden_dim
    hyper_params['output_dim'] = output_dim
    hyper_params['batch_size'] = batch_size
    hyper_params['num_layers'] = num_layers
    hyper_params['epochs'] = epochs
    hyper_params['cell_type'] = cell_type
    hyper_params['max_src_len'] = MAX_SRC_LEN
    hyper_params['max_tar_len'] = MAX_TAR_LEN

    model_file = 'model'
    weights_file = 'weights'
    model, src_tokenizer, tar_tokenizer = train(
        X, Y, hyper_params, epochs, model_file=model_file, weights_file=weights_file)

    model = model_from_json(model_file)

    test_txt = []
    for i in range(1):
        test_txt.append(get_file_content(src_file_name))

    test(test_txt, src_tokenizer, tar_tokenizer.vocab, model)
