from __future__ import print_function
# os.environ['KERAS_BACKEND'] = 'theano'
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, TimeDistributed, RepeatVector
from keras.layers import LSTM, GRU, SimpleRNN
from keras.callbacks import ModelCheckpoint
from timeit import default_timer as timer
from Tokenizer import Tokenizer
from data import Vocab
from util import *
from preprocess import Preprocessor


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
    RNNCell = cell_type

    # lstm model
    model = Sequential()
    model.add(Embedding(input_dim + 1, embedding_dim, init='lecun_uniform', dropout=0.2,
                        input_shape=(max_src_len, input_dim), mask_zero=True))

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
    json_string = model.to_json()
    with open(model_file, mode='wb') as f:
        pickle.dump(json_string, f)
    model.summary()
    return model


def train(generator, hyper_params, batch_size, epochs, model_file, weights_file):
    model = build_graph(hyper_params, model_file)
    batch_size = hyper_params['batch_size']
    checkpoint_callback = ModelCheckpoint(
        weights_file, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    start = timer()
    model.fit_generator(generator, samples_per_epoch=batch_size, nb_epoch=epochs, callbacks=[checkpoint_callback], nb_worker=2)
    end = timer()
    print('\n\n\nTime to train\t{}\t{}s'.format('-' * 50 + '>', end - start))
    return model


if __name__ == '__main__':
    tokenizer = Tokenizer(Vocab(vocab_file, 10000))
    with open(tokenizer_file, mode='wb') as file:
        pickle.dump(tokenizer, file)

    input_dim = tokenizer.vocab.NumIds() + 1
    embedding_dim = 100
    hidden_dim = 50
    output_dim = tokenizer.vocab.NumIds() + 1
    batch_size = 2
    num_layers = 2
    epochs = 1
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

    p = Preprocessor(batch_size, pro_text_dir, tokenizer, MAX_SRC_LEN, MAX_TAR_LEN)
    gen = p.training_generator()
    model = train(generator=gen, hyper_params=hyper_params, batch_size=batch_size, epochs=epochs, model_file=model_file, weights_file=weights_file)

    # gen = p.testing_generator()
    # test(test_txt, src_tokenizer, tar_tokenizer.vocab, model, batch_size)
