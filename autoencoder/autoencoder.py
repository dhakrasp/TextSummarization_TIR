import numpy as np
import pickle
from pprint import pprint
from keras.layers import LSTM, RepeatVector, Embedding, Input
from keras.models import Model
from keras import regularizers
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from data import Vocab
from preprocess import Preprocessor
from tokenizer import Tokenizer

model_weights = 'model.weights'
model_file = 'model.model'


class AutoEncoder():

    ''' To Do:
        store the hyper params in object instance
        store paths of where to save model and its weights
        figure out how to use pretrained embeddings

    '''

    def __init__(self, max_sequence_len, vocab_size, embedding_dim, hidden_dim, optimizer=None, word_embeddings_file=None):
        if word_embeddings_file is None:
            # Get the word embeddings from the file and set to embed_param
            self.embed_param = None
        else:
            self.embed_param = None
        self.word_embeddings = None
        self.vocab_size = vocab_size
        self.max_sequence_len = max_sequence_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.optimizer = optimizer

    def build_models(self):
        if self.optimizer is None:
            self.optimizer = optimizers.SGD(lr=0.1, momentum=0.95, nesterov=True)

        inputs = Input(shape=(self.max_sequence_len,))
        if self.embed_param is None:
            embedding_layer = Embedding(input_dim=self.vocab_size + 1, output_dim=self.embedding_dim, input_length=self.max_sequence_len)
        else:
            embedding_layer = Embedding(input_dim=self.vocab_size + 1, output_dim=self.embedding_dim, input_length=self.max_sequence_len, weights=[self.embed_param], trainable=False)
        embeddings = embedding_layer(inputs)
        encoded = LSTM(self.hidden_dim)(embeddings)
        repeat = RepeatVector(self.max_sequence_len)(encoded)
        outputs = LSTM(self.vocab_size, return_sequences=True)(repeat)

        self.sequence_autoencoder = Model(inputs, outputs)
        self.encoder = Model(inputs, encoded)
        self.sequence_autoencoder.compile(optimizer=self.optimizer, loss='mse', metrics=['accuracy'])
        self.decoder = self.build_decoder()
        self.sequence_autoencoder.summary()

    def build_decoder(self):
        inputs = Input(shape=(self.hidden_dim,))
        repeat = RepeatVector(self.max_sequence_len)(inputs)
        lstm_layer = LSTM(self.vocab_size, return_sequences=True)
        lstm_weights = self.sequence_autoencoder.layers[2].get_weights()
        lstm_layer.set_weights = [lstm_weights]
        outputs = lstm_layer(repeat)
        decoder = Model(inputs, outputs)
        return decoder

    def train(self, x, batch_size=1, num_epochs=10, val_split=0.15, callbacks_list=None):
        y = np.array([to_categorical(sample - 1, num_classes=self.vocab_size) for sample in x], dtype=np.int32)
        self.sequence_autoencoder.fit(x, y, batch_size=batch_size, verbose=1, epochs=num_epochs, validation_split=val_split, callbacks=callbacks_list)

    def predict(self, x):
        return self.sequence_autoencoder.predict(x)

    def encode(self, sentence):
        return self.encoder.predict(sentence)

    def decode(self, sentence_vector):
        return self.decoder(sentence_vector)


if __name__ == '__main__':
    vocab_file = '../vocab/vocab'
    tokenizer_file = '../tokenizer/src_tokenizer'
    vocab = Vocab(vocab_file, 100000)
    tokenizer = Tokenizer(vocab)
    with open(tokenizer_file, mode='wb') as file:
        pickle.dump(tokenizer, file)
    max_sequence_len = 50
    p = Preprocessor(1, 'data/sentences.txt', tokenizer, max_sequence_len)

    embedding_dim = 128
    hidden_dim = 512
    ae = AutoEncoder(max_sequence_len, vocab.NumIds(), embedding_dim, hidden_dim)
    ae.build_models()
    reducelr_cb = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=1e-20)
    checkpoint_cb = ModelCheckpoint(model_weights, period=5)
    earlystopping_cb = EarlyStopping(min_delta=0.0001, patience=10)
    callbacks_list = [reducelr_cb, checkpoint_cb]
    exit()
    x = p.get_data()
    print('-' * 30, 'Loaded data', '-' * 30)
    ae.train(x, batch_size=1, num_epochs=1, callbacks_list=callbacks_list)
