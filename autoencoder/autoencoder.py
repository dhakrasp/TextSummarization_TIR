import numpy as np
import pickle
import random
from keras.layers import LSTM, RepeatVector, Embedding, Input, TimeDistributed, Dense, GRU, Bidirectional
from keras.models import Model, model_from_json
from keras.utils.np_utils import to_categorical
from keras import optimizers
from data import Vocab
from preprocess import Preprocessor
from tokenizer import Tokenizer

model_weights = 'model.weights'
model_file = 'model.model'
model_params = 'model.params'


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
        self.sequence_autoencoder = None
        self.encoder = None
        self.decoder = None

    # def __init__(self, hyper_params):
    #     self.word_embeddings = None
    #     self.vocab_size = hyper_params['vocab_size']
    #     self.max_sequence_len = hyper_params['max_sequence_len']
    #     self.embedding_dim = hyper_params['embedding_dim']
    #     self.hidden_dim = hyper_params['hidden_dim']
    #     self.optimizer = hyper_params['optimizer']
    #     self.sequence_autoencoder = None
    #     self.encoder = None
    #     self.decoder = None


    def build_encoder(self):
        inputs = Input(shape=(self.max_sequence_len,))
        embedding_layer = Embedding(input_dim=self.vocab_size + 1, output_dim=self.embedding_dim, input_length=self.max_sequence_len)
        embeddings = embedding_layer(inputs)
        gru_layer = GRU(self.hidden_dim)
        encoded = gru_layer(embeddings)
        
        # embedding_layer.set_weights([self.sequence_autoencoder.layers[1].get_weights()])
        # gru_layer.set_weights([self.sequence_autoencoder.layers[2].get_weights()])

        encoder = Model(inputs, encoded)
        encoder.summary()
        return encoder

    def build_decoder(self):
        inputs = Input(shape=(self.hidden_dim,))
        repeat = RepeatVector(self.max_sequence_len)(inputs)

        lstm_layer = GRU(self.hidden_dim, return_sequences=True)
        hidden_rep = lstm_layer(repeat)
        time_dist_dense_layer = TimeDistributed(Dense(self.vocab_size, activation='softmax'))
        outputs = time_dist_dense_layer(hidden_rep)
        
        lstm_layer.set_weights([self.sequence_autoencoder.layers[4].get_weights()])
        time_dist_dense_layer.set_weights([self.sequence_autoencoder.layers[5].get_weights()])

        decoder = Model(inputs, outputs)
        decoder.summary()
        return decoder

    def build_models(self):
        if self.optimizer is None:
            self.optimizer = optimizers.SGD(lr=1e-4, momentum=0.9, nesterov=True)
            # self.optimizer = optimizers.RMSprop(lr=1e-4)

        inputs = Input(shape=(self.max_sequence_len,))
        embedding_layer = Embedding(input_dim=self.vocab_size + 1, output_dim=self.embedding_dim, input_length=self.max_sequence_len)
        # if self.embed_param is None:
        #     embedding_layer = Embedding(input_dim=self.vocab_size + 1, output_dim=self.embedding_dim, input_length=self.max_sequence_len)
        # else:
        #     embedding_layer = Embedding(input_dim=self.vocab_size + 1, output_dim=self.embedding_dim, input_length=self.max_sequence_len, weights=[self.embed_param], trainable=False)
        embeddings = embedding_layer(inputs)
        encoded = Bidirectional(GRU(self.hidden_dim))(embeddings)
        repeat = RepeatVector(self.max_sequence_len)(encoded)
        outputs = GRU(self.embedding_dim, return_sequences=True)(repeat)
        outputs = TimeDistributed(Dense(self.vocab_size, activation='softmax'))(outputs)

        self.sequence_autoencoder = Model(inputs, outputs)
        # self.encoder = self.build_encoder()
        # self.decoder = self.build_decoder()
        self.sequence_autoencoder.compile(optimizer=self.optimizer, loss='mse', metrics=['accuracy'])
        self.sequence_autoencoder.summary()

    def train(self, x, batch_size=16, num_epochs=1, val_split=0.15, callbacks_list=None):
        # y = np.array([to_categorical(sample - 1, num_classes=self.vocab_size) for sample in x], dtype=np.int32)
        def data_generator(data, vocab_size, batch_size):
            while True:
                Y = []
                X = []
                for count, x in enumerate(data):
                    Y = []
                    X = []
                    Y.append(to_categorical(x - 1, num_classes=vocab_size))
                    X.append(x)
                    if count % batch_size == 0:
                        yield (np.array(X), np.array(Y))
                yield (np.array(X), np.array(Y))
        split = int(val_split * len(x))
        random.shuffle(x)
        val_data = x[:split]
        train_data = x[split:]
        x = None

        self.sequence_autoencoder.fit_generator(
            data_generator(train_data, self.vocab_size, batch_size),
            steps_per_epoch=len(train_data) / batch_size,
            epochs=num_epochs,
            validation_data=data_generator(val_data, self.vocab_size, batch_size),
            validation_steps=len(val_data) / batch_size)

    def predict(self, x):
        predict_probs = self.sequence_autoencoder.predict(x)
        predictions = []
        for sample in predict_probs:
            pred_sequence = [np.argmax(step) for step in sample]
            predictions.append(pred_sequence)
        return predictions

    def encode(self, x):
        return self.encoder.predict(x)

    def decode(self, sentence_vectors):
        predict_probs = self.decoder.predict(sentence_vectors)
        predictions = []
        for sample in predict_probs:
            pred_sequence = [np.argmax(step) for step in sample]
            predictions.append(pred_sequence)
        return predictions

    def get_hyper_params(self):
        hyper_params = {}
        hyper_params['vocab_size'] = self.vocab_size
        hyper_params['max_sequence_len'] = self.max_sequence_len
        hyper_params['embedding_dim'] = self.embedding_dim
        hyper_params['hidden_dim'] = self.hidden_dim
        # hyper_params['optimizers'] = self.optimizer
        return hyper_params

    @classmethod
    def load(self, hyper_params_file, model_file, model_weights):
        model = None
        with open(hyper_params_file, mode='rb') as file:
            hyper_params = pickle.load(file)
        vocab_size = hyper_params['vocab_size']
        max_seq_len = hyper_params['max_sequence_len']
        embedding_dim = hyper_params['embedding_dim']
        hidden_dim = hyper_params['hidden_dim']
        # optimizer = hyper_params['optimizers']
        model = AutoEncoder(max_seq_len, vocab_size, embedding_dim, hidden_dim)
        with open(model_file, mode='rb') as file:
            json_string = pickle.load(file)
        model.sequence_autoencoder = model_from_json(json_string)
        model.sequence_autoencoder.load_weights(model_weights)
        # model.build_encoder()
        # model.build_decoder()
        return model

    def save(self, hyper_params_file, model_file, model_weights):
        self.sequence_autoencoder.save_weights(model_weights)
        with open(model_file, mode='wb') as file:
            pickle.dump(self.sequence_autoencoder.to_json(), file)
        with open(hyper_params_file, mode='wb') as file:
            pickle.dump(self.get_hyper_params(), file)


def predict_sentences(predictions, vocab):
    return [" ".join(vocab.IdToWord(i) for i in prediction) for prediction in predictions]


if __name__ == '__main__':
    vocab_file = '../vocab/vocab'
    tokenizer_file = '../tokenizer/src_tokenizer'
    vocab = Vocab(vocab_file, 100000)
    tokenizer = Tokenizer(vocab)
    with open(tokenizer_file, mode='wb') as file:
        pickle.dump(tokenizer, file)
    max_sequence_len = 10
    batch_size = 4
    p = Preprocessor(batch_size, 'data/sentences.txt', tokenizer, max_sequence_len)

    embedding_dim = 50
    hidden_dim = 100
    ae = AutoEncoder(max_sequence_len, vocab.NumIds(), embedding_dim, hidden_dim)
    ae.build_models()
    reducelr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=1e-20)
    checkpoint_cb = ModelCheckpoint(model_weights, period=1)
    earlystopping_cb = EarlyStopping(min_delta=0.0001, patience=10)
    callbacks_list = [reducelr_cb, checkpoint_cb, earlystopping_cb]
    x = p.get_data()[:5000]
    print(len(x))
    print('-' * 30, 'Loaded data', '-' * 30)
    ae.train(x, batch_size, num_epochs=30, callbacks_list=callbacks_list)

    # AutoEncoder.save(ae, model_params, model_file, model_weights)
    # ae = AutoEncoder.load(model_params, model_file, model_weights)
    data = p.process_sentences(['malignant melanoma is now the fifth most common cancer .', 'the babies are conjoined from the chest to the navel.', 'experts believe the rapidly rising rate in older people.'])
    print(data)
    prediction = ae.predict(data)
    sentences = predict_sentences(prediction, vocab)
    for s in sentences:
        print(s)
