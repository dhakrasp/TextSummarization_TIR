import numpy as np
from pprint import pprint
from keras.layers import LSTM, RepeatVector, Embedding, Input, Dense, Activation, TimeDistributed
from keras.models import Sequential, Model
from keras import regularizers
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from keras import optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau


embedding_dim = 15
rep_size = 256

max_sequence_len = 7
vocab_size = 16

total_samples = 1000
batch_size = 1
num_epochs = 100
val_split = 0.2

model_file = 'autoencoder.model'
model_weights = 'autoencoder.weights'

def get_batch_input(batch_size):
    batch_input = []
    global_data = get_data()
    for _ in range(batch_size):
        i = np.random.randint(low=0, high=total_samples)
        batch_input.append(global_data[i])
    return np.array(batch_input)


def get_data():
    labels = np.random.randint(low=1, high=vocab_size, size=(total_samples, max_sequence_len), dtype=np.int32)
    # data = np.array([to_categorical(sample) for sample in labels], dtype=np.int32)
    return labels

# def get_data():
#     filename = 'data_new/sentences'
#     with open(filename, mode='r', encoding='utf-8') as file:

global_data = get_data()
print(global_data[:10])


def get_max_class(prob_vec):
    return np.argmax(prob_vec)


def to_classes(sample):
    return [get_max_class(prob_vec) + 1 for prob_vec in sample]


def build_model():
    inputs = Input(shape=(max_sequence_len, vocab_size))
    # embedding_layer = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_sequence_len, trainable=True, weights=[embed])
    # embeddings = embedding_layer(inputs)
    encoded = LSTM(rep_size)(inputs)
    # encoded = LSTM(rep_size)(inputs)
    # encoded = Dense(rep_size, activation='sigmoid')(encoded)
    repeat = RepeatVector(max_sequence_len)(encoded)
    outputs = LSTM(vocab_size, return_sequences=True)(repeat)
    # soft_max = Activation('softmax')

    sequence_autoencoder = Model(inputs, outputs)
    # word_embeddings = Model(inputs, embeddings)
    encoder = Model(inputs, encoded)
    # decoder = Model(repeat, outputs)
    # opt = optimizers.RMSprop(lr=1, rho=0.9, epsilon=1e-08)  # , clipvalue=0.5)
    opt = optimizers.SGD(lr=1, momentum=0.95, nesterov=True)

    sequence_autoencoder.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    return sequence_autoencoder, None, encoder, None

def decoder():
    input_vector = Input(shape=(rep_size,))
    repeat = RepeatVector(max_sequence_len)(input_vector)
    outputs = LSTM(vocab_size, return_sequences=True)(repeat)


def get_model(sequential=False):
    # if sequential:
    #     return build_sequential_model(), None, None, None
    return build_model()


if __name__ == '__main__':

    autoencoder_model, _, encoder, _ = get_model()
    autoencoder_model.summary()
    autoencoder_model.save(model_file)

    x = global_data
    old_wts = autoencoder_model.layers[1].get_weights()

    reducelr_cb = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=1e-20)
    checkpoint_cb = ModelCheckpoint(model_weights, period=5)
    earlystopping_cb = EarlyStopping(min_delta=0.0001, patience=10)
    callbacks_list = [reducelr_cb, checkpoint_cb]
    # callbacks_list.append(earlystopping_cb)

    y = np.array([to_categorical(sample - 1, num_classes=vocab_size) for sample in x], dtype=np.int32)

    autoencoder_model.fit(y, y, batch_size=batch_size, verbose=1, epochs=num_epochs, validation_split=val_split, callbacks=callbacks_list)

    new_wts = autoencoder_model.layers[1].get_weights()

    # print(new_wts[0] - old_wts[0])

    print('\n')
    print('|' * 50)
    print('\n')

    x = get_batch_input(5)
    samples = np.array([to_categorical(sample - 1, num_classes=vocab_size) for sample in x], dtype=np.int32)
    # rep = encoder.predict(samples)
    # pprint(rep)

    # print('\n')
    # print('|' * 50)
    # print('\n')

    preds = autoencoder_model.predict(samples)
    pprint(x)
    print('|' * 50)
    out = np.array([to_classes(p) for p in preds])
    pprint(out)
