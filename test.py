import pickle
from preprocess import Preprocessor
from util import *
from keras.models import model_from_json


def test(generator, preprocessor, model):
    for X in preprocessor:
        if len(X) > 0:
            print(len(X))
            pred = model.predict_classes(X, batch_size=len(X))
            print(pred.shape)
            outputs = preprocessor.decode(pred)
            for op in outputs:
                print(op)
                print('-' * 100)
            print('|' * 100)


if __name__ == '__main__':
    with open(model_file, mode='rb') as file:
        json_string = pickle.load(file)
    with open(tokenizer_file, mode='rb') as file:
        tokenizer = pickle.load(file)
    model = model_from_json(json_string)
    model.load_weights(weights_file)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    batch_size = 4
    p = Preprocessor(batch_size, pro_text_dir, tokenizer, MAX_SRC_LEN, MAX_TAR_LEN)
    gen = p.testing_generator()
    test(gen, p, model)
