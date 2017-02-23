import codecs
import numpy as np
from keras.preprocessing.sequence import pad_sequences


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


def encode_input(text, tokenizer, max_src_len):
    assert tokenizer is not None
    assert max_src_len > 0
    return tokenizer.texts_to_sequences(text)


def encode_output(text, tokenizer, max_tar_len):
    assert tokenizer is not None
    assert max_tar_len > 0
    x = tokenizer.texts_to_sequences(text)
    enc_data = np.zeros(
        (len(x), max_tar_len, tokenizer.vocab.NumIds() + 1), dtype=np.bool)
    for i in range(0, len(x)):
        sample = x[i]
        for j in range(0, len(sample)):
            k = sample[j]
            enc_data[i][j][k] = 1
        for j in range(len(sample), max_tar_len):
            k = 0
            enc_data[i][j][k] = 1
    return enc_data


def preprocess_data(src_text, src_tokenizer, max_src_len, tar_text=None, tar_tokenizer=None, max_tar_len=0):
    assert src_tokenizer is not None
    assert src_text is not None
    src_data = encode_input(src_text, src_tokenizer, max_src_len)
    Y = None
    if tar_text is not None:
        assert tar_tokenizer is not None
        Y = encode_output(tar_text, tar_tokenizer, max_tar_len)
    X = pad_sequences(src_data, maxlen=max_src_len, padding='post')
    return [X, Y]

def get_file_content(file_name):
    with codecs.open(file_name, mode='r', encoding='utf-8') as f:
        return f.read()
