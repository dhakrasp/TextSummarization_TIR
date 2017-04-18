from keras.preprocessing.sequence import pad_sequences

UNKNOWN_TOKEN = '<UNK>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'


class Preprocessor:
    def __init__(self, batch_size, filename, tokenizer, max_src_len):
        assert batch_size > 0
        assert filename is not None
        self.batch_size = batch_size
        self.filename = filename
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len

    def get_data(self):
        lines = []
        with open(self.filename, mode='r', encoding='utf-8') as file:
            lines = file.readlines()
        data = self.tokenizer.texts_to_sequences(lines)
        for line in data:
            line = line.append(self.tokenizer.vocab.WordToId(EOS_TOKEN))
        # return pad_sequences(data, maxlen=self.max_src_len, padding='post')
        return data
