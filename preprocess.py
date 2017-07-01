import codecs
import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences


class Preprocessor:
    def __init__(self, batch_size, dir_name, tokenizer, max_src_len, max_tar_len):
        assert batch_size > 0
        assert dir_name is not None
        self.batch_size = batch_size
        self.dir_name = dir_name
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tar_len = max_tar_len

    def decode(self, prediction):
        outputs = []
        for i in range(0, prediction.shape[0]):
            sample = []
            e = prediction[i]
            for j in range(e.shape[0]):
                k = e[j]
                if k != 0:
                    sample.append(self.tokenizer.vocab.IdToWord(k))
            outputs.append(sample)
        return outputs

    @staticmethod
    def get_article(filename):
        with codecs.open(filename, encoding='utf-8') as f:
            article = f.readlines()
        return article[0]

    @staticmethod
    def get_summary(filename):
        with codecs.open(filename, encoding='utf-8') as f:
            summary = f.readlines()
        return summary[2]

    def __iter__(self):
        return self.testing_generator()

    def training_generator(self):
        while True:
            counter = 0
            articles = []
            summaries = []

            for fname in os.listdir(self.dir_name):
                if counter % self.batch_size == 0:
                    articles = []
                    summaries = []
                file_name = os.path.join(self.dir_name, fname)
                articles.append(Preprocessor.get_article(file_name))
                summaries.append(Preprocessor.get_summary(file_name))
                counter += 1
                if counter % self.batch_size == 0:
                    X = self.encode_input(articles)
                    Y = self.encode_output(summaries)
                    yield (X, Y)

            # This is to handle cases where number of elements (left) is less than batch_size
            X = self.encode_input(articles)
            print(X.shape)
            print('|' * 50)
            Y = self.encode_output(summaries)
            print(Y.shape)
            print('|' * 50)
            yield (X, Y)

    def testing_generator(self):
        counter = 0
        articles = []
        for fname in os.listdir(self.dir_name):
            file_name = os.path.join(self.dir_name, fname)
            articles.append(Preprocessor.get_article(file_name))
            counter += 1
            if counter % self.batch_size == 0:
                X = self.encode_input(articles)
                articles = []
                yield X
        # This is to handle cases where number of elements (left) is less than batch_size
        X = self.encode_input(articles)
        yield X

    def encode_input(self, text):
        X = self.tokenizer.texts_to_sequences(text)
        return pad_sequences(X, maxlen=self.max_src_len, padding='post')

    def encode_output(self, text):
        x = self.tokenizer.texts_to_sequences(text)
        enc_data = np.zeros(
            (len(x), self.max_tar_len, self.tokenizer.vocab.NumIds() + 1), dtype=np.bool)
        for i in range(len(x)):
            sample = x[i]
            for j in range(len(sample)):
                k = sample[j]
                enc_data[i][j][k] = True
            for j in range(len(sample), self.max_tar_len):
                k = 0
                enc_data[i][j][k] = True
        return enc_data

    @staticmethod
    def process_and_copy_data(dir_name1, dir_name2):
        counter = 0
        for fname in os.listdir(dir_name1):
            counter += 1
            try:
                with codecs.open(os.path.join(dir_name1, fname), 'r', encoding='utf-8') as f:
                    text = f.readlines()
            except UnicodeDecodeError:
                print('Error while reading file --------->\t', fname)
                continue
            i = 2
            j = 1
            art = [''] * 4
            # entities=[0]*200
            entities = ''
            while j <= 3:

                article = ''
                while i < len(text) and text[i] != '\n':
                    if j == 1:
                        words = text[i].replace('@', '').split()
                        if words[-1] == '1':
                            # print(words[-1])
                            st = " ".join(words[:-1])
                            st.replace('\t', ';')
                            st += '. '
                            article += st
                    if j == 2:
                        lords = text[i].replace('@', '').split()
                        lords = " ".join(lords)
                        lords.replace('\n', ' ')
                        lords += '. '
                        article += lords

                    if j == 3:
                        entities = entities + ' ' + text[i]
                    i += 1
                art[j] = article
                i = i + 1
                j += 1
            new_file_name = os.path.join(dir_name2, str(counter) + '.txt')
            with codecs.open(new_file_name, 'w', encoding='utf-8') as f:
                f.write(art[1] + '\n\n')
                f.write(art[2] + '\n\n')
                # f.write(entities)
