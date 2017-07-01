import nltk
import os
import codecs
import re
import xml.etree.ElementTree as et
from util import *

UNKNOWN_TOKEN = '<UNK>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'


def generate_vocab(data_dir_name):
    word_counts = dict()
    word_counts[UNKNOWN_TOKEN] = 0
    word_counts[SOS_TOKEN] = 1
    word_counts[EOS_TOKEN] = 2
    for fname in os.listdir(data_dir_name):
        for line in codecs.open(os.path.join(data_dir_name, fname), encoding='utf-8'):
            tokens = nltk.word_tokenize(line)
            for t in tokens:
                tok = t.lower()
                if tok in word_counts.keys():
                    word_counts[tok] = word_counts[tok] + 1
                else:
                    word_counts[tok] = 1
    return word_counts


def convert_to_raw_text(data_dir_name):
    for fname in os.listdir(data_dir_name):
        with codecs.open(os.path.join(data_dir_name, fname), encoding='utf-8') as f:
            xml_data = f.read()
            root = et.fromstring(xml_data)
            text = root.find('content').text
            re.sub('<[^<]+?>', '', text)
            with codecs.open(os.path.join(data_dir_name, fname), mode='w', encoding='utf-8') as raw:
                raw.write(text)


def write_vocab(data_dir_name, vocab_file_name):
    word_counts = generate_vocab(data_dir_name)
    with codecs.open(vocab_file_name, mode='w', encoding='utf-8') as f:
        for k, v in word_counts.items():
            f.write(u'{}\t{}\n'.format(k, v))


if __name__ == '__main__':
    data_dir_name = pro_text_dir
    vocab_file_name = 'vocab/vocab'
    write_vocab(data_dir_name, vocab_file_name)
    # data_dir_name = 'source'
    # vocab_file_name = 'vocab/src_vocab'
    # write_vocab(data_dir_name, vocab_file_name)
