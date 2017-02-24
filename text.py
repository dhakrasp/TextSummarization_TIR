import os
from preprocess import Preprocessor
from Tokenizer import Tokenizer
from data import Vocab
from util import *


class X:
    def __init__(self, n):
        self.n = n

    def gen(self):
        for i in range(self.n):
            yield i


if __name__ == '__main__':
    Preprocessor.process_and_copy_data(raw_text_dir, pro_text_dir)