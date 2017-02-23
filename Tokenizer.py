import nltk
# from data import Vocab, UNKNOWN_TOKEN


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    # List of samples. A sample may be a single sentence or multiple sentences
    # (paragraph or document).
    def texts_to_sequences(self, texts):
        output = []
        for sample in texts:
            sequence = []
            tokenized = nltk.word_tokenize(sample.lower())
            for word in tokenized:
                sequence.append(self.vocab.WordToId(word))
            output.append(sequence)
        return output


# if __name__ == '__main__':
#     vocab = Vocab('vocab/src_vocab', 10000)
#     tk = Tokenizer(vocab)
#     print(tk.vocab.CheckVocab(UNKNOWN_TOKEN))
#     print(tk.text_to_sequences(["Hi, Pranav."]))
