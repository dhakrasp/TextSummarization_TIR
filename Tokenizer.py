import nltk

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
