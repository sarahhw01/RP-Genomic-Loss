class NTLevelTokenizer:
    def __init__(self, vocab=None, default="N"):
        if vocab is None:
            self.vocab = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "D": 5}
        else:
            self.vocab = vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        if default not in self.vocab:
            raise ValueError("Default character not in vocab")

        self.default = self.vocab[default]

    def __len__(self):
        return self.vocab_size

    def tokenize(self, string):
        return [self.vocab.get(c, self.default) for c in string]

    def tokenize_batch(self, strings):
        return [self.tokenize(s) for s in strings]

    def detokenize(self, tokens):
        return "".join([self.inv_vocab[int(t)] for t in tokens])

    def detokenize_batch(self, token_batches):
        return [self.detokenize(t) for t in token_batches]
