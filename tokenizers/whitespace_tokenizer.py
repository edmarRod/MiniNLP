# Implementation of a simple word tokenizer, where tokens are separated with whitespace
from collections import Counter

class WhitespaceTokenizer():
    def __init__(self, vocab_size: int = 200) -> None:
        self.vocab_size = vocab_size

    def create_vocab(self, string: str) -> None:

        cnt = Counter(string.split(' '))

        most_common = cnt.most_common(self.vocab_size-2)
        tok_idx = {}
        idx_tok = {}
        for i,token in enumerate(most_common):
            token, _ = token
            tok_idx[token] = i
            idx_tok[i] = token

        tok_idx['<unk>'] = self.vocab_size-1
        idx_tok[self.vocab_size-1] = '<unk>'
        
        self.tok_idx = tok_idx
        self.idx_tok = idx_tok

    def encode(self, string: str) -> list[int]:
        return [self.tok_idx.get(token, self.vocab_size-1) for token in string.split(' ')]

    
    def decode(self, tokens: list[int]) -> list[str]:
        return [self.idx_tok.get(token, '<unk>') for token in tokens]


if __name__ == '__main__':
    train_string = """This is a possible example training text"""

    text = 'This is a possible text wow!'

    tokenizer = WhitespaceTokenizer()
    tokenizer.create_vocab(train_string)
    encoded_text = tokenizer.encode(text)
    decoded_text = tokenizer.decode(encoded_text)

    print(f'Text: {text}')
    print(f'Encoded text: {encoded_text}')
    print(f'Decoded text: {decoded_text}')