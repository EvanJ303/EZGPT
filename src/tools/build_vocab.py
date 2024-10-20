import tiktoken
import json
import os

def build_vocab(vocab_path: str):
    encoder = tiktoken.get_encoding("cl100k_base")
    vocab = encoder.vocab
    print(vocab)

if __name__ == "__main__":
    build_vocab("vocab.json")
