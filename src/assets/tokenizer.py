import json
import nltk
from collections import Counter

nltk.download('gutenberg')

vocabulary = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
}

for character in range(0, 256):
    vocabulary[chr(character)] = len(vocabulary)

corpus1 = nltk.corpus.gutenberg.raw("shakespeare-macbeth.txt")
corpus2 = nltk.corpus.gutenberg.raw("shakespeare-hamlet.txt")
corpus3 = nltk.corpus.gutenberg.raw("melville-moby_dick.txt")
corpus4 = nltk.corpus.gutenberg.raw("austen-emma.txt")

corpus = corpus1 + corpus2 + corpus3 + corpus4


while True:
    pairs = []
    for pair_idx in range(1, len(corpus)):
        pairs.append((corpus[pair_idx-1], corpus[pair_idx]))

    pair_counter = Counter(pairs)
    most_common_pair = pair_counter.most_common(1)[0][0]

    print(most_common_pair)

    new_token = f"{most_common_pair[0]}{most_common_pair[1]}"

    if new_token not in vocabulary:
        vocabulary[new_token] = len(vocabulary)

    break


