import math
import string
import argparse
from itertools import chain
from collections import Counter
from typing import Set, List, Counter as CounterType, Optional, Callable
import faiss
import numpy as np
from .utils import load_embeddings, load_dataset, read_conll_dataset, read_label_first, STOP_WORDS


def jaccard(a: Set, b: Set) -> float:
    denom = a | b
    if len(denom) == 0:
        return 0
    return len(a & b) / len(denom)


class NearestNeighbors:
    def __init__(self, vocab, vectors):
        self.vocab = vocab
        self.rev_vocab = {v: k for k, v in vocab.items()}
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.vectors = vectors

    def nearest(self, word: str, k: int) -> List[str]:
        if word not in self.vocab:
            return set()
        vector = self.vectors[self.vocab[word]]
        _, I = self.index.search(np.expand_dims(vector, 0), k + 1)
        # Skip over yourself
        words = set(self.rev_vocab[i] for i in I[0][1:])
        return words


def frequencies(tokens: List[List[str]], transform: Callable[[str], str] = lambda x: x, filter_tokens: Optional[Set[str]] = None) -> CounterType[str]:
    filter_tokens = set() if filter_tokens is None else filter_tokens
    return Counter(filter(lambda x: x not in filter_tokens, map(transform, chain(*tokens))))



def main():
    parser = argparse.ArgumentParser(description="Calculate the Average Jaccard similarity between nearest neighbors in embeddings spaces.")
    parser.add_argument("embeddings", nargs=2)
    parser.add_argument("--embeddings_index", "--embeddings-index", default="configs/embeddings.json")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--datasets_index", "--datasets-index", default="configs/datasets.json")
    parser.add_argument("--data_format", "--data-format", default="conll", choices=("conll", "label-first"))
    parser.add_argument("--cache", default="data")
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--most_common", type=int, default=200)
    parser.add_argument("--filter_tokens", type=set, nargs="*", default=STOP_WORDS)
    parser.add_argument("--lowercase", action="store_true")
    args = parser.parse_args()

    embeddings = load_embeddings(args.embeddings, args.embeddings_index, args.cache)

    nearest = {emb: NearestNeighbors(*embeddings[emb]) for emb in args.embeddings}

    read = read_conll_dataset if args.data_format == "conll" else read_label_first

    data = load_dataset(args.dataset, args.datasets_index, args.cache, read=read)

    transform = str.lower if args.lowercase else lambda x: x

    for dataset in ('train', 'dev', 'test'):
        freqs = frequencies(data[dataset], transform, args.filter_tokens)
        with open(f"{dataset}.txt", "w") as wf:
            wf.write("\n".join(t for t, _ in freqs.most_common(args.most_common)))

        jaccards = []
        for token, _ in freqs.most_common(args.most_common):
            neighbors = [near.nearest(token, args.neighbors) for near in nearest.values()]
            jaccards.append(jaccard(*neighbors))

        avg_jaccard = math.fsum(jaccards) / len(jaccards)

        print(
            f"{avg_jaccard * 100:.4f}: Average Jaccard Similarity between the top {args.neighbors} nearest neighbors "
            f"for the {args.most_common} most common words in the {args.dataset} {dataset} data using the "
            f"{' and '.join(args.embeddings)} embeddings."
        )


if __name__ == "__main__":
    main()
