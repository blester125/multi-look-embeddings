import argparse
from typing import Set, List
import faiss
import numpy as np
from .utils import load_embeddings#, load_dataset


def jaccard(a: Set, b: Set) -> float:
    denom = a | b
    if len(denom) == 0:
        return 0
    return (a & b) / denom


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
            return []
        vector = self.vectors[self.vocab[word]]
        _, I = self.index.search(np.expand_dims(vector, 0), k + 1)
        # Skip over yourself
        words = [self.rev_vocab[i] for i in I[0][1:]]
        return words


def read_dataset():
    pass




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings", nargs=2)
    parser.add_argument("--embeddings_index", "--embeddings-index", default="configs/embeddings.json")
    parser.add_argument("--cache", default="data")
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--most_common", type=int, default=200)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    embeddings = load_embeddings(args.embeddings, args.embeddings_index, args.cache)

    nearest = {emb: NearestNeighbors(*embeddings[emb]) for emb in args.embeddings}

    import pdb; pdb.set_trace()




if __name__ == "__main__":
    main()
