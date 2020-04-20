import argparse
from typing import Set
import faiss
from .utils import load_embeddings


def jaccard(a: Set, b: Set) -> float:
    denom = a | b
    if len(denom) == 0:
        return 0
    return (a & b) / denom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings", nargs=2)
    parser.add_argument("--embeddings_index", "--embeddings-index", default="configs/embeddings.json")
    parser.add_argument("--cache", deafult="data")
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--most_common", type=int, default=200)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()


if __name__ == "__main__":
    main()
