import os
import json
import argparse
from typing import Tuple
import numpy as np
import word_vectors as wv
from .utils import required_length, load_embeddings, parse_file_type, write_embeddings


def combine_vocabs(
    base_vocab: wv.Vocab, base_vectors: wv.Vectors, new_vocab: wv.Vocab, new_vectors: wv.Vectors
) -> Tuple[wv.Vocab, wv.Vectors]:
    new_new_vectors = []
    for word, idx in new_vocab.items():
        if word not in base_vocab:
            base_vocab[word] = len(base_vocab)
            new_new_vectors.append(new_vectors[idx])
    new_new_vectors = np.array(new_new_vectors)
    return base_vocab, np.concatenate([base_vectors, new_new_vectors], axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Add new words from other pretrained embeddings to some base embeddings"
    )
    parser.add_argument(
        "embeddings",
        nargs="+",
        action=required_length(2),
        help="The pretrained embeddings to process, the first embedding is the base that others are processed relative too.",
    )
    parser.add_argument("--embeddings_index", "--embeddings-index", default="configs/embeddings.json")
    parser.add_argument("--cache", default="data")
    parser.add_argument(
        "--output_format", "--output-format", default="w2v", choices=("glove", "w2v"), type=parse_file_type,
    )
    parser.add_argument("--output_label", "--output-label")
    parser.add_argument("--output_file", "--output-file")
    args = parser.parse_args()

    args.output_label = (
        f"{args.embeddings[0]}-vocab-extended-by-{'-'.join(args.embeddings[1:])}"
        if args.output_label is None
        else args.output_label
    )

    args.output_file = os.path.join(args.cache, args.output_label) if args.output_file is None else args.output_file

    embeddings = load_embeddings(args.embeddings, args.embeddings_index, args.cache)

    dsz = set(e[1].shape[1] for e in embeddings.values())
    if len(dsz) != 1:
        raise ValueError(f"For extending vocab experiments all embedding need to be the same size, got sizes of {dsz}")

    base_vocab, base_vectors = embeddings[args.embeddings[0]]

    for embed_label in args.embeddings[1:]:
        new_vocab, new_vectors = embeddings[embed_label]
        base_vocab, base_vectors = combine_vocabs(base_vocab, base_vectors, new_vocab, new_vectors)
        del embeddings[embed_label]

    write_embeddings(args.output_file, base_vocab, base_vectors, args.output_format)

    print(json.dumps({"label": args.output_label, "file": args.output_file, "dsz": base_vectors.shape[1],}, indent=2))


if __name__ == "__main__":
    main()
