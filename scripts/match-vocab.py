import os
import json
import argparse
from typing import Tuple
import numpy as np
import word_vectors as wv
from .utils import required_length, load_embeddings, parse_file_type, write_embeddings


def match_vocab(base_vocab: wv.Vocab, new_vocab: wv.Vocab, new_vectors: wv.Vectors) -> Tuple[wv.Vocab, wv.Vectors]:
    new_new_vectors = []
    new_new_vocab = {}
    new_idx = 0
    for word, idx in new_vocab.items():
        if word in base_vocab and word not in new_new_vocab:
            new_new_vocab[word] = new_idx
            new_idx += 1
            new_new_vectors.append(new_vectors[idx])
    import pdb; pdb.set_trace()
    new_new_vectors = np.array(new_new_vectors)
    assert len(new_new_vocab) == new_new_vectors.shape[0]
    return new_new_vocab, new_new_vectors


def main():
    parser = argparse.ArgumentParser(
        description="Prune the vocab for pretrained embeddings so the vocabs match another pretrained embedding"
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
        "--output_format", "--output-format", default="w2v", choices=("glove", "w2v"), type=parse_file_type
    )
    parser.add_argument("--output_labels", "--output-labels", nargs="+")
    parser.add_argument("--output_files", "--output-files", nargs="+")
    args = parser.parse_args()

    args.output_labels = (
        [f"{embed}-matched-to-{args.embeddings[0]}" for embed in args.embeddings[1:]]
        if args.output_labels is None
        else args.output_labels
    )
    args.output_files = (
        [os.path.join(args.cache, f"{embed}-matched-to-{args.embeddings[0]}") for embed in args.embeddings[1:]]
        if args.output_files is None
        else args.output_files
    )

    if len(args.output_labels) != len(args.embeddings) - 1:
        raise ValueError(
            f"Supplied output labels must be provided for each embedding you are matching, needed {len(args.embeddings) - 1} labels got {len(args.output_labels)}"
        )

    if len(args.output_files) != len(args.embeddings) - 1:
        raise ValueError(
            f"Supplied output files must be provided for each embedding you are matching, needed {len(args.embeddings) - 1} files got {len(args.output_files)}"
        )

    embeddings = load_embeddings(args.embeddings, args.embeddings_index, args.cache)

    base_vocab, _ = embeddings[args.embeddings[0]]

    embedding_info = []

    for embed_label, label, output in zip(args.embeddings[1:], args.output_labels, args.output_files):
        new_vocab, new_vectors = embeddings[embed_label]
        vocab, vectors = match_vocab(base_vocab, new_vocab, new_vectors)
        embedding_info.append(
            {"label": label, "file": output, "dsz": vectors.shape[1],}
        )
        write_embeddings(output, vocab, vectors, args.output_format)

    print(json.dumps(embedding_info, indent=2))


if __name__ == "__main__":
    main()
