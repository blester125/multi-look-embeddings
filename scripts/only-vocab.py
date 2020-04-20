import json
import argparse
import numpy as np
from baseline.utils import EmbeddingDownloader
from mead.utils import index_by_label, read_config_file_or_json
import word_vectors as wv


def required_length(min_):
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if len(values) < min_:
                raise argparse.ArgumentTypeError(
                    f'Argument "{self.dest}" requires at least {min_} arguments, got {len(values)}'
                )
            setattr(args, self.dest, values)
    return RequiredLength


def combine_vocabs(base_vocab, base_vectors, new_vocab, new_vectors):
    new_new_vectors = []
    for word, idx in new_vocab.items():
        if word not in base_vocab:
            base_vocab[word] = len(base_vocab)
            new_new_vectors.append(new_vectors[idx])
    new_new_vectors = np.array(new_new_vectors)
    return base_vocab, np.concatenate([base_vectors, new_new_vectors], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings", nargs="+", action=required_length(2))
    parser.add_argument("--embedding_index", "--embedding-index", default="../configs/embeddings.json")
    parser.add_argument("--cache", default="../data")
    parser.add_argument("--output_format", "--output-format", default="w2v", choices=("dense", "glove", "w2v"))
    parser.add_argument("--output_label", "--output-label")
    parser.add_argument("--output_file", "--output-file")
    args = parser.parse_args()

    embedding_index = index_by_label(read_config_file_or_json(args.embedding_index))
    embeddings = {}

    args.output_label = f"{args.embeddings[0]}-vocab-extended-by-{'-'.join(args.embeddings[1:])}" if args.output_label is None else args.output_label

    dsz = []
    for embed_label in args.embeddings:
        embedding = embedding_index[embed_label]
        dsz.append(embedding['dsz'])
        embeddings[embed_label] = EmbeddingDownloader(
            embedding['file'],
            embedding['dsz'],
            embedding.get('sha1'),
            args.cache
        ).download()

    if len(set(dsz)) != 1:
        raise ValueError(f"For extending vocab experiments all embedding need to be the same size, got sizes of {dsz}")

    args.output_file = f"{embeddings[args.embeddings[0]]}-vocab-extended-by-{'-'.join(args.embeddings[1:])}" if args.output_file is None else args.output_file

    for embed_label in args.embeddings:
        embeddings[embed_label] = wv.read(embeddings[embed_label], False, False)

    base_vocab, base_vectors = embeddings[args.embeddings[0]]

    for embed_label in args.embeddings[1:]:
        new_vocab, new_vectors = embeddings[embed_label]
        base_vocab, base_vectors = combine_vocabs(base_vocab, base_vectors, new_vocab, new_vectors)
        del embeddings[embed_label]

    if args.output_format == "dense":
        write = wv.write_dense
    elif args.output_format == "w2v":
        write = wv.write_w2v
    else:
        write = wv.write_glove

    write(args.output_file, base_vocab, base_vectors)

    print(json.dumps({
        "label": args.output_label,
        "file": args.output_file,
        "dsz": base_vectors.shape[1],
    }, indent=2))


if __name__ == "__main__":
    main()
