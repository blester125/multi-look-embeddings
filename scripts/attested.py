import argparse
from operator import or_
from itertools import chain
from functools import reduce
from collections import Counter
from typing import List, Set, Dict, Union
from tabulate import tabulate
from .utils import load_embeddings, load_dataset, read_conll_dataset, read_label_first


def lowercase(tokens: List[List[str]]) -> List[List[str]]:
    return [
        [t.lower() for t  in example]
        for example in tokens
    ]


def calc_attested(dataset: List[List[str]], vocab: Union[Set[str], Dict[str, int]]):
    counts = Counter(chain(*dataset))
    types = 0
    tokens = 0
    for word, count in counts.items():
        if word in vocab:
            types += 1
            tokens += count
    tok_count = sum(counts.values())
    return {
        "types": types,
        "tokens": tokens,
        "type_percent": types / len(counts),
        "tokens_percent": tokens / tok_count,
        "total_types": len(counts),
        "total_tokens": tok_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate how many words in the dataset are attested in the pretrained vocab")
    parser.add_argument("embeddings", nargs="+")
    parser.add_argument("--embeddings_index", "--embeddings-index", default="configs/embeddings.json")
    parser.add_argument("--cache", default="data")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--datasets_index", "--datasets-index", default="configs/datasets.json")
    parser.add_argument("--data_type", "--data-type", default="conll", choices=("conll", "label-first"))
    parser.add_argument("--surface_index", "--surface-index", default=0, type=int)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--delim")
    args = parser.parse_args()

    embeddings = load_embeddings(args.embeddings, args.embeddings_index, args.cache)

    read = read_conll_dataset if args.data_type == "conll" else read_label_first
    dataset = load_dataset(
        args.dataset,
        args.datasets_index,
        args.cache,
        read=read,
        surface_index=args.surface_index,
        delim=args.delim
    )

    if args.lowercase:
        dataset = {k: lowercase(v) for k, v in dataset.items()}

    if len(args.embeddings) > 1:
        joint_vocab = reduce(or_, (set(v[0].keys()) for v in embeddings.values()))

        for embedding in args.embeddings[1:]:
            embeddings[f"{embedding} - {args.embeddings[0]}"] = (set(embeddings[embedding][0].keys()) - set(embeddings[args.embeddings[0]][0].keys()), None)
        embeddings["joint"] = (joint_vocab, None)

    dataset['full'] = list(chain(*dataset.values()))

    attested = []
    for dataset_name, data in dataset.items():
        for name, (vocab, _) in embeddings.items():
            attested.append({
                "dataset": dataset_name,
                "embedding": name,
            })
            attested[-1].update(calc_attested(data, vocab))

    print(tabulate(attested, headers="keys", tablefmt="psql", floatfmt=".2f"))


if __name__ == "__main__":
    main()
