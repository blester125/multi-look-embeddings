import argparse
from .utils import load_embeddings, load_dataset, read_conll_dataset, read_label_first


def main():
    parser = argparse.ArgumentParser(description="Calculate how many words in the dataset are attested in the pretrained vocab")
    parser.add_argument("embedding")
    parser.add_argument("--embeddings_index", "--embeddings-index", default="configs/embeddings.json")
    parser.add_argument("--cache", default="data")
    parser.add_argument("dataset")
    parser.add_argument("--datasets_index", "--datasets-index", default="configs/datasets.json")
    parser.add_argument("--data_type", "--data-type", default="conll", choices=("conll", "label-first"))
    parser.add_argument("--surface_index", "--surface-index", default=0, type=int)
    parser.add_argument("--delim")
    args = parser.parse_args()

    read = read_conll_dataset if args.data_type == "conll" else read_label_first
    dataset = load_dataset(
        args.dataset,
        args.datasets_index,
        args.cache,
        read=read,
        surface_index=args.surface_index,
        delim=args.delim
    )

    import pdb; pdb.set_trace()



if __name__ == "__main__":
    main()
