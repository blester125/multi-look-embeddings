import argparse
from typing import List, Dict, Tuple, Callable, Optional, Set
import word_vectors as wv
from file_or_name import file_or_name
from baseline.utils import EmbeddingDownloader, DataDownloader, read_conll, read_label_first_data
from mead.utils import index_by_label, read_config_file_or_json

STOP_WORDS: Set[str] = {
    "-docstart-",
    ",",
    ".",
    "in",
    "a",
    "the",
    "of",
    "to",
    "(",
    ")",
    "and",
    '"',
    "on",
    "'s",
    "for",
    "at",
    "with",
    "that",
    "from",
    "is",
    "by",
    "as",
    "had",
    "has",
    "was",
    "it",
    "but",
    "its",
    "who",
    "they",
}

def required_length(min_):
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if len(values) < min_:
                raise argparse.ArgumentTypeError(
                    f'Argument "{self.dest}" requires at least {min_} arguments, got {len(values)}'
                )
            setattr(args, self.dest, values)

    return RequiredLength


def parse_file_type(file_type):
    if file_type == "glove":
        return wv.FileType.GLOVE
    if file_type == "w2v":
        return wv.FileType.W2V
    if file_type == "dense":
        return wv.FileType.DENSE
    raise ValueError(f"Unsupported file type requested, got {file_type}")


def download_embeddings(embeddings: List[str], embeddings_index: str, cache: str) -> Dict[str, str]:
    embeddings_index = index_by_label(read_config_file_or_json(embeddings_index))
    embeddings_map = {}
    for embed_label in embeddings:
        embedding = embeddings_index[embed_label]
        embeddings_map[embed_label] = EmbeddingDownloader(
            embedding["file"], embedding["dsz"], embedding.get("sha1"), cache
        ).download()
    return embeddings_map


def download_dataset(dataset: str, dataset_index: str, cache: str) -> Dict[str, str]:
    dataset_index = index_by_label(read_config_file_or_json(dataset_index))
    return DataDownloader(dataset_index[dataset], cache).download()


def read_conll_dataset(
    file_name: str,
    surface_index: int = 0,
    delim: Optional[str] = None,
    **kwargs
) -> List[List[str]]:
    surfaces = []
    for sentence in read_conll(file_name, delim):
        surf = list(zip(*sentence))[surface_index]
        surfaces.append(surf)
    return surfaces


def read_label_first(
    file_name: str,
    **kwargs
) -> List[List[str]]:
    _, surfaces = read_label_first_data(file_name)
    return surfaces


def read_parallel(
    f: str,
    **keargs
) -> List[List[str]]:
    with open(f + ".txt") as f:
        return [l.strip().split() for l in f]


def load_dataset(
    dataset: str,
    dataset_index: str,
    cache: str,
    download: Callable = download_dataset,
    read: Callable = read_conll_dataset,
    **kwargs
) -> Dict[str, List[List[str]]]:
    dataset = download_dataset(dataset, dataset_index, cache)
    return {
        'train': read(dataset['train_file'], **kwargs),
        'dev': read(dataset['valid_file'], **kwargs),
        'test': read(dataset['test_file'], **kwargs)
    }


def read_embeddings(embeddings: Dict[str, str]) -> Dict[str, Tuple[wv.Vocab, wv.Vectors]]:
    return {k: wv.read(v, False, False) for k, v in embeddings.items()}


def load_embeddings(
    embeddings: List[str],
    embeddings_index: str,
    cache: str,
    download: Callable = download_embeddings,
    read: Callable = read_embeddings,
) -> Dict[str, Tuple[wv.Vocab, wv.Vectors]]:
    return read(download(embeddings, embeddings_index, cache))


def write_embeddings(file_name: str, vocab: wv.Vocab, vectors: wv.Vectors, file_type: wv.FileType):
    if file_type is wv.FileType.DENSE:
        write = wv.write_dense
    elif file_type is wv.FileType.W2V:
        write = wv.write_w2v
    elif file_type is wv.FileType.GLOVE:
        write = wv.write_glove
    else:
        raise ValueError(f"Unsupported file type requested, got {file_type}")
    write(file_name, vocab, vectors)
