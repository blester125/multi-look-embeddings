import argparse
from typing import List, Dict, Tuple, Callable
import word_vectors as wv
from baseline.utils import EmbeddingDownloader
from mead.utils import index_by_label, read_config_file_or_json


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
