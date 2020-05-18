import random
import argparse
from typing import Optional
import numpy as np
import tensorflow as tf
from eight_mile.tf.embeddings import LookupTableEmbeddings
from .utils import load_embeddings


def norm_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = (norms == 0) + norms
    return vectors / norms


def prune_vectors(words, vocab, vectors):
    new_vectors = np.zeros((len(words), vectors.shape[1]), dtype=vectors.dtype)
    new_vocab = {}
    for i, word in enumerate(words):
        new_vectors[i, :] = vectors[vocab[word]]
        new_vocab[word] = i
    return new_vocab, new_vectors


class EmbeddingMapping(tf.keras.layers.Layer):
    def __init__(self, src, tgt, bias: bool = True, trainable: bool=True, dtype=tf.float32, name: Optional[str] = None):
        super().__init__(trainable=trainable, dtype=dtype, name=name)
        self.src = LookupTableEmbeddings(trainable=False, name="source_embeddings", weights=src)
        self.tgt = LookupTableEmbeddings(trainable=False, name="target_embeddings", weights=tgt)
        self.mapping = tf.keras.layers.Dense(self.tgt.output_dim, activation=None, use_bias=bias)

    def call(self, x):
        return self.mapping(self.src(x))

    def loss(self, x, y):
        w = self(x)
        z = self.tgt(y)
        return tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(w, z)), axis=1))


def main():
    parser = argparse.ArgumentParser(description="Learn a transformation between two embedding spaces.")
    parser.add_argument("embeddings", nargs=2)
    parser.add_argument("--embeddings_index", "--embeddings-index", default="configs/embeddings.json")
    parser.add_argument("--cache", default="data")
    parser.add_argument("--no_bias", "--no-bias", action="store_false", dest="bias")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batchsz", default=64, type=int)
    parser.add_argument("--optim", default="adam")
    parser.add_argument("--eta", default=3e-4, type=float)
    args = parser.parse_args()

    embeddings = load_embeddings(args.embeddings, args.embeddings_index, args.cache)

    src_vocab, src_vectors = embeddings[args.embeddings[0]]
    tgt_vocab, tgt_vectors = embeddings[args.embeddings[1]]

    shared_words = list(set(src_vocab.keys()) & set(tgt_vocab.keys()))

    src_vocab, src_vectors = prune_vectors(shared_words, src_vocab, src_vectors)
    tgt_vocab, tgt_vectors = prune_vectors(shared_words, tgt_vocab, tgt_vectors)

    src_vectors = norm_vectors(src_vectors)
    tgt_vectors = norm_vectors(tgt_vectors)

    m = EmbeddingMapping(
        src_vectors,
        tgt_vectors,
        args.bias
    )
    optim = tf.keras.optimizers.Adam(learning_rate=args.eta)

    for epoch in range(args.epochs):
        total_loss = 0
        random.shuffle(shared_words)
        for batch in (shared_words[i : i + args.batchsz] for i in range(0, len(shared_words), args.batchsz)):
            x = [src_vocab[w] for w in batch]
            y = [tgt_vocab[y] for y in batch]
            with tf.GradientTape() as tape:
                loss = m.loss(x, y)
                grads = tape.gradient(loss, m.trainable_variables)
                optim.apply_gradients(zip(grads, m.trainable_variables))
            total_loss += loss.numpy().item() * len(batch)
        print(f"Epoch: {epoch + 1} Loss: {total_loss / len(shared_words)}")
        print(f"Frobenius Norm of the weight: {tf.norm(m.trainable_variables[0], ord='fro', axis=(0, 1)).numpy()}")



if __name__ == "__main__":
    main()
