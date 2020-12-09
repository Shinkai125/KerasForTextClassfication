"""
@file: embeddings.py
@time: 2020-12-02 16:41:16
"""
import logging
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

# 感谢https://fastnlp.readthedocs.io/zh/latest/tutorials/tutorial_3_embedding.html提供的下载链接
_EMBEDDING_TYPES = {
    'glove.42B.300d': {
        'file': 'glove.42B.300d.txt',
        'url': 'http://212.129.155.247/embedding/glove.42B.300d.zip'
    },

    'glove.6B.50d': {
        'file': 'glove.6B.50d.txt',
        'url': 'http://212.129.155.247/embedding/glove.6B.50d.zip'
    },

    'glove.6B.100d': {
        'file': 'glove.6B.100d.txt',
        'url': 'http://212.129.155.247/embedding/glove.6B.100d.zip'
    },

    'glove.6B.200d': {
        'file': 'glove.6B.200d.txt',
        'url': 'http://212.129.155.247/embedding/glove.6B.200d.zip'
    },

    'glove.6B.300d': {
        'file': 'glove.6B.300d.txt',
        'url': 'http://212.129.155.247/embedding/glove.6B.300d.zip'
    },

    'glove.840B.300d': {
        'file': 'glove.840B.300d.txt',
        'url': 'http://212.129.155.247/embedding/glove.840B.300d.zip'
    },

    'sgns.literature.word': {
        'file': 'sgns.literature.word.txt',
        'url': 'http://212.129.155.247/embedding/sgns.literature.word.txt.zip'
    }

}


def build_embeddings_index(pretrained_embeddings_file):
    logging.info('Building embeddings index...')
    index = {}
    with open(pretrained_embeddings_file, 'r') as f:
        for line in tqdm(f, desc="building embeddings index"):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            if len(vector) == 1:
                continue
            index[word] = vector
    return index


def build_embedding_weights(word_index, embeddings_index, return_oov=True):
    """
    Builds an embedding matrix for all words in vocab using embeddings_index
    """
    logging.info('Loading embeddings for all words in the corpus')

    embedding_dim = list(embeddings_index.values())[0].shape[-1]
    logging.info('Embedding dimensions: {}'.format(embedding_dim))

    pretrained_embeddings_vocab = set(embeddings_index.keys())
    tokens = list(word_index.keys())
    oov_tokens_in_pretrained_vocab = [t for t in tokens if t not in pretrained_embeddings_vocab and
                                      str.lower(t) not in pretrained_embeddings_vocab and
                                      str.upper(t) not in pretrained_embeddings_vocab and
                                      str.capitalize(t) not in pretrained_embeddings_vocab]

    logging.info('Percentage of tokens in pretrained embeddings: {}%'.format(
        (1 - len(oov_tokens_in_pretrained_vocab) / len(tokens)) * 100))

    # +1 since tokenizer words are indexed from 1. 0 is reserved for padding and unknown words.
    embedding_weights = np.zeros((len(word_index) + 1, embedding_dim))

    for word, i in word_index.items():
        word_vector = embeddings_index.get(word)
        if word_vector is not None:
            # Words not in embedding will be all zeros which can stand for padded words.
            embedding_weights[i] = word_vector
            continue

        word = word.lower()
        word_vector = embeddings_index.get(word)
        if word_vector is not None:
            embedding_weights[i] = word_vector
            continue

        word = word.upper()
        word_vector = embeddings_index.get(word)
        if word_vector is not None:
            embedding_weights[i] = word_vector
            continue

        word = word.capitalize()
        word_vector = embeddings_index.get(word)
        if word_vector is not None:
            embedding_weights[i] = word_vector
            continue

    if return_oov:
        return embedding_weights, oov_tokens_in_pretrained_vocab
    else:
        return embedding_weights


def get_embeddings_index(embedding_file_or_type='glove.42B.300d', cache_dir=""):
    """Retrieves embeddings index from embedding name. Will automatically download and cache as needed.
    Args:
        embedding_type: The embedding type to load.
    Returns:
        The embeddings indexed by word.
        :param embedding_file_or_type:
        :param cache_dir:
    """

    data_obj = _EMBEDDING_TYPES.get(embedding_file_or_type)
    if data_obj is None:
        if os.path.exists(embedding_file_or_type):
            file_path = embedding_file_or_type
            embeddings_index = build_embeddings_index(file_path)
            return embeddings_index
        else:
            raise ValueError("Embedding name should be one of '{}'".format(_EMBEDDING_TYPES.keys()))

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    file_path = tf.keras.utils.get_file(embedding_file_or_type, origin=data_obj['url'], extract=True,
                                        cache_dir=cache_dir, cache_subdir='embeddings')
    file_path = os.path.join(os.path.dirname(file_path), data_obj['file'])

    embeddings_index = build_embeddings_index(file_path)
    return embeddings_index


def convert_big_embeddings_to_small(big_filename, small_filename, number_line=500000):
    line_count = 0
    with open(big_filename) as reader, open(small_filename, 'w') as writer:
        for index, line in enumerate(tqdm(reader)):
            if line_count <= number_line:
                writer.write(line)
                line_count += 1
            else:
                break
    writer.close()


if __name__ == '__main__':
    word_index = {'<PAD>': 0, '我': 1, '爱': 2, '你': 3}
    # embed_index = get_embeddings_index(embedding_file_or_type='sgns.literature.word',
    #                                    cache_dir='.')
    embed_index = get_embeddings_index(embedding_file_or_type='./Tencent_AILab_ChineseEmbedding_500000.txt',
                                       cache_dir='.')
    embed_weights, oov = build_embedding_weights(word_index, embed_index, return_oov=True)
    print(embed_weights.shape)
    print(oov)

    convert_big_embeddings_to_small(big_filename='/home/Tencent_AILab_ChineseEmbedding.txt',
                                    small_filename='Tencent_AILab_ChineseEmbedding_1000000.txt',
                                    number_line=1000000)
