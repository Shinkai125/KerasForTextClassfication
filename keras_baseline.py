"""
@file: train.py.py
@time: 2020-12-02 18:12:16
"""
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import Model
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from embeddings import get_embeddings_index, build_embedding_weights
from layers import Attention
from preprogress import clean_urls
from tokenizer import tokenize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def clean_text(text: str):
    text = clean_urls(text)
    text = ' '.join(text.split())
    return text


class Dataset(object):
    def __init__(self, train_filename, valid_filename, test_filename, label_map, vocab_size, max_len, seed=42):
        self.train_filename = train_filename
        self.valid_filename = valid_filename
        self.test_filename = test_filename
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.label_map = label_map
        self.word_index = None
        self.index_word = None
        self.num_classes = len(label_map)
        self.map_label = dict((v, k) for k, v in self.label_map.items())
        self.train_texts, self.train_label, self.train_raw_texts = self.read_dataset(train_filename)
        self.valid_texts, self.valid_label, self.valid_raw_texts = self.read_dataset(valid_filename)
        self.test_texts, self.test_label, self.test_raw_texts = self.read_dataset(test_filename)

        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test = self.token_padding()
        self.embedding_matrix = self.build_embeddings_matrix()
        self.kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(self.X_train, self.y_train)

    def read_dataset(self, filename, no_label=False):
        logging.info('Reading dataset from csv file:  %s', filename)
        df = pd.read_csv(filename, sep='\t')
        print(df.head(5))
        raw_texts = df['text_a'].tolist()
        logging.info('Cleaning texts')
        cleaned_texts = [clean_text(i) for i in raw_texts]
        logging.info('Tokenizing texts')
        token_texts = tokenize(cleaned_texts, sep=' ', join=True)
        if no_label:
            return token_texts, raw_texts
        else:
            labels = df['label'].tolist()
            mapped_labels = [self.label_map[label] for label in labels]
            return token_texts, mapped_labels, raw_texts

    def build_embeddings_matrix(self):
        embed_index = get_embeddings_index(embedding_file_or_type='Tencent_AILab_ChineseEmbedding_1000000.txt', cache_dir='.')
        embed_weights, oov = build_embedding_weights(self.word_index, embed_index, return_oov=True)
        print(oov[:50])
        return embed_weights

    def token_padding(self):
        tokenizer = Tokenizer(num_words=self.vocab_size, filters='', lower=False, split=' ')

        logging.info('Fitting tokenizer...')
        tokenizer.fit_on_texts(self.train_texts + self.valid_texts + self.test_texts)
        self.word_index = tokenizer.word_index
        self.index_word = tokenizer.index_word

        logging.info('Building training set...')
        X_train = tokenizer.texts_to_sequences(self.train_texts)
        y_train = self.train_label

        logging.info('Building validation set...')
        X_valid = tokenizer.texts_to_sequences(self.valid_texts)
        y_valid = self.valid_label

        logging.info('Building test set ...')
        X_test = tokenizer.texts_to_sequences(self.test_texts)

        logging.info('Padding sequences...')
        X_train = pad_sequences(X_train, maxlen=self.max_len, padding='post')
        X_valid = pad_sequences(X_valid, maxlen=self.max_len, padding='post')
        X_test = pad_sequences(X_test, maxlen=self.max_len, padding='post')

        return np.array(X_train), np.array(y_train), np.array(X_valid), np.array(y_valid), np.array(X_test)


def build_model(dataset):
    sequence_input = Input(shape=(dataset.max_len,), dtype=tf.int32)

    embedding_layer = Embedding(*dataset.embedding_matrix.shape,
                                weights=[dataset.embedding_matrix],
                                trainable=False)

    x = embedding_layer(sequence_input)
    x = SpatialDropout1D(0.3)(x)
    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(128, return_sequences=True)(x)

    att = Attention(dataset.max_len)(x)
    avg_pool1 = GlobalAveragePooling1D()(x)
    max_pool1 = GlobalMaxPooling1D()(x)
    hidden = concatenate([att, avg_pool1, max_pool1])

    hidden = Dense(512, activation='relu')(hidden)
    hidden = Dense(128, activation='relu')(hidden)
    out = Dense(dataset.num_classes, activation='softmax')(hidden)
    model = Model(sequence_input, out)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")], )
    model.summary()

    return model


def train_model(dataset):
    cb1 = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    cb2 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, cooldown=0, min_lr=0.0001)
    cvscores = []
    for train, test in dataset.kfold:
        model = build_model(dataset)
        model.fit(dataset.X_train[train], dataset.y_train[train]
                  , validation_data=(dataset.X_train[test], dataset.y_train[test])
                  , epochs=4, batch_size=64, verbose=1
                  , callbacks=[cb1, cb2])
        scores = model.evaluate(dataset.X_train[test], dataset.y_train[test], verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


if __name__ == '__main__':
    # chnsenticorp: https://bj.bcebos.com/paddlehub-dataset/chnsenticorp.tar.gz
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    dataset = Dataset(train_filename='chnsenticorp/train.tsv',
                      valid_filename='chnsenticorp/dev.tsv',
                      test_filename='chnsenticorp/test.tsv',
                      label_map={0: 0, 1: 1},
                      vocab_size=30000,
                      max_len=1024)
    train_model(dataset)
