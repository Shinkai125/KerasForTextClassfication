"""
@file: model.py
@time: 2020-12-17 16:12:50
"""
from tensorflow.keras.layers import *
from layers import Attention
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy


def lsrm_attention_model(dataset):
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
                  metrics=[SparseCategoricalAccuracy(name="acc")], )
    model.summary()
