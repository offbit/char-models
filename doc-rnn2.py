import pandas as pd
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, Dropout, MaxPooling1D, Convolution1D
from keras.layers import LSTM, Lambda, merge, Masking
from keras.layers import Embedding, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import numpy as np
import tensorflow as tf
import re
from keras import backend as K
import keras.callbacks
import sys
import os


def binarize(x, sz=71):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))


def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 71


def max_1d(x):
    return K.max(x, axis=1)


def striphtml(html):
    p = re.compile(r'<.*?>')
    return p.sub('', html)


def clean(s):
    return re.sub(r'[^\x00-\x7f]', r'', s)

total = len(sys.argv)
cmdargs = str(sys.argv)

print ("Script name: %s" % str(sys.argv[0]))
checkpoint = None
if len(sys.argv) == 2:
    if os.path.exists(str(sys.argv[1])):
        print ("Checkpoint : %s" % str(sys.argv[1]))
        checkpoint = str(sys.argv[1])


data = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
txt = ''
docs = []
sentences = []
sentiments = []

for cont, sentiment in zip(data.review, data.sentiment):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', clean(striphtml(cont)))
    sentences = [sent.lower() for sent in sentences]
    docs.append(sentences)
    sentiments.append(sentiment)

num_sent = []
for doc in docs:
    num_sent.append(len(doc))
    for s in doc:
        txt += s

chars = set(txt)

print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print('Sample doc{}'.format(docs[1200]))

maxlen = 512
max_sentences = 15

X = np.ones((len(docs), max_sentences, maxlen), dtype=np.int64) * -1
y = np.array(sentiments)

for i, doc in enumerate(docs):
    for j, sentence in enumerate(doc):
        if j < max_sentences:
            for t, char in enumerate(sentence[-maxlen:]):
                X[i, j, (maxlen-1-t)] = char_indices[char]

print('Sample chars in X:{}'.format(X[1200, 2]))
print('y:{}'.format(y[1200]))

ids = np.arange(len(X))
np.random.shuffle(ids)

# shuffle
X = X[ids]
y = y[ids]

X_train = X[:20000]
X_test = X[22500:]

y_train = y[:20000]
y_test = y[22500:]

filter_length = [5, 3, 3]
nb_filter = [128, 128, 256]
pool_length = 2
layers = [128, 128]

sequence = Input(shape=(max_sentences, maxlen), dtype='int64')


in_sentence = Input(shape=(maxlen,), dtype='int64')
embedded = Lambda(binarize, output_shape=binarize_outshape)(in_sentence)

embedded = Convolution1D(nb_filter=nb_filter[0],
                         filter_length=filter_length[0],
                         border_mode='valid',
                         activation='relu',
                         init='glorot_normal',
                         subsample_length=1)(embedded)

embedded = Dropout(0.1)(embedded)
embedded = MaxPooling1D(pool_length=pool_length)(embedded)
embedded = Convolution1D(nb_filter=nb_filter[1],
                         filter_length=filter_length[1],
                         border_mode='valid',
                         activation='relu',
                         init='glorot_normal',
                         subsample_length=1)(embedded)

embedded = Dropout(0.1)(embedded)
embedded = MaxPooling1D(pool_length=pool_length)(embedded)

embedded = Convolution1D(nb_filter=nb_filter[2],
                         filter_length=filter_length[2],
                         border_mode='valid',
                         activation='relu',
                         init='glorot_normal',
                         subsample_length=1)(embedded)

embedded = Dropout(0.1)(embedded)
embedded = MaxPooling1D(pool_length=pool_length)(embedded)


forward_sent = LSTM(layers[0], return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu')(embedded)
backward_sent = LSTM(layers[0], return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu', go_backwards=True)(embedded)

sent_encode = merge([forward_sent, backward_sent], mode='concat', concat_axis=-1)
sent_encode = Dropout(0.5)(sent_encode)

encoder = Model(input=in_sentence, output=sent_encode)
encoded = TimeDistributed(encoder)(sequence)

forwards = LSTM(80, return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu')(encoded)
backwards = LSTM(80, return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu', go_backwards=True)(encoded)

merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
output = Dropout(0.5)(merged)
output = Dense(128, activation='relu')(output)
output = Dropout(0.5)(output)
output = Dense(1, activation='sigmoid')(output)

model = Model(input=sequence, output=output)


if checkpoint:
    model.load_weights(checkpoint)

file_name = os.path.basename(sys.argv[0]).split('.')[0]

check_cb = keras.callbacks.ModelCheckpoint('checkpoints/'+file_name+'.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                           verbose=0, save_best_only=True, mode='min')

earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
logger_cb = keras.callbacks.BaseLogger()

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10,
          nb_epoch=30, shuffle=True, callbacks=[check_cb])

print logger_cb
