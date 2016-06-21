import pandas as pd

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, Dropout, MaxPooling1D, Convolution1D
from keras.layers import LSTM, Lambda, merge
from keras.layers import Embedding, TimeDistributed
from keras.optimizers import SGD
import numpy as np
import tensorflow as tf
import re
from keras import backend as K
import keras.callbacks
import sys
import os


def binarize(x, sz=72):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))


def binarize_outshape(in_shape):
    return (in_shape[0], in_shape[1], 72)


def max_1d(X):
    return K.max(X, axis=1)


def striphtml(s):
    p = re.compile(r'<.*?>')
    return p.sub('', s)


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
char_indices = dict((c, i + 1) for i, c in enumerate(chars))
indices_char = dict((i + 1, c) for i, c in enumerate(chars))

print('Sample doc{}'.format(docs[1200]))

maxlen = 512
max_sentences = 15

X = np.ones((len(docs), max_sentences, maxlen), dtype=np.int64) * -1
y = np.array(sentiments)

for i, doc in enumerate(docs):
    for j, sentence in enumerate(doc):
        if j < max_sentences:
            for t, char in enumerate(sentence[-maxlen:]):
                X[i, j, t] = char_indices[char]
                # X[i, j, t, char_indices[char]] = 1
    # y[i, sentiments[i]] = 1


# print('Sample X:{}'.format(X[1200, 2]))
# print('y:{}'.format(y[1200]))

ids = np.arange(len(X))
np.random.shuffle(ids)

# shuffle
X = X[ids]
y = y[ids]

X_train = X[:20000]
X_test = X[22500:]

y_train = y[:20000]
y_test = y[22500:]


def char_block(in_layer, nb_filter=[64, 100], filter_length=[3, 3], subsample=[2, 1], pool_length=None):
    block = in_layer
    for i in range(len(nb_filter)):

        block = Convolution1D(nb_filter=nb_filter[i],
                              filter_length=filter_length[i],
                              border_mode='valid',
                              activation='relu',
                              init='glorot_normal',
                              subsample_length=subsample[i])(block)
        # block = BatchNormalization()(block)

        block = Dropout(0.2)(block)
        if i == len(nb_filter)-1:
            continue
        if pool_length:
            block = MaxPooling1D(pool_length=pool_length)(block)

    block = Lambda(max_1d, output_shape=(nb_filter[-1],))(block)
    return block

max_features = len(chars) + 1
char_embedding = 40
drop_p = 0.3
sequence = Input(shape=(max_sentences, maxlen), dtype='int64')

in_sentence = Input(shape=(maxlen, ), dtype='int64')
# embedded = Embedding(max_features, char_embedding, input_length=maxlen)(in_sentence)
embedded = Lambda(binarize, output_shape=binarize_outshape)(in_sentence)

block1 = char_block(embedded, [64, 128], filter_length=[3, 3], subsample=[1, 1], pool_length=2)
block2 = char_block(embedded, [96, 160], filter_length=[5, 3], subsample=[2, 1], pool_length=2)
block3 = char_block(embedded, [128, 192], filter_length=[7, 3], subsample=[2, 1], pool_length=2)

sent_encode = merge([block1, block2, block3], mode='concat', concat_axis=-1)
# sent_encode = Dropout(drop_p)(sent_encode)
sent_encode = Dense(256, activation='relu', init='glorot_normal')(sent_encode)
sent_encode = Dropout(drop_p)(sent_encode)

encoder = Model(input=in_sentence, output=sent_encode)
encoded = TimeDistributed(encoder)(sequence)

lstm_h = 100
forwards = LSTM(lstm_h, return_sequences=False, dropout_W=0.2, dropout_U=0.2,
                consume_less='gpu')(encoded)
backwards = LSTM(lstm_h, return_sequences=False, dropout_W=0.2, dropout_U=0.2,
                 consume_less='gpu', go_backwards=True)(encoded)

merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
# output = Dropout(drop_p)(merged)
# output = Dense(128, activation='relu', init='he_normal')(output)
# output = Dropout(drop_p)(output)
output = Dense(1, activation='sigmoid', init='glorot_normal')(merged)

model = Model(input=sequence, output=output)


if checkpoint:
    model.load_weights(checkpoint)

file_name = os.path.basename(sys.argv[0]).split('.')[0]


def schedule(epoch):
    if epoch < 5:
        return 0.01

    elif 5 < epoch < 10:
        return 0.001
    elif 10 < epoch < 20:
        return 0.0001
    else:
        return 0.00005

check_cb = keras.callbacks.ModelCheckpoint('checkpoints/'+file_name+'.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                           verbose=0, save_best_only=True, mode='min')
earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
scheduler = keras.callbacks.LearningRateScheduler(schedule)


opt = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10,
          nb_epoch=30, shuffle=True, callbacks=[check_cb, earlystop_cb])

# loss = model.evaluate(X_test, y_test, batch_size=16)
# print loss

# model.save_weights('checkpoints/doc-cnn3.h5', overwrite=True)
