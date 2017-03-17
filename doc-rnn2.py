import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D
from keras.layers import LSTM, Lambda
from keras.layers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
import numpy as np
import tensorflow as tf
import re
import keras.callbacks
import sys
import os


def binarize(x, sz=71):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))


def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 71


def striphtml(html):
    p = re.compile(r'<.*?>')
    return p.sub('', html)


def clean(s):
    return re.sub(r'[^\x00-\x7f]', r'', s)


# record history of training
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


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
                X[i, j, (maxlen - 1 - t)] = char_indices[char]

print('Sample chars in X:{}'.format(X[1200, 2]))
print('y:{}'.format(y[1200]))

ids = np.arange(len(X))
np.random.shuffle(ids)

# shuffle
X = X[ids]
y = y[ids]

X_train = X[:20000]
X_test = X[20000:]

y_train = y[:20000]
y_test = y[20000:]

filter_length = [5, 3, 3]
nb_filter = [196, 196, 256]
pool_length = 2
# document input
document = Input(shape=(max_sentences, maxlen), dtype='int64')
# sentence input
in_sentence = Input(shape=(maxlen,), dtype='int64')
# char indices to one hot matrix, 1D sequence to 2D 
embedded = Lambda(binarize, output_shape=binarize_outshape)(in_sentence)
# embedded: encodes sentence
for i in range(len(nb_filter)):
    embedded = Conv1D(filters=nb_filter[i],
                      kernel_size=filter_length[i],
                      padding='valid',
                      activation='relu',
                      kernel_initializer='glorot_normal',
                      strides=1)(embedded)

    embedded = Dropout(0.1)(embedded)
    embedded = MaxPooling1D(pool_size=pool_length)(embedded)

bi_lstm_sent = \
    Bidirectional(LSTM(128, return_sequences=False, dropout=0.15, recurrent_dropout=0.15, implementation=0))(embedded)

# sent_encode = merge([forward_sent, backward_sent], mode='concat', concat_axis=-1)
sent_encode = Dropout(0.3)(bi_lstm_sent)
# sentence encoder
encoder = Model(inputs=in_sentence, outputs=sent_encode)
encoder.summary()

encoded = TimeDistributed(encoder)(document)
# encoded: sentences to bi-lstm for document encoding 
b_lstm_doc = \
    Bidirectional(LSTM(128, return_sequences=False, dropout=0.15, recurrent_dropout=0.15, implementation=0))(encoded)

output = Dropout(0.3)(b_lstm_doc)
output = Dense(128, activation='relu')(output)
output = Dropout(0.3)(output)
output = Dense(1, activation='sigmoid')(output)

model = Model(inputs=document, outputs=output)

model.summary()

if checkpoint:
    model.load_weights(checkpoint)

file_name = os.path.basename(sys.argv[0]).split('.')[0]
check_cb = keras.callbacks.ModelCheckpoint('checkpoints/' + file_name + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                                           monitor='val_loss',
                                           verbose=0, save_best_only=True, mode='min')
earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
history = LossHistory()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10,
          epochs=5, shuffle=True, callbacks=[earlystop_cb, check_cb, history])

# just showing access to the history object
print history.losses
print history.accuracies
