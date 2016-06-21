import pandas as pd
import numpy as np
from gensim import utils
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, MaxPooling1D, Convolution1D, Input, merge
from keras.layers import LSTM, GRU
from keras.layers import Embedding
import numpy as np
from unidecode import unidecode
import re

import sys
import os
 
total = len(sys.argv)
cmdargs = str(sys.argv)

print ("Script name: %s" % str(sys.argv[0]))
checkpoint = None
if len(sys.argv) == 2:
    if os.path.exists(str(sys.argv[1])):
        print ("Checkpoint : %s" % str(sys.argv[1]))
        checkpoint = str(sys.argv[1])

csv = 'cocacola_url_alexa_sentiment_text_title.csv'
data = pd.read_csv(csv)

txt = ''
docs = []
sentences = []
sentiments = []

for cont, sentiment in zip(data.text, data.sentiment):
    cont = unidecode(utils.to_unicode(cont))    
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', cont )
    sentences = [sent.strip().lower() for sent in sentences]
    if len(sentences) > 30:
        continue
    sentences = [' '.join(word for word in s.split() if len(word)<30) for s in sentences]
    docs.append(sentences)
#     txt += (' '.join(word for word in s.split() if len(word)<30).strip().lower())
    sentiments.append(sentiment)

for doc in docs:
    for s in doc:
        txt+=s


chars = set(txt)

print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


maxlen = 50
step = 3
sentences = []
next_chars = []
for i in range(0, len(txt) - maxlen, step):
    sentences.append(txt[i: i + maxlen])
    next_chars.append(txt[i+maxlen])
print('text len:', len(txt))
print('nb sequences:', len(sentences))

train_len = 500000

sentences = sentences[:train_len]
next_chars = next_chars[:train_len]

X = np.zeros((len(sentences), maxlen), dtype=np.int)
y = np.zeros((len(sentences), len(chars) + 1), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t] = char_indices[char]
    y[i, char_indices[next_chars[i]]] = 1
    
# Model parameters
max_features = len(chars)
char_embedding = 40

layers = [256, 256]

sequence = Input(shape=(maxlen,), dtype='int32')
embedded = Embedding(max_features, char_embedding, input_length=maxlen)(sequence)

forwards = LSTM(layers[0], return_sequences=True, dropout_W=0.2, dropout_U=0.2, consume_less='gpu')(embedded)
forwards = LSTM(layers[1], return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu')(forwards)

backwards = LSTM(layers[0],return_sequences=True, dropout_W=0.2, dropout_U=0.2, consume_less='gpu', go_backwards=True)(embedded)
backwards = LSTM(layers[1],return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu', go_backwards=True)(backwards)

merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
merged = Dropout(0.5)(merged)

output = Dense(max_features, activation='softmax')(merged)

model = Model(input=sequence, output=output)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

if checkpoint:
    print("Resuming....")
    model.load_weights(checkpoint)

model.fit(X, y, batch_size=256, nb_epoch=1, shuffle=True)

model.save_weights('checkpoints/biderectional.h5', overwrite=True)
