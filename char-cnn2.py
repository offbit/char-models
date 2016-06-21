import pandas as pd
import numpy as np
from gensim import utils

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, MaxPooling1D, Convolution1D
from keras.layers import LSTM, GRU
from keras.layers import Embedding
import numpy as np
from unidecode import unidecode
import re

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


maxlen = 60
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
        if char in char_indices.keys():
            X[i, t] = char_indices[char]
        else:
            X[i, t] = len(chars)
    nchar = next_chars[i]

    if nchar in char_indices.keys():
        y[i, char_indices[nchar]] = 1
    else:
        y[i, -1] = 1

embedding_size = 40
filter_length = [3, 2]
nb_filter = [64, 128]
pool_length = 2

model = Sequential()
model.add(Embedding(len(chars) + 1, embedding_size, input_length=maxlen))
model.add(Dropout(0.2))
model.add(Convolution1D(nb_filter=nb_filter[0],
                        filter_length=filter_length[0],
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))

model.add(Convolution1D(nb_filter=nb_filter[1],
                        filter_length=filter_length[1],
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))                        
# model.add(MaxPooling1D(pool_length=pool_length))

model.add(LSTM(300, return_sequences=True, dropout_W=0.2, dropout_U=0.2, consume_less='gpu'))
model.add(LSTM(300, return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu'))
model.add(Dropout(0.25))
model.add(Dense(len(chars) + 1))
model.add(Activation('softmax'))
# model.load_weights('weights')

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X, y, batch_size=256, nb_epoch=10, shuffle=True)

model.save_weights('char-cnn2.h5', overwrite=True)
