import pandas as pd
import numpy as np
from gensim import utils

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed, Convolution1D
from keras.layers import LSTM, GRU
from keras.layers import Embedding
import numpy as np

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

maxlen = 128
step = 3
sentences = []
next_chars = []
for i in range(0, len(txt) - maxlen, step):
    sentences.append(txt[i: i + maxlen])
    next_chars.append(txt[i+maxlen])
print('nb sequences:', len(sentences))

sentences = sentences[:10000]
next_chars = next_chars[:10000]
# X = np.zeros((len(sentences), maxlen, len(chars)+1), dtype=np.bool)
X = np.zeros((len(sentences), maxlen), dtype=np.int)
y = np.zeros((len(sentences), len(chars) + 1), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        if char in char_indices.keys():
            X[i, t] = char_indices[char]
        # X[i, t, char_indices[char]] = 1
        else:
            X[i, t] = len(chars) + 1
        # X[i, t, -1] = 1
    nchar = next_chars[i]

    if nchar in char_indices.keys():
        y[i, char_indices[nchar]] = 1
    else:
        y[i, -1] = 1

embedding_size = 30
model = Sequential()
model.add(Embedding(len(chars) + 1, embedding_size, input_length=maxlen))
model.add(LSTM(300, return_sequences=True, dropout_W=0.2, dropout_U=0.2, consume_less='gpu'))
model.add(LSTM(300, return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu'))
model.add(Dropout(0.3))
model.add(Dense(len(chars) + 1))
model.add(Activation('softmax'))
model.load_weights('weights')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(X, y, batch_size=128, nb_epoch=20)

model.save_weights('weights', overwrite=True)
