


 
import pickle


from unidecode import unidecode
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from string import punctuation   
 

import keras 
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Input
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten
from keras.models import Model
#from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import RMSprop 
import io
def leerStopWords(ruta):
    StopWords = []
    with io.open(ruta,'r', encoding="utf-8") as file:
        for line in file:
            if '#' not in line:
                for token in line.split(','):
                    word = StopWords.append(unidecode(token).rsplit()[0])
                    if word:
                        if len(StopWords) == 0:
                            StopWords.append(word)
                        elif word not in StopWords:
                            StopWords.append(word)
    return StopWords

CustomStopWords = leerStopWords('Stopwords.txt')
non_words = list(punctuation)
non_words.extend(['¿', '¡', 'q', 'd', 'x', 'xq', '...', '..','…','``','`',"'","''","<",">","<div>","</div>"
,"<br>"])
stop_words = CustomStopWords + non_words

def preproccess_data(comentarios):
    #Tokenize
    print("tokenize")
    print(comentarios)
    tokens = [word_tokenize(com,"spanish") for com in comentarios]
    #Remove StopWords
    cleanTokens = []
    for words in tokens:
        cleanWords = [unidecode(token.lower()) for token in words if token.lower() not in stop_words]
        cleanTokens.append(cleanWords)
    print("cleantokens")
    print(cleanTokens)
    # Data Returning
    coms = []
    for x in cleanTokens:
        oracion = ""
        for token in x:
            oracion += (" " + token)
        coms.append(oracion)
    print("coms")
    print(coms)
    return coms

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.optimizers import Adam
lstm_out = 16 #MP-AGREGADA
max_words = 10000
maxlen = 130
embedding_dim = 300 

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
#model.add(LSTM(lstm_out))
model.add(layers.Bidirectional(LSTM(lstm_out))) #MP-AGREGADA
model.add(Dropout(0.8)) #MP-AGREGADA#MP-AGREGADA
#model.add(Flatten())
model.add(Dense(6, activation='sigmoid'))#MP-AGREGADA
model.add(Dense(3, activation='softmax'))#MP-AGREGADA
#model.add(Dense(32, activation='relu'))#MP-porque en dense 32
#model.add(Dense(3, activation='sigmoid'))
#model.summary()
#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# compile the model
opt = Adam(learning_rate=0.001) #MP-AGREGADA
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])#MP-AGREGADA
# summarize the model
print(model.summary()) #MP-AGREGADA
# fit the model

model.load_weights('modelo_word2vec_weights24.hdf5')