import os
import numpy
from keras.layers import LSTM, Dense
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
import nltk
from nltk.tokenize import word_tokenize
from keras.models import load_model
import random
from sklearn.preprocessing import LabelEncoder
#open file
f = list(open("wwhitman-clean-processed.txt","r"))
for x in range(len(f)):
    f[x] = f[x].lower()
print("Done!")

for x in range(len(f)):

    f[x] = word_tokenize(f[x])
for item in f:
    if len(item) is 0:
        item.append("EOP")
    else:
        item.append("EOL")
flat_list = [item for sublist in f for item in sublist]
print(flat_list)
#tokenize
print("Number of words in text:",len(flat_list))
wordsNoDup=list(set(flat_list))
print("number of words without duplicates:",len(wordsNoDup))
#====================================================================================
values = array(flat_list)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# one hot encode
encoded = to_categorical(integer_encoded)
print(encoded)
print("encoded shape:",encoded.shape)
# invert encoding
# inverted = argmax(encoded[0])
# print(inverted)
print("encoded lines: ",len(encoded))

dictOneHot = dict()
dictEncode=dict()
for x in range(len(integer_encoded)):
    dictOneHot[integer_encoded[x]] = flat_list[x]
    dictEncode[integer_encoded[x]]=encoded[x]
print("dict one hot",dictOneHot)
print("dict encode one hot,",dictEncode)
#=====================================================================================
#process data and labels:   encoded is array of 1 HOT

k=5
data=list()
labels=list()

for i in range(len(encoded)):
    if(i+k+1==len(encoded)):
        break
    temp = list()
    for j in range(k+1):
        if(k is j):
            labels.append(encoded[i+j])
        else:
            temp.append(encoded[i+j])
    data.append(temp)

data=numpy.asarray(data)
labels=numpy.asarray(labels)

print(labels.shape)
print(data.shape)

exists =os.path.isfile("Task1Model.h5")
if not exists:
    model= Sequential()

    model.add(LSTM(250, input_shape=(data.shape[1], data.shape[2]), return_sequences=True))
    model.add(LSTM(125, input_shape=(data.shape[1], data.shape[2]), return_sequences=False))
    model.add(Dense(data.shape[2],activation="softmax"))
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    model.fit(data,labels,epochs=50,batch_size=64)

    model.save("Task1Model.h5")
    model.save_weights("Task1Model_weights.h5")
else:
    model = load_model("Task1Model.h5")
    model.load_weights("Task1Model_weights.h5")

seed = list()
poemEncoded=list()
temp=list()
for x in range(k):

    randomEncoded= labels[random.randint(0, len(data) - 1)]
    temp.append(randomEncoded)
    poemEncoded.append(randomEncoded)
seed.append(temp)
poemEncoded.append(temp)

for i in range(50):
    temp = list()
    seed = numpy.asarray(seed)
    print("seed shape",seed.shape)
    prediction = model.predict(seed)
    inverted = argmax(prediction)
    print("predic shape",prediction.shape)
    print("inverted: ", inverted)
    print("the word is:",dictOneHot[inverted])
    test=dictEncode[inverted]
    print("the word one hot is:",dictEncode[inverted])
    print("the one hot is after argmax:",argmax(test))
    for i in range(1, k):
        temp.append(seed[0][i])
    temp.append(test)
    poemEncoded.append(test)
    seed=[]
    seed.append(temp)


print("poem:",poemEncoded)
poem=list()
for word in poemEncoded:
    inverted = argmax(word)
    try:
        word=dictOneHot[inverted]
        poem.append(word)
        print("inverted:", inverted)
        print("word:", word)
    except:
        print(inverted)
        pass

print(poem)