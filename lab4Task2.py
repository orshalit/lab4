import os

from pickle import dump
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, TimeDistributed, RepeatVector
from keras.models import load_model
from numpy import argmax
import numpy
from nltk import word_tokenize
from numpy import array
from sklearn.preprocessing import LabelEncoder

import TranslationDataPrep as prep
import TranslationHelp as helper


# === Integer Encode ===
maxLength = 0
# open file
targetFileRAW = open("wwhitman-clean-processed.txt", "r", encoding="utf-8")
f = list(open("wwhitman-clean-processed.txt", "r"))
for x in range(len(f)):
    f[x] = f[x].lower()
print("Done!")

for x in range(len(f)):
    f[x] = word_tokenize(f[x])
for item in f:
    if len(item) > maxLength:
        maxLength = len(item)
    if len(item) is 0:
        item.append("EOP")
    else:
        item.append("EOL")
flat_list = [item for sublist in f for item in sublist]
print(flat_list)
# tokenize
print("Number of words in text:", len(flat_list))
wordsNoDup = list(set(flat_list))
print("number of words without duplicates:", len(wordsNoDup))
# ==================== Integer Encode Text===================================
values = array(flat_list)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

# ===== Integer Encode language using Pickle and given Functions ============
# <! -- Create Pickle Files if not exists -- !>
pairsFilePath = "rus.txt"
allPickle, cleanPickle, trainPickle, testPickle = "allpickle.obj", "cleanPickle.obj", "trainPickle.obj", "testPickle.obj"
exists = os.path.isfile(allPickle)
if not exists:
    maxNumOfSentences, trainTestSplit = 10000, 0.9
    a = prep.TranslationDataPrep(pairsFilePath, allPickle, cleanPickle, trainPickle, testPickle, maxNumOfSentences, trainTestSplit)

all=helper.load_clean_sentences(allPickle)
train=helper.load_clean_sentences(trainPickle)
test=helper.load_clean_sentences(testPickle)
dataset=helper.load_clean_sentences(cleanPickle)
print(len(dataset))
#process DATA and LABELS of language A and B
oneHot="oneHotLabels.obj"
savedPickle=os.path.isfile("oneHotLabels.obj")
if not savedPickle:
    uncodedData=list()
    uncodedLabels=list()
    for i in range(len(dataset)):
        uncodedData.append(dataset[i][0])
        uncodedLabels.append(dataset[i][1])
    # region tokenize, encode DATA!
    tokenizeData=helper.create_tokenizer(uncodedData)
    encodedData=helper.encode_sequences(tokenizeData, helper.max_length(uncodedData),uncodedData)
    # tokenize, encode, 1 hot LABELS
    tokenizeLabels=helper.create_tokenizer(uncodedLabels)
    encodedLabels=helper.encode_sequences(tokenizeLabels,helper.max_length(uncodedLabels),uncodedLabels)
    oneHotLabels=helper.encode_output(encodedLabels,len(set(uncodedLabels)))
    dump(oneHotLabels, open("oneHotLabels", 'wb'))
    print('Saved: %s' % oneHot)
    dump(encodedData, open("encodedData", 'wb'))
    dump(uncodedData, open("uncodedData", 'wb'))
# endregion
else:
    print("here")
    uncodedData=helper.load_clean_sentences("uncodedData")
    encodedData=helper.load_clean_sentences("encodedData")
    oneHotLabels=helper.load_clean_sentences(oneHot)


src_timesteps=helper.max_length(uncodedData)
TARGET_SIZE=oneHotLabels.shape[2]


print(type(src_timesteps))
trainX=encodedData[:round(len(encodedData)*0.9)]
testX=encodedData[round(len(encodedData)*0.9):]
trainY=oneHotLabels[:round(len(oneHotLabels)*0.9)]
testY=oneHotLabels[round(len(oneHotLabels)*0.9):]
print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

exists = os.path.isfile("Task2Model.h5")
if not exists:
    N_NERURONS=256
    model= Sequential()
    model.add(Embedding(len(encodedData),N_NERURONS,input_length=src_timesteps,mask_zero=True))
    model.add(LSTM(N_NERURONS, input_shape=(trainY.shape[1],trainY.shape[2]),return_sequences=False))
    model.add(RepeatVector(trainY.shape[1]))
    model.add(LSTM(N_NERURONS//2, return_sequences=True))
    model.add(TimeDistributed(Dense(TARGET_SIZE,activation="softmax")))
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    model.fit(trainX,trainY,validation_data=(testX,testY),epochs=30,batch_size=50)

    model.save("Task2Model.h5")
    model.save_weights("Task2Model_weights.h5")
else:
    model = load_model("Task2Model.h5")
    model.load_weights("Task2Model_weights.h5")

# SourceFileList = list(open("wwhitman-clean-processed.txt", "r"))
SourceFileList =list(open("poem.txt", "r"))
source_tokenizer = helper.create_tokenizer(SourceFileList)
source_encoded_sequence = helper.encode_sequences(source_tokenizer,trainX.shape[1],SourceFileList)

translation = model.predict(source_encoded_sequence,batch_size=32)
for x in range(translation.shape[0]):
    # for y in range(int(translation.shape[2]/4)):
    # word = uncodedLabels[argmax(translation[x, :, y])]
    word = uncodedLabels[translation[x]]
    print(word)

print(translation.argmax(axis=1))