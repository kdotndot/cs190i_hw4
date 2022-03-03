import numpy as np
import torch
from torch.autograd import Variable



test_file = "test.txt"
train_file = "train.txt"
validation_file = "validation.txt"
words_file = "words.txt"
tags_file = "tags.txt"


f = open(train_file, "r")
w = open(words_file, "w")
t = open(tags_file, "w")


vocab = {}
tag_map = {}

string = ""


#Building words.txt and tags.txt
length = 0
ind = 0
for x in f:
    temp = x.split(" ")
    
    if len(temp) > 1:
        length += 1
        w.write(temp[0] + "\n")
        t.write(temp[1][:len(temp[1])])
    else:
        length = 0
        w.write("END" + "\n")
        t.write("O" + "\n")
    ind += 1
maxwords = 0
maxtags = 0

#Building vocab and tag_map dictionary
with open(words_file) as w:
    for i, l in enumerate(w.read().splitlines()):
        if int(i) > maxwords:
            maxwords = int(i)
        vocab[l] = i
with open(tags_file) as t:
    for i, l in enumerate(t.read().splitlines()):
        tag_map[l] = i
vocab["UNK"] = maxwords + 1     
vocab["PAD"] = -1


train_sentences = []        
train_labels = []


#End and UNK and label = "O"
f = open(train_file, "r")
arr1 = []
arr2 = []
for x in f:
    temp = x.split(" ")
    if len(temp) > 1:
        if temp[0] in vocab:
            arr1.append(vocab[temp[0]])
            arr2.append( tag_map[temp[1][:len(temp[1]) - 1]])
        else:
            arr1.append(vocab["UNK"])
            arr2.append(tag_map["O"])
    else:
        
        train_sentences.append(arr1)
        train_labels.append(arr2)
        arr1 = []
        arr2 = []

MAX_LEN = 0
for x in train_sentences:
    if len(x) > MAX_LEN:
        MAX_LEN = len(x)
NUM_SENTENCES = len(train_sentences)


batch_data = vocab['PAD']*np.ones((NUM_SENTENCES, MAX_LEN))
batch_labels = -1*np.ones((NUM_SENTENCES, MAX_LEN))

for j in range(len(train_sentences)):
    cur_len = len(train_sentences[j])
    batch_data[j][:cur_len] = train_sentences[j]
    batch_labels[j][:cur_len] = train_labels[j]

batch_data, batch_labels = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)

#convert Tensors to Variables
batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)

print(batch_data)
print(" ")
print(batch_labels)

""" 
data = list(vocab.items())
an_array = np.array(data)
print(an_array)

print(" ")
data1 = list(tag_map.items())
an_array1 = np.array(data1)
print(an_array1) """

    

    











