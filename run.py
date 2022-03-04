from cmath import inf
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
count = 0
with open(words_file) as w:
    for x in w:
        x = x[:len(x) - 1]
        if x not in vocab.keys():
            vocab[x] = count
            count += 1
    """ for i, l in enumerate(w.read().splitlines()):
        vocab[l] = i """
count = 0
with open(tags_file) as t:
    for x in t:
        x = x[:len(x) - 1]
        if x not in tag_map.keys():
            tag_map[x] = count
            count += 1
    """ for i, l in enumerate(t.read().splitlines()):
        tag_map[l] = i """
vocab["UNK"] = len(vocab)    
vocab["PAD"] = len(vocab) 



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

MAX_LEN = 0 #1238
for x in train_sentences:
    if len(x) > MAX_LEN:
        MAX_LEN = len(x)
NUM_SENTENCES = len(train_sentences) #8322



batch_data = vocab['PAD']*np.ones((NUM_SENTENCES, MAX_LEN))
batch_labels = (-1)*np.ones((NUM_SENTENCES, MAX_LEN))

for j in range(len(train_sentences)):
    cur_len = len(train_sentences[j])
    batch_data[j][:cur_len] = train_sentences[j]
    batch_labels[j][:cur_len] = train_labels[j]

batch_data, batch_labels = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)

#convert Tensors to Variables
batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)

batch_data = batch_data[:4]
batch_labels = batch_labels[:4]



import torch.nn as nn
import torch.nn.functional as F
#Vocab_size = 26102
#Embedding_dim = 300
#LSTM_hidden_dim = number of tags = 256
#Number_of_Tags = len(tags) = 9



class Parameters():
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, number_of_tags):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.number_of_tags = number_of_tags



class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()

        #maps each token to an embedding_dim vector
        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)

        #the LSTM takens embedded sentence
        self.lstm = nn.LSTM(params.embedding_dim, params.lstm_hidden_dim, batch_first=True) # What is LSTM_hidden?

        #fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(params.lstm_hidden_dim, params.number_of_tags)
    
    def forward(self, s):
        #apply the embedding layer that maps each token to its embedding
        s = self.embedding(s)   # dim: batch_size x batch_max_len x embedding_dim

        #run the LSTM along the sentences of length batch_max_len
        s, _ = self.lstm(s)     # dim: batch_size x batch_max_len x lstm_hidden_dim                

        #reshape the Variable so that each row contains one token
        s = s.reshape(-1, s.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim

        #apply the fully connected layer and obtain the output for each token
        s = self.fc(s)          # dim: batch_size*batch_max_len x num_tags

        return F.log_softmax(s, dim=1)   # dim: batch_size*batch_max_len x num_tags
    
    def loss_fn(outputs, labels):
        #reshape labels to give a flat vector of length batch_size*seq_len
        
        labels = labels.reshape(-1)  
        
        #mask out 'PAD' tokens
        mask = (labels >= 0).float()
        
        #the number of tokens is the sum of elements in mask
        num_tokens = int(torch.sum(mask).item())

        #pick the values corresponding to labels and multiply by mask
        outputs = outputs[range(outputs.shape[0]), labels]*mask

        #cross entropy loss for all non 'PAD' tokens
        return -torch.sum(outputs)/num_tokens

params = Parameters(vocab_size = 26102, embedding_dim = 300, lstm_hidden_dim = 256, number_of_tags= 9)

Neural_Net = Net(params)


soft_max = Neural_Net.forward(batch_data)
pred_values = []

for x in range(len(soft_max)):
    max = -inf
    ind = 0
    for y in range(9):
        if soft_max[x][y] > max:
            max = soft_max[x][y]
            ind = y
    pred_values.append(list(tag_map)[ind])

print(batch_labels)








#print(Neural_Net.loss_fn(batch_labels))


