from cmath import inf
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

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





test_file = "test.txt"
train_file = "train.txt"
validation_file = "validation.txt"
words_file = "words.txt"
tags_file = "tags.txt"

#Vocab_size = 23715, including pad and unk
#Embedding_dim = 300
#LSTM_hidden_dim = number of tags = 256
#Number_of_Tags = len(tags) = 9


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
        x = x.upper()
        if x not in vocab.keys():
            vocab[x] = count
            count += 1
count = 0
with open(tags_file) as t:
    for x in t:
        x = x[:len(x) - 1]
        if x not in tag_map.keys():
            tag_map[x] = count
            count += 1
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


params = Parameters(vocab_size = 23715, embedding_dim = 300, lstm_hidden_dim = 256, number_of_tags= 9)

Neural_Net = Net(params)

for x in range(0,8322, 57):
    train_sentences = batch_data[x-57:x]
    train_labels = batch_labels[x-57:x]
    soft_max = Neural_Net.forward(train_sentences)
    loss = loss_fn(outputs = soft_max, labels= train_labels)
    loss.backward()
    print(x)




#Trained with only testing

#Now prediction

test_sentences = []
with open(test_file) as t:
    temp_arr = []
    for x in t:
        if x != "\n":
            x = x[:len(x) - 1]
            x = x.upper()
            
            if x in vocab.keys():
                temp_arr.append(vocab[x])
            else:
                
                temp_arr.append(vocab["UNK"])
        else:
            
            test_sentences.append(temp_arr)
            temp_arr = []


batch_testing = vocab['PAD']*np.ones((len(test_sentences), MAX_LEN))
for j in range(len(test_sentences)):
    cur_len = len(test_sentences[j])
    batch_testing[j][:cur_len] = test_sentences[j]
    
batch_testing = torch.LongTensor(batch_testing)
batch_testing = Variable(batch_testing)

print(len(batch_testing)) #Num of sentences in test = 1516


for x in range(0,1516,4):
    test_sentences = batch_testing[:x]
    soft_max = Neural_Net.forward(train_sentences)
    for x in range(len(soft_max)):
        max = -inf
        ind = 0
        arr = np.array([])
        for y in range(9):
            if soft_max[x][y] > max:
                max = soft_max[x][y]
                ind = y
        pred_values = np.append(pred_values, ind) 

print(vocab)
print(pred_values)



