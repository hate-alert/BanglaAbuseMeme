#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

#Change the dataset path, based on your dataset path
allData = pd.read_csv("emnlp_bengalimeme.csv")


# In[2]:


import re
import emoji

puncts=[">","+",":",";","*","’","_","●","■","•","-",".","''","``","'","|","​","!",",","@","?","\u200d","#","(",")","|","%","।","=","``","&","[","]","/","'","”","‘","‘", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def valid_bengali_letters(char):
    if char in puncts:
        return False
    else:
        return True
    #return ord(char) >= 2433 and ord(char) <= 2543 

def get_replacement(char):
    if valid_bengali_letters(char):
        return char
    newlines = [10, 2404, 2405, 2551, 9576]
    if ord(char) in newlines: 
        return ' '
    return ' ';

def get_valid_lines(line):
    copy_line = ''
    for letter in line:
        copy_line += get_replacement(letter)
    return copy_line



def Diff(a,b):
    return list(set(a) -set(b))

def re_sub(pattern, repl,text):
    return re.sub(pattern, repl, text)


def preprocess_sent(sent):
    sent = re.sub(r"http\S+", " ", get_valid_lines(sent.lower()))
    sent = re.sub(r"@\S+", "@user", sent)

    #print(sent)
    sent = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "",sent)
    sent = emoji.demojize(sent)
    sent = re_sub(r"[:\*]", " ",sent)
    sent = re.sub(r"[<\*>]", " ",sent)
    sent = sent.replace("&amp;", " ")
    sent = sent.replace("ðŸ¤§", " ")
    sent = sent.replace("\n", " ")
    sent = sent.replace("ðŸ˜¡", " ")
    return sent


# In[3]:


allData.tail()


# In[6]:


from transformers import *
from normalizer import normalize # pip install git+https://github.com/csebuetnlp/normalizer
import torch


# ### Extract meme text features from BanglaBERT

# In[7]:


model = ElectraModel.from_pretrained("csebuetnlp/banglabert")
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")


# In[8]:


banglaBERTEmbedding ={}

from tqdm import tqdm
for index, row in tqdm(allData.iterrows(), total = allData.shape[0]):
    new_sentence = preprocess_sent(row['caption'])
    new_sentence = normalize(new_sentence) # this normalization step is required before tokenizing the text
    new_tokens = tokenizer.tokenize(new_sentence)
    new_inputs = tokenizer.encode(new_sentence, return_tensors="pt")
    with torch.no_grad():
        discriminator_outputs = model(new_inputs)[0]
        banglaBERTEmbedding[row['Ids']] = discriminator_outputs[0][0].numpy()
        del(discriminator_outputs)


# In[9]:


import pickle
with open("AllFeatures/banglaBERTEmbedding.p", "wb") as fp:
    pickle.dump(banglaBERTEmbedding, fp)


# ### Extract meme text features from m-BERT

# In[10]:


from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained("bert-base-multilingual-cased")
model.cuda()


# In[14]:


mBERTEmbedding ={}

from tqdm import tqdm
for index, row in tqdm(allData.iterrows(), total = allData.shape[0]):
    new_sentence = preprocess_sent(row['caption'])
    encoded_input = tokenizer(new_sentence, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input.to('cuda')).last_hidden_state
        mBERTEmbedding[row['Ids']] = output[0][0].cpu().numpy()
        del(output)


# In[15]:


import pickle
with open("AllFeatures/mEmbedding_bn_memes.p", "wb") as fp:
    pickle.dump(mBERTEmbedding, fp)


# ### Extract meme text features from MuRIL

# In[2]:


from transformers import *

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
model = BertModel.from_pretrained("google/muril-base-cased")

model.cuda()


# In[ ]:


murilBERTEmbedding ={}

from tqdm import tqdm
import torch
for index, row in tqdm(allData.iterrows(), total = allData.shape[0]):
    new_sentence = preprocess_sent(row['caption'])
    encoded_input = tokenizer(new_sentence, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input.to('cuda')).last_hidden_state
        murilBERTEmbedding[row['Ids']] = output[0][0].cpu().numpy()
        del(output)


# In[22]:


import pickle
with open("AllFeatures/MuRILEmbedding_bn_memes.p", "wb") as fp:
    pickle.dump(murilBERTEmbedding, fp)


# ### Extract meme text features from XLM-Roberta

# In[23]:


from transformers import *
import  torch

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaModel.from_pretrained("xlm-roberta-base")

model.cuda()


# In[26]:


xlmBERTEmbedding ={}

from tqdm import tqdm
for index, row in tqdm(allData.iterrows(), total = allData.shape[0]):
    new_sentence = preprocess_sent(row['caption'])
    encoded_input = tokenizer(new_sentence, return_tensors='pt')
    with torch.no_grad():    
        output = model(**encoded_input.to('cuda')).last_hidden_state
        xlmBERTEmbedding[row['Ids']] = output[0][0].cpu().numpy()
        del(output)


# In[27]:


import pickle
with open("AllFeatures/xlmBERTEmbedding_bn_memes.p", "wb") as fp:
    pickle.dump(xlmBERTEmbedding, fp)

