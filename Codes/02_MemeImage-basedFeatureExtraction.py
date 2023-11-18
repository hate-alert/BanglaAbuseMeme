#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import os
import numpy as np
import torch
import random
import functools
import operator
import collections
import torchvision.models as models
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm, trange
import json
import pickle
import numpy as np
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import random
import os
from tqdm import tqdm


# In[1]:


get_ipython().system('export CUDA_VISIBLE_DEVICES=1')


# In[4]:


rootFolder = './'
allInfo = rootFolder+'FinalAnnFirstTime/'

import pandas as pd
#Change the dataset path, based on your dataset path

allData = pd.read_csv("emnlp_bengalimeme.csv")


# ### Extract meme image features from ResNet-152

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained ResNet-152 model and move it to the GPU
resnet152 = models.resnet152(pretrained=True)
resnet152.to(device)

# Remove the fully connected layer
resnet152 = torch.nn.Sequential(*list(resnet152.children())[:-1])

# Set the model to evaluation mode
resnet152.eval()

# Preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to ResNet input size
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

trainValFeature ={}

from tqdm import tqdm
import numpy as np
for i in tqdm(list(allData['Ids'])):
    if i in trainValFeature:
        continue
    imgId = allInfo+"/"+i
    image = get_image(imgId)

    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Move the image to the GPU
    image = image.to(device)

    # Extract features
    with torch.no_grad():
        features = resnet152(image)

    # Move the features back to the CPU if necessary
    features = features.cpu()

    # Flatten the features
    features = torch.flatten(features)

    trainValFeature[i] = features.numpy()

import pickle
with open("AllFeatures/resNet152_newFeatures_224.p", 'wb') as fp:
    pickle.dump(trainValFeature,fp)

# ### Extract meme image features from Vision Transformer(VIT)

# In[26]:


trainValFeature ={}


# In[6]:


from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTModel
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)


# In[7]:


from PIL import Image
import requests


# In[10]:


def get_image(path):
        image = Image.open(path).convert('RGB')
        if len(np.array(image).shape) == 2:
            image = np.array(image)
            image = np.stack([image,image,image],axis=2)
            image = Image.fromarray(image)
        else:
            image = Image.open(imgId).convert('RGB')
        return image


# In[17]:


from tqdm import tqdm
for i in tqdm(list(allData['Ids'])):
    if i in trainValFeature:
        continue
    imgId = allInfo+"/"+i
    inputs = feature_extractor(get_image(imgId), return_tensors="pt")
    inputs =inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        trainValFeature[i] = last_hidden_states[0][0].cpu().numpy()


# In[19]:


with open("AllFeatures/vit_newFeatures_wOResize.p", 'wb') as fp:
    pickle.dump(trainValFeature,fp)


# ### Extract meme image features from VGG16

# In[21]:


class FeatureExtractor(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor, self).__init__()
		# Extract VGG-16 Feature Layers
    self.features = list(model.features)
    self.features = nn.Sequential(*self.features)
		# Extract VGG-16 Average Pooling Layer
    self.pooling = model.avgpool
		# Convert the image into one-dimensional vector
    self.flatten = nn.Flatten()
		# Extract the first part of fully-connected layer from VGG16
    self.fc = model.classifier[0]
  
  def forward(self, x):
		# It will take the input 'x' until it returns the feature vector called 'out'
    out = self.features(x)
    out = self.pooling(out)
    out = self.flatten(out)
    out = self.fc(out) 
    return out 


# In[22]:


model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model)

# Change the device to GPU
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
new_model = new_model.to(device)


# In[24]:


# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# In[25]:


vgg16Emb ={}
new_model.to('cuda')
from tqdm import tqdm
for index, row in tqdm(allData.iterrows(), total = allData.shape[0]):
    input_image = Image.open("FinalAnnFirstTime/"+row['Ids']).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
    with torch.no_grad():
        output = new_model(input_batch)
    vgg16Emb[row['Ids']] = output[0].cpu().detach().numpy()


# In[26]:


with open("AllFeatures/vgg16.p", 'wb') as fp:
    pickle.dump(vgg16Emb,fp)


# ### Extract meme text and image features from CLIP

# In[9]:


import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# In[10]:


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


# In[11]:


trainValFeature = {}


# In[ ]:


from tqdm import tqdm
for i, j in tqdm(zip(list(allData['Ids']), list(allData['caption']))):
    if i in trainValFeature:
        continue
    imgId = allInfo+"/"+i
    image = preprocess(get_image(imgId)).unsqueeze(0).to(device)
    text  = clip.tokenize(preprocess_sent(j), truncate = True).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        trainValFeature[i] ={'text': text_features[0].cpu().numpy(), 'image': image_features[0].cpu().numpy()}


# In[ ]:


with open("AllFeatures/CLIP_newFeatures.p", 'wb') as fp:
    pickle.dump(trainValFeature,fp)


# ### Extract meme image features from Visual-Attention-Network(VAN)

# In[43]:


trainValFeature ={}


# In[2]:


from transformers import AutoImageProcessor, VanModel

image_processor = AutoImageProcessor.from_pretrained("Visual-Attention-Network/van-base")
model = VanModel.from_pretrained("Visual-Attention-Network/van-base")
model.cuda()


# In[23]:


for i in tqdm(list(allData['Ids'])):
    if i in trainValFeature:
        continue
    imgId = allInfo+"/"+i
    with torch.no_grad():
        inputs = image_processor(get_image(imgId), return_tensors="pt")
        outputs = model(**inputs.to('cuda'))
    trainValFeature[i] =outputs.pooler_output[0].cpu().numpy()


# In[24]:


import pickle
with open("AllFeatures/van_newFeatures.p", 'wb') as fp:
    pickle.dump(trainValFeature,fp)

