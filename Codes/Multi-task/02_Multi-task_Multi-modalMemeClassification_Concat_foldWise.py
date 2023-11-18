#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import json
import random
import time
import datetime
import random
import re
import numpy as np
import emoji
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from sklearn.metrics import *


# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name())

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# In[2]:


def fix_the_random(seed_val = 42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


# In[3]:


fix_the_random(2021)


# In[4]:


def evalMetric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    mf1Score = f1_score(y_true, y_pred, average='macro')
    recallScore = recall_score(y_true, y_pred, labels = np.unique(y_pred), average='macro')
    precisionScore = precision_score(y_true, y_pred, labels = np.unique(y_pred), average='macro')
    return dict({"accuracy": accuracy, 'mF1Score': mf1Score,
           'precision': precisionScore, 'recall': recallScore})



def getFeaturesandLabel(X, abusive_labels, sentiment_labels, sarcasm_labels, vulgar_labels, text_features, image_features):
    X_text_data = []
    X_image_data = []
    for i in X:
        try:
            X_text_data.append(text_features[i]['text'])
        except:
            X_text_data.append(text_features[i])
        try:
            X_image_data.append(image_features[i]['image'])
        except:
            X_image_data.append(image_features[i])
    X_text_data = torch.tensor(X_text_data).float()
    X_image_data = torch.tensor(X_image_data).float()
    y_abusive   = torch.tensor(abusive_labels)
    y_sentiment = torch.tensor(sentiment_labels)
    y_sarcasm = torch.tensor(sarcasm_labels)
    y_vulgar = torch.tensor(vulgar_labels)
    return X_text_data, X_image_data, y_abusive, y_sentiment, y_sarcasm, y_vulgar


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# In[5]:


import torch.nn as nn
import torch.nn.functional as F

class Uni_Model(nn.Module):
    def __init__(self, input_size, fc1_hidden, fc2_hidden):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(input_size,fc1_hidden),
            nn.ReLU(),
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
        )
        self.abusive   = nn.Linear(fc2_hidden,2)
        self.sentiment = nn.Linear(fc2_hidden,3)
        self.sarcasm   = nn.Linear(fc2_hidden,2)
        self.vulgar    = nn.Linear(fc2_hidden,2)

    def forward(self, x_text, x_vid):
        inp = self.network(torch.cat((x_text, x_vid), dim = 1))
        abusive_out = self.abusive(inp)
        sentiment_out = self.sentiment(inp)
        sarcasm_out = self.sarcasm(inp)
        vulgar_out = self.vulgar(inp)
        return abusive_out, sentiment_out, sarcasm_out, vulgar_out


# In[6]:


def computeClassWeight(train_classes):
    class_weights = compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(train_classes),
                                        y = train_classes                                                    
                                    )
#     print(class_weights)
#     class_weights = dict(zip(np.unique(train_classes), class_weights))
#     print(class_weights)
    return class_weights


# In[7]:


import torch
import torch.nn as nn


# In[8]:


import numpy as np
def getProb(temp):
    t = np.exp(temp)
    return t[1]/(sum(t))


# In[9]:


import pandas as pd
def getPerformanceOfLoader(model,test_dataloader, loadType):
    model.eval()
    # Tracking variables 
    true_abusive, true_sentiment, true_sarcasm, true_vulgar = [], [], [], []
    pred_abusive, pred_sentiment, pred_sarcasm, pred_vulgar = [], [], [], []
    # Predict 
    for batch in test_dataloader:
        #print(batch)
    # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
  
        # Unpack the inputs from our dataloader
        b_text_ids, b_image_ids, y_abusive, y_sentiment, y_sarcasm, y_vulgar = batch
  
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_text_ids, b_image_ids)
        
        logits = [torch.max(outputs[i], 1).indices for i in range(4)]
        # Move logits and labels to CPU
        # Store predictions and true labels

        pred_abusive.extend(logits[0].detach().cpu().numpy())
        true_abusive.extend(y_abusive.to('cpu').numpy())

        pred_sentiment.extend(logits[1].detach().cpu().numpy())
        true_sentiment.extend(y_sentiment.to('cpu').numpy())

        pred_sarcasm.extend(logits[2].detach().cpu().numpy())
        true_sarcasm.extend(y_sarcasm.to('cpu').numpy())

        pred_vulgar.extend(logits[3].detach().cpu().numpy())
        true_vulgar.extend(y_vulgar.to('cpu').numpy())

    print('DONE.')

    #pred = [i[0] for i in predictions]
    df = pd.DataFrame()
    if loadType =='val':
        df['Ids']=val_list
    else:
        df['Ids']=test_list
    df['true_abusive'] = true_abusive
    df['pred_abusive'] = pred_abusive

    df['true_sentiment'] = true_sentiment
    df['pred_sentiment'] = pred_sentiment
    
    df['true_sarcasm'] = true_sarcasm
    df['pred_sarcasm'] = pred_sarcasm
    
    df['true_vulgar'] = true_vulgar
    df['pred_vulgar'] = pred_vulgar

    
    #df['score'] = proba
    return df


# In[10]:


# Tell pytorch to run this model on the GPU.
def trainModel(model, train_dataloader, validation_dataloader, test_dataloader):
    model.cuda()

    bestValAcc  = 0
    bestValMF1  = 0
    besttest_df  = None
    bestEpochs = -1
    # Get all of the model's parameters as a list of tuples.
    learning_rate = 1e-4
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)   # optimize all cnn parameters

    epochs = 30
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_text_ids = batch[0].to(device)
            b_image_ids = batch[1].to(device)
            b_abusive   = batch[2].to(device)
            b_sentiment = batch[3].to(device)
            b_sarcasm   = batch[4].to(device)
            b_vulgar    = batch[5].to(device)

            model.zero_grad()        

            outputs = model(b_text_ids, b_image_ids)
            # y_preds = torch.max(outputs, 1)[1]  # y_pred != output

            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            
            loss =    F.cross_entropy(outputs[0], b_abusive, weight=torch.FloatTensor(abusiveWeight).to(device)) \
                    + 0.5*F.cross_entropy(outputs[1], b_sentiment, weight=torch.FloatTensor(sentiWeight).to(device)) \
                    + 0.5*F.cross_entropy(outputs[2], b_sarcasm, weight=torch.FloatTensor(sarcasmWeight).to(device)) \
                    + 0.5*F.cross_entropy(outputs[3], b_vulgar, weight=torch.FloatTensor(vulgarWeight).to(device))
            

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()


        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader) 

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()



        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        val_df = getPerformanceOfLoader(model, validation_dataloader, "val")
        

        
        # Report the final accuracy for this validation run.
        valMf1ScoreAbs   = evalMetric(list(val_df['true_abusive']), list(val_df['pred_abusive']))['mF1Score']
        valMf1ScoreSenti = evalMetric(list(val_df['true_sentiment']), list(val_df['pred_sentiment']))['mF1Score']
        valMf1ScoreSar  = evalMetric(list(val_df['true_sarcasm']), list(val_df['pred_sarcasm']))['mF1Score']
        valMf1ScoreVul  = evalMetric(list(val_df['true_vulgar']), list(val_df['pred_vulgar']))['mF1Score']
        avgwMF1Score = (valMf1ScoreAbs + 0.5*valMf1ScoreSenti + 0.5*valMf1ScoreSar + 0.5*valMf1ScoreVul) / (1 + 3*0.5)
        
        if (avgwMF1Score > bestValMF1):
            bestEpochs = epoch_i
            bestValMF1 = avgwMF1Score
            #bestValAcc  = tempValAcc
            besttest_df = getPerformanceOfLoader(model,test_dataloader, "test")
            #torch.save(model, "./BERT_CODE_BERT/bert_code_bert_256_128")
        #print("  Accuracy: {0:.2f}".format(tempValAcc))
        print("Abusive Macro F1: {0:.2f}, Sentiment Macro F1: {0:.2f}, Sarcasm Macro F1: {0:.2f}, \
             Vulgar Macro F1: {0:.2f}".format(valMf1ScoreAbs, valMf1ScoreSenti, valMf1ScoreSar, valMf1ScoreVul))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print(bestEpochs)
    print("Training complete!")
    return besttest_df


# In[11]:


import pickle
with open('./FoldWiseDetailBengaliMultiTask_emnlp.p', 'rb') as fp:
    allDataAnnotation = pickle.load(fp)


# In[1]:


FOLDER_NAME="./"


# In[2]:


modelNameMapping = {
    "mBERTEmbedding": FOLDER_NAME+'AllFeatures/mEmbedding_bn_memes.p',
    "MuRILEmbedding" : FOLDER_NAME+'AllFeatures/MuRILEmbedding_bn_memes.p',
    "banglaBERTEmb" : FOLDER_NAME+'AllFeatures/banglaBERTEmbedding.p', 
    "XLMREmb" : FOLDER_NAME+'AllFeatures/xlmBERTEmbedding_bn_memes.p',
    
    "vgg16" : FOLDER_NAME+'AllFeatures/vgg16.p',
    "resNet152_new" : FOLDER_NAME+'AllFeatures/resNet152_newFeatures_224.p',
    "vit_new_wOReize" : FOLDER_NAME+'AllFeatures/vit_newFeatures_wOResize.p',
    'CLIP': FOLDER_NAME+'AllFeatures/CLIP_newFeatures.p',
}


# In[14]:


metricType = ['accuracy', 'mF1Score', 'precision', 'recall']


# In[19]:


# training parameters
k = 2            # number of target category
epochs = 30
batch_size = 32
learning_rate = 1e-4
log_interval = 1
import numpy as np

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

allF = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']

import pickle

outputFp = open("MultiTaskResultFolder/MultiTask_MMFoldWiseConcatLessParam.txt", 'a')

image_models =["resNet152_new", "vit_new_wOReize"]
text_models =["XLMREmb", "MuRILEmbedding"]

# image_models =["CLIP"]
# text_models =["CLIP"]

for text_mod in text_models:
    for image_mod in image_models:
        modelName = text_mod+"_"+image_mod+"_concat_"

        with open(modelNameMapping[text_mod],'rb') as fp:
            inputTextFeatures = pickle.load(fp)

        with open(modelNameMapping[image_mod],'rb') as fp:
            inputImageFeatures = pickle.load(fp)

        # Audio parameters
        if text_mod == "CLIP":
            input_size_text, input_size_image = 512, 512
        else:
            input_size_text = len(inputTextFeatures['image_759.jpg'])
            input_size_image = len(inputImageFeatures['image_759.jpg'])

        fc1_hidden_audio, fc2_hidden_audio = 256, 256

        finalOutputAccrossFold ={}

        for fold in allF:
            # train, test split
            train_list, train_abusive, train_sentiment, train_sarcasm, train_vulgar = allDataAnnotation[fold]['train']
            val_list, val_abusive, val_sentiment, val_sarcasm, val_vulgar = allDataAnnotation[fold]['val']
            test_list, test_abusive, test_sentiment, test_sarcasm, test_vulgar = allDataAnnotation[fold]['test']

            abusiveWeight = computeClassWeight(train_abusive)
            sentiWeight = computeClassWeight(train_sentiment)
            sarcasmWeight = computeClassWeight(train_sarcasm)
            vulgarWeight = computeClassWeight(train_vulgar)
            
            X_train_text, X_train_image, y_train_abs, y_train_senti, y_train_sar, y_train_vul = getFeaturesandLabel(train_list, train_abusive, train_sentiment, train_sarcasm, train_vulgar, inputTextFeatures, inputImageFeatures)
            X_val_text, X_val_image, y_val_abs, y_val_senti, y_val_sar, y_val_vul = getFeaturesandLabel(val_list, val_abusive, val_sentiment, val_sarcasm, val_vulgar, inputTextFeatures, inputImageFeatures)
            X_test_text, X_test_image, y_test_abs, y_test_senti, y_test_sar, y_test_vul = getFeaturesandLabel(test_list, test_abusive, test_sentiment, test_sarcasm, test_vulgar, inputTextFeatures, inputImageFeatures)


            BATCH_SIZE = 32
            #Dataset wrapping tensors.
            train_data = TensorDataset(X_train_text, X_train_image, y_train_abs, y_train_senti, y_train_sar, y_train_vul)
            val_data = TensorDataset(X_val_text, X_val_image, y_val_abs, y_val_senti, y_val_sar, y_val_vul)
            test_data = TensorDataset(X_test_text, X_test_image, y_test_abs, y_test_senti, y_test_sar, y_test_vul)

            #Samples elements randomly. If without replacement(default), then sample from a shuffled dataset.
            train_sampler = RandomSampler(train_data)
            val_sampler = SequentialSampler(val_data)
            test_sampler = SequentialSampler(test_data)

            #represents a Python iterable over a dataset
            train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = BATCH_SIZE)
            validation_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size = BATCH_SIZE)
            test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size = BATCH_SIZE)


            comb = Uni_Model(input_size_text+input_size_image, fc1_hidden_audio, fc2_hidden_audio).to(device)

            besttest_df = trainModel(comb, train_dataloader, validation_dataloader, test_dataloader)


            finalOutputAccrossFold[fold] = besttest_df

        with open("MultiTaskResultFolder/MultiTask_"+modelName+"_foldWise_res_concat_LessParam.p", 'wb') as fp:
            pickle.dump(finalOutputAccrossFold,fp)

        allValueDict ={}
        for fold in allF:
            evalObject = evalMetric(finalOutputAccrossFold[fold]['true_abusive'], finalOutputAccrossFold[fold]['pred_abusive'])
            for metType in metricType:
                try:
                    allValueDict[metType].append(evalObject[metType])
                except:
                    allValueDict[metType]=[evalObject[metType]]

        print(modelName+"--> Abusive Score")
        for i in allValueDict:
            print(f"{i} : Mean {np.mean(allValueDict[i])}  STD: {np.std(allValueDict[i])}")

        outputFp.write(modelName+"--> Abusive Score\n")
        for i in allValueDict:
            outputFp.write(f"{i} : Mean {np.mean(allValueDict[i])}  STD: {np.std(allValueDict[i])}")
            outputFp.write("\n")
        outputFp.write("====================\n")


        allValueDict ={}
        for fold in allF:
            evalObject = evalMetric(finalOutputAccrossFold[fold]['true_sentiment'], finalOutputAccrossFold[fold]['pred_sentiment'])
            for metType in metricType:
                try:
                    allValueDict[metType].append(evalObject[metType])
                except:
                    allValueDict[metType]=[evalObject[metType]]

        print(modelName +"--> Sentiment Score")
        for i in allValueDict:
            print(f"{i} : Mean {np.mean(allValueDict[i])}  STD: {np.std(allValueDict[i])}")

        outputFp.write(modelName+"--> Sentiment Score\n")
        for i in allValueDict:
            outputFp.write(f"{i} : Mean {np.mean(allValueDict[i])}  STD: {np.std(allValueDict[i])}")
            outputFp.write("\n")
        outputFp.write("====================\n")


        allValueDict ={}
        for fold in allF:
            evalObject = evalMetric(finalOutputAccrossFold[fold]['true_sarcasm'], finalOutputAccrossFold[fold]['pred_sarcasm'])
            for metType in metricType:
                try:
                    allValueDict[metType].append(evalObject[metType])
                except:
                    allValueDict[metType]=[evalObject[metType]]

        print(modelName +"--> Sarcasm Score")
        for i in allValueDict:
            print(f"{i} : Mean {np.mean(allValueDict[i])}  STD: {np.std(allValueDict[i])}")

        outputFp.write(modelName+"--> Sarcasm Score\n")
        for i in allValueDict:
            outputFp.write(f"{i} : Mean {np.mean(allValueDict[i])}  STD: {np.std(allValueDict[i])}")
            outputFp.write("\n")
        outputFp.write("====================\n")


        allValueDict ={}
        for fold in allF:
            evalObject = evalMetric(finalOutputAccrossFold[fold]['true_vulgar'], finalOutputAccrossFold[fold]['pred_vulgar'])
            for metType in metricType:
                try:
                    allValueDict[metType].append(evalObject[metType])
                except:
                    allValueDict[metType]=[evalObject[metType]]

        print(modelName +"--> Vulgar Score")
        for i in allValueDict:
            print(f"{i} : Mean {np.mean(allValueDict[i])}  STD: {np.std(allValueDict[i])}")

        outputFp.write(modelName+"--> Vulgar Score\n")
        for i in allValueDict:
            outputFp.write(f"{i} : Mean {np.mean(allValueDict[i])}  STD: {np.std(allValueDict[i])}")
            outputFp.write("\n")
        outputFp.write("====================\n")

        print("*******************************")

outputFp.close()

