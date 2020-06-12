#!/usr/bin/env python
# coding: utf-8

# Tokenize and Train 

# Author        : Stephen Lee

# Goal          : Classify news source based on the article text
#                   Using data from
#                   - Fox News 
#                   - Vox News 
#                   - PBS News 

# Pre           : Cleaned csv file with articles 
# Post          : Csv files with params and F1 results
 
# Date          : 4.8.19
#               : 7.22.19 (converetd to ,py file from notebook)

################################################################################
# Imports
################################################################################

from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential
from keras.layers import GRU, Dense, Bidirectional, LSTM, Activation
from keras.utils import to_categorical

import os 
import math 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn import metrics

from tqdm import tqdm
import numpy as np
import csv 
from twilio.rest import Client

################################################################################
# Constants
################################################################################

FOLDER_READ = '/home/smlee_981/data'
FOLDER_WRITE = '/home/smlee_981/results'
FILE = 'clean_article_df.csv'
EMBEDS = 'glove.840B.300d.txt'
ARTICLE_LENGTH = 500                                 # max length for an article

SID = 'something'
TOKEN = 'something'
FROM_ = 'something'
TO_ = 'something'

################################################################################
# Helper Functions
################################################################################

def write_row(lst, file_out): 
    if os.getcwd() != FOLDER_WRITE: 
        os.chdir(FOLDER_WRITE)

    with open(file_out, 'a') as f: 
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(lst)

def target_to_one_hot(target, num_classes=3):
    return to_categorical(target, num_classes=num_classes)

def text_to_array(text, article_length=500):
    empty_emb = np.zeros(300)                   # each word is represented by a length 300 vector
    text = text[:-1].split()[:article_length]   # each article is length 500
    
    # look for word embedding, return zero array otherwise. 
    embeds = [embeddings_index.get(x, empty_emb) for x in text]
    embeds += [empty_emb] * (article_length - len(embeds))
    return np.array(embeds)

def batch_gen(train_df, batch_size=64, article_length=500, num_classes=3):
    n = math.ceil(len(train_df) / batch_size)
    while True: 
        train_df = train_df.sample(frac=1.0)
        
        for i in range(n):
            texts = train_df['clean_articles'][i*batch_size: (i+1)*batch_size]
            targets = train_df['targets'][i*batch_size: (i+1)*batch_size]
            
            targets = np.array([target_to_one_hot(t, num_classes) for t in targets])
            text_arr = np.array([text_to_array(text, article_length=article_length) for text in texts])
            yield text_arr, targets

def send_text(message): 
    client = Client(SID, TOKEN)
    client.messages.create(to=TO_, from_=FROM_, body=message)

################################################################################
# Define and Train Models
################################################################################

def train(df, file_out):

    '''
        INPUT... a dataframe with 'clean_articles' and 'targets'
        OUTPUT.. a file with the parameters and results of each run 
    '''

    # Set headers on output file

    headers = ['Model', 'Article Length', 'Batch Size', 'Dropout', 'Recurant Dropout', 'Steps Per Epoch', 'F1']
    write_row(headers, file_out)

    # Parameters 

    ARTICLE_LENGTH = [250, 500]
    BATCH_SIZE = [32, 64]
    DROPOUT = [0.1, 0.2]
    REC_DROPOUT = [0.1, 0.2]
    EPOCHS = 1 
    STEPS_PER_EPOCH = [500, 1000]

    # Split into test and training
    train_df, test_df = train_test_split(df, test_size=0.1)

    message = "Starting to train"
    send_text(message)

    try: 
            
        counter = 1
        for l in ARTICLE_LENGTH: 

            x_test = np.array([text_to_array(x, article_length=l) for x in tqdm(test_df["clean_articles"])])
            y_test = np.array([target_to_one_hot(t) for t in tqdm(test_df["targets"])])

            for bs in BATCH_SIZE: 
                for d in DROPOUT: 
                    for rd in REC_DROPOUT: 
                        for steps in STEPS_PER_EPOCH: 
                
                            # Model 1: Bidirectional LSTM
                            # notes...
                            #
                            #      batch_size         -> words per batch
                            #      article_length     -> words per article
                            #      embed_length       -> vector length per word

                            input_shape = (l, 300)
                            lstm_in = int(bs/2)

                            model_1 = Sequential()
                            model_1.add(Bidirectional(LSTM(lstm_in, return_sequences=False, dropout=d, recurrent_dropout=rd), input_shape=input_shape))
                            model_1.add(Activation('relu'))
                            model_1.add(Dense(3, activation="sigmoid"))
                            model_1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

                            # Model 2: Regular LSTM
                            # note...
                            #
                            #      batch_size         -> words per batch
                            #      article_length     -> words per article
                            #      embed_length       -> vector length per word

                            lstm_in = int(bs)

                            model_2 = Sequential()
                            model_2.add(LSTM(lstm_in, return_sequences=False, dropout=d, recurrent_dropout=rd, input_shape=input_shape))
                            model_2.add(Activation('relu'))
                            model_2.add(Dense(3, activation="sigmoid"))
                            model_2.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

                            # Train

                            message = "Fitting the models"
                            send_text(message)

                            data = batch_gen(train_df, batch_size=bs, article_length=l)
                            model_1.fit_generator(data, epochs=EPOCHS, steps_per_epoch=steps, validation_data=None, verbose=True)
                            model_2.fit_generator(data, epochs=EPOCHS, steps_per_epoch=steps, validation_data=None, verbose=True)

                            # Look at predictions

                            y_pred_1 = model_1.predict(x_test, batch_size=bs)
                            y_pred_class_1 = np.argmax(y_pred_1, axis=1)
                            y_pred_one_hot_1 = to_categorical(y_pred_class_1, num_classes=3)

                            y_pred_2 = model_2.predict(x_test, batch_size=bs)
                            y_pred_class_2 = np.argmax(y_pred_2, axis=1)
                            y_pred_one_hot_2 = to_categorical(y_pred_class_2, num_classes=3)

                            res_1 = metrics.f1_score(y_test, y_pred_one_hot_1, average='micro')
                            res_2 = metrics.f1_score(y_test, y_pred_one_hot_2, average='micro')

                            # headers = ['Model', 'Article Length', 'Batch Size', 'Dropout', 'Recurant Dropout', 'Steps Per Epoch', 'F1']
                            info_1 = ["Bidirectional", l, bs, d, rd, steps, res_1]
                            info_2 = ["Regular", l, bs, d, rd, steps, res_2]

                            write_row(info_1, file_out)
                            write_row(info_2, file_out)

                            message = "UPDATED: f1 scores of {one} and {two}".format(one=res_1, two=res_2)
                            send_text(message)
                            counter += 1
        
    except: 
        message = "ERROR"
        send_text(message)

################################################################################
                 
# RUN PROGRAM


################################################################################
# Read data file
################################################################################

os.chdir(FOLDER_READ)
df_all = pd.read_csv(FILE, sep='|').drop('Unnamed: 0', axis=1).drop('article', axis=1)

################################################################################
# Process
################################################################################

# change targets from 1 .. 3 to 0 .. 2 
df_all['targets'] = df_all['targets'].replace(3,0)

# Were going to make two datasets. Since the counts are inconsistent: 
# 1) resample to balance 
# 2) subsample to balance

# Resample dataset

fox = df_all[df_all['source'] == 'Fox']
vox = df_all[df_all['source'] == 'Vox']
pbs = df_all[df_all['source'] == 'PBS']

# start with the full dataset and append with 
# shorter targets
pbs_fox_diff = len(pbs) - len(fox)
pbs_vox_diff = len(pbs) - len(vox)

fox_append = fox.sample(pbs_fox_diff, replace=True)
vox_append = vox.sample(pbs_vox_diff)

df_resample = df_all.copy()
df_resample = df_resample.append(fox_append, ignore_index=True)
df_resample = df_resample.append(vox_append, ignore_index=True)
print("First dataset complete: ")
print(df_resample.groupby('source').count())


# Subsample dataset

df_all.groupby('source').count()

# start with the smallest dataset and 
# sample from larger sets
len_fox = len(fox)

df_subsample = fox.copy()
df_subsample = df_subsample.append(pbs.sample(len_fox), ignore_index=True)
df_subsample = df_subsample.append(vox.sample(len_fox), ignore_index=True)
print("Second dataset complete: ")
print(df_subsample.groupby('source').count())

message = "Constructed both datasets"
send_text(message)

################################################################################
# Get Embeddings
################################################################################

embeddings_index = {}
 
with open(EMBEDS, encoding='utf8') as embed:
    for line in tqdm(embed):
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        
print("Found {n} word vectors".format(n=len(embeddings_index)))
message = "Read in the embeddings..."
send_text(message)

################################################################################
# Main
################################################################################               

train(df_resample, 'results_all_resample.csv')
train(df_subsample, 'results_all_subsample.csv')

send_text('Complete!')