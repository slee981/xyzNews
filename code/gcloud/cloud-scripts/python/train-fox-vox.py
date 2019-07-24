#!/usr/bin/env python
# coding: utf-8

# Tokenize and Train 

# Author        : Stephen Lee

# Goal          : Classify news source based on the article text
#                   Using data from
#                   - Fox News 
#                   - Vox News 

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

SID = 'AC452ff5bfa2b523d0380e4938761d59ba'
TOKEN = 'bfb56b152a943a98d7e216d2a709c7c4'
FROM_ = '+16292095399'
TO_ = '+16159444486'

################################################################################
# Helper Functions
################################################################################

def write_row(lst, file_out): 
    if os.getcwd() != FOLDER_WRITE: 
        os.chdir(FOLDER_WRITE)

    with open(file_out, 'a') as f: 
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(lst)

def relabel(source, target):
    if source == target:
        return 1 
    else: 
        return 0

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
    DROPOUT = [0.1]
    REC_DROPOUT = [0.1]
    EPOCHS = 1 
    STEPS_PER_EPOCH = [500, 1000]

    # Split into test and training
    train_df, test_df = train_test_split(df, test_size=0.1)

    send_text("Starting to train")

    counter = 1
    for l in ARTICLE_LENGTH: 

        x_test = np.array([text_to_array(x, article_length=l) for x in tqdm(test_df["clean_articles"])])
        y_test = np.array(test_df["targets"])

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
                        model_1.add(Dense(1, activation="sigmoid"))
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
                        model_2.add(Dense(1, activation="sigmoid"))
                        model_2.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

                        # Train

                        message = "Fitting the models"
                        send_text(message)

                        data = batch_gen(train_df, batch_size=bs, article_length=l)
                        model_1.fit_generator(data, epochs=EPOCHS, steps_per_epoch=steps, validation_data=None, verbose=True)
                        model_2.fit_generator(data, epochs=EPOCHS, steps_per_epoch=steps, validation_data=None, verbose=True)

                        # Look at predictions

                        y_pred_1 = model_1.predict(x_test, batch_size=bs)
                        y_pred_2 = model_2.predict(x_test, batch_size=bs)

                        threshold = 0.5
                        res_1 = metrics.f1_score(y_test, y_pred_1, y_pred_1 > threshold)
                        res_2 = metrics.f1_score(y_test, y_pred_2, y_pred_2 > threshold)

                        # headers = ['Model', 'Article Length', 'Batch Size', 'Dropout', 'Recurant Dropout', 'Steps Per Epoch', 'F1']
                        info_1 = ["Bidirectional", l, bs, d, rd, steps, res_1]
                        info_2 = ["Regular", l, bs, d, rd, steps, res_2]

                        write_row(info_1, file_out)
                        write_row(info_2, file_out)

                        message = "\nUPDATED: f1 scores of {one} and {two}".format(one=round(res_1,2), two=round(res_2,2))
                        send_text(message)
                        counter += 1

################################################################################
                 
# RUN PROGRAM


################################################################################
# Read data file
################################################################################

os.chdir(FOLDER_READ)
df_all = pd.read_csv(FILE, sep='|').drop('Unnamed: 0', axis=1).drop('article', axis=1)
df_all = df_all[df_all['source'] != "PBS"]

################################################################################
# Process
################################################################################

# Were going to make two datasets. Since the counts are inconsistent: 
# 1) resample to balance 
# 2) subsample to balance

fox = df_all[df_all['source'] == 'Fox']
vox = df_all[df_all['source'] == 'Vox']

len_vox = len(vox)                          # <-- longer dataset
len_fox = len(fox)                          # <-- shorter dataset

# RESAMPLE DATASE

# start with the longer dataset and append with fill from shorter
df_resample = vox.copy()
df_resample = df_resample.append(fox.sample(len_vox, replace=True), ignore_index=True)

print("First dataset complete: ")
print(df_resample.groupby('source').count())


# SUBSAMPLE DATASET

# start with shorter data and append with subsample from longer
df_subsample = fox.copy()
df_subsample = df_subsample.append(vox.sample(len_fox), ignore_index=True)

print("Second dataset complete: ")
print(df_subsample.groupby('source').count())

send_text("Constructed both datasets")

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

train(df_resample, 'results_fox_vox_resample.csv')
train(df_subsample, 'results_fox_vox_subsample.csv')

send_text('Complete!')