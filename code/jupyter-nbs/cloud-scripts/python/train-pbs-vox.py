#!/usr/bin/env python
# coding: utf-8

# # PBS and Vox
# 
# ### Author 
# Stephen Lee
# 
# ### Goal
# Classify news source based on the article text. Training data: 
# - Fox News
# - PBS News
# 
# ### Date 
# First  : 4.8.19
# 
# Update : 6.24.19
# 

# ## Read Data

# In[1]:


from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential
from keras.layers import GRU, Dense, Bidirectional, LSTM, Activation

import os 
import math 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[2]:


FOLDER_READ = '/home/smlee_981/data'
FILE = 'clean_article_df.csv'


# In[3]:


os.getcwd()


# In[4]:


os.chdir(FOLDER_READ)


# In[5]:


os.listdir()


# In[6]:


df_all = pd.read_csv(FILE, sep='|').drop('Unnamed: 0', axis=1)
df_all.head()


# # MAKE DATASETS
# 
# ## First Dataset

# #### Remove Fox

# In[7]:


df_all = df_all[df_all['source'] != "Fox"].drop('article', axis=1)
df_all.groupby('source').count()


# In[8]:


pbs = df_all[df_all['source'] == 'PBS']
vox = df_all[df_all['source'] == 'Vox']


# #### Duplicate Vox to balance

# In[9]:


# balance df by resampling from vox 
diff = len(pbs) - len(vox)

df_first = df_all.append(vox.sample(diff), ignore_index=True)
df_first.groupby('source').count()


# #### Relabel the targets

# In[10]:


from tqdm import tqdm
import numpy as np


# In[11]:


def label_fox(source):
    if source == "PBS":
        return 1 
    elif source == "Vox": 
        return 0
    else: 
        print(source)
        return None

targets = np.array([label_fox(t) for t in tqdm(df_first["source"])])
df_first['targets'] = targets


# In[12]:


# make sure that the targets are correct
df_first.groupby('source').describe()


# ## Second Dataset
# #### Sample from Vox to balance

# In[13]:


vox.count()


# In[14]:


pbs.count()


# In[15]:


# make second dataset for training
# start with vox and append with a sample of PBS
df_second = vox.copy()

len_vox = len(vox)
df_second = df_second.append(pbs.sample(len_vox), ignore_index=True)
df_second.groupby('source').count()


# #### Relabel the targets

# In[16]:


targets = np.array([label_fox(t) for t in tqdm(df_second["source"])])
df_second['targets'] = targets
df_second.head()


# In[17]:


# make sure that the targets are correct
df_second.groupby('source').describe()


# ## Get Embeddings and Define Helper Functions

# In[18]:


# glove embeddings and data are in same folder

EMBEDS = 'glove.840B.300d.txt'

embeddings_index = {}
 
with open(EMBEDS, encoding='utf8') as embed:
    for line in tqdm(embed):
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        
print("Found {n} word vectors".format(n=len(embeddings_index)))


# In[19]:


def text_to_array(text, article_length=500):
    empty_emb = np.zeros(300)                   # each word is represented by a length 300 vector
    text = text[:-1].split()[:article_length]   # each article is length 500
    
    # look for word embedding, return zero array otherwise. 
    embeds = [embeddings_index.get(x, empty_emb) for x in text]
    embeds += [empty_emb] * (article_length - len(embeds))
    return np.array(embeds)


# In[20]:


def batch_gen(train_df, batch_size=64, article_length=500):
    n = math.ceil(len(train_df) / batch_size)
    while True: 
        train_df = train_df.sample(frac=1.0)
        
        for i in range(n):
            texts = train_df['clean_articles'][i*batch_size: (i+1)*batch_size]
            targets = np.array(train_df['targets'][i*batch_size: (i+1)*batch_size])
            text_arr = np.array([text_to_array(text, article_length=article_length) for text in texts])
            yield text_arr, targets


# # Train Dataset 1

# ### Define Models
# 
# #### Model 1: Bidirectional LSTM

# In[21]:


# parameters
ARTICLE_LENGTH = 500
BATCH_SIZE = 64
DROPOUT = 0.2
REC_DROPOUT = 0.1


# In[22]:


# SINGLE LAYER BIDIRECTIONAL LTSM
# 
# note...
#
#      batch_size         -> words per batch
#      article_length     -> words per article
#      embed_length       -> vector length per word

input_shape = (ARTICLE_LENGTH, 300)
lstm_in = int(BATCH_SIZE/2)

model_1 = Sequential()
model_1.add(Bidirectional(LSTM(lstm_in, return_sequences=False,                         dropout=DROPOUT, recurrent_dropout=REC_DROPOUT),                         input_shape=input_shape))

model_1.add(Activation('relu'))
#model.add(Bidirectional(LSTM(lstm_in)))

model_1.add(Dense(1, activation="sigmoid"))
model_1.compile(loss="binary_crossentropy",               optimizer="adam",               metrics=["accuracy"])

model_1.summary()


# #### Model 2: Regular LSTM

# In[23]:


# SINGLE LAYER BIDIRECTIONAL LTSM
# 
# note...
#
#      batch_size         -> words per batch
#      article_length     -> words per article
#      embed_length       -> vector length per word

input_shape = (ARTICLE_LENGTH, 300)
lstm_in = int(BATCH_SIZE)

model_2 = Sequential()
model_2.add(LSTM(lstm_in, return_sequences=False, dropout=DROPOUT,                  recurrent_dropout=REC_DROPOUT, input_shape=input_shape))

model_2.add(Activation('relu'))

model_2.add(Dense(1, activation="sigmoid"))
model_2.compile(loss="binary_crossentropy",               optimizer="adam",               metrics=["accuracy"])

model_2.summary()


# #### Split into test and training

# In[24]:


train_df, test_df = train_test_split(df_first, test_size=0.1)


# #### Check for similarity between test and training

# In[25]:


test_df.groupby('source').describe()


# In[26]:


train_df.groupby('source').describe()


# #### Prepare test set for validation

# In[27]:


# witheld for validation

x_test = np.array([text_to_array(x, article_length=ARTICLE_LENGTH)                           for x in tqdm(test_df["clean_articles"])])
y_test = np.array(test_df["targets"])


# #### Train

# In[28]:


data = batch_gen(train_df, batch_size=BATCH_SIZE, article_length=ARTICLE_LENGTH)
model_1.fit_generator(data, epochs=2, steps_per_epoch=250,                     validation_data=None, verbose=True)


# In[29]:


data = batch_gen(train_df, batch_size=BATCH_SIZE, article_length=ARTICLE_LENGTH)
model_2.fit_generator(data, epochs=2, steps_per_epoch=250,                     validation_data=None, verbose=True)


# #### Look at predictions

# In[30]:


y_pred_1 = model_1.predict(x_test)
y_pred_1[:7]


# In[31]:


y_pred_2 = model_2.predict(x_test)
y_pred_2[:7]


# In[32]:


test_df[['source', 'clean_articles', 'targets']].head(7)


# In[33]:


for i in np.arange(0.25, 0.8, 0.05):
    res_1 = metrics.f1_score(y_test, y_pred_1 > i)
    res_2 = metrics.f1_score(y_test, y_pred_2 > i)
    print("Threshold {i} \nF1 score model 1: {res_1} \nF1 score model 2: {res_2} \n".format(                                                                                       i=round(i,2),                                                                                        res_1 = round(res_1, 3),                                                                                        res_2 = round(res_2, 3)))


# # Train Dataset 2

# ### Refresh Models
# 
# #### Model 1: Bidirectional LSTM

# In[34]:


# SINGLE LAYER BIDIRECTIONAL LTSM
# 
# note...
#
#      batch_size         -> words per batch
#      article_length     -> words per article
#      embed_length       -> vector length per word

input_shape = (ARTICLE_LENGTH, 300)
lstm_in = int(BATCH_SIZE/2)

model_1 = Sequential()
model_1.add(Bidirectional(LSTM(lstm_in, return_sequences=False,                         dropout=DROPOUT, recurrent_dropout=REC_DROPOUT),                         input_shape=input_shape))

model_1.add(Activation('relu'))
#model.add(Bidirectional(LSTM(lstm_in)))

model_1.add(Dense(1, activation="sigmoid"))
model_1.compile(loss="binary_crossentropy",               optimizer="adam",               metrics=["accuracy"])

model_1.summary()


# #### Model 2: Regular LSTM

# In[37]:


# SINGLE LAYER BIDIRECTIONAL LTSM
# 
# note...
#
#      batch_size         -> words per batch
#      article_length     -> words per article
#      embed_length       -> vector length per word

input_shape = (ARTICLE_LENGTH, 300)
lstm_in = int(BATCH_SIZE)

model_2 = Sequential()
model_2.add(LSTM(lstm_in, return_sequences=False,                         dropout=DROPOUT, recurrent_dropout=REC_DROPOUT,                         input_shape=input_shape))

model_2.add(Activation('relu'))

model_2.add(Dense(1, activation="sigmoid"))
model_2.compile(loss="binary_crossentropy",               optimizer="adam",               metrics=["accuracy"])

model_2.summary()


# #### Split into test and training

# In[42]:


train_df, test_df = train_test_split(df_second, test_size=0.1)
train_df.groupby('source').describe()


# #### Check for similarity between test and training

# In[43]:


test_df.groupby('source').describe()


# #### prepare test set for validation

# In[44]:


# witheld for validation
 
x_test = np.array([text_to_array(x, article_length=ARTICLE_LENGTH)                           for x in tqdm(test_df["clean_articles"])])
y_test = np.array(test_df["targets"])


# #### train

# In[45]:


data = batch_gen(train_df, batch_size=BATCH_SIZE, article_length=ARTICLE_LENGTH)
model_1.fit_generator(data, epochs=2, steps_per_epoch=250,                     validation_data=None, verbose=True)


# In[46]:


data = batch_gen(train_df, batch_size=BATCH_SIZE, article_length=ARTICLE_LENGTH)
model_2.fit_generator(data, epochs=2, steps_per_epoch=250,                     validation_data=None, verbose=True)


# #### Look at predictions

# In[47]:


y_pred_1 = model_1.predict(x_test)
y_pred_1[:7]


# In[48]:


y_pred_2 = model_2.predict(x_test)
y_pred_2[:7]


# In[49]:


for i in np.arange(0.25, 0.8, 0.05):
    res_1 = metrics.f1_score(y_test, y_pred_1 > i)
    res_2 = metrics.f1_score(y_test, y_pred_2 > i)
    print("Threshold {i} \nF1 score model 1: {res_1} \nF1 score model 2: {res_2} \n".format(                                                                                       i=round(i,2),                                                                                        res_1 = round(res_1, 3),                                                                                        res_2 = round(res_2, 3)))


# In[ ]:




