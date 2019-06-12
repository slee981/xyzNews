# Get Average Word Count per Article per Source

import pandas as pd 
import os 

dpath = '/home/stephen/Dropbox/CodeWorkspace/data-sets/Thesis/'
f = 'clean_article_df.csv'

df = pd.read_csv(dpath+f, sep="|").drop("Unnamed: 0", axis=1)

fox = df[df['source'] == "Fox"]
vox = df[df['source'] == "Vox"]
pbs = df[df['source'] == "PBS"]

fox_words = fox['clean_articles'].str.split().str.len()
vox_words = vox['clean_articles'].str.split().str.len()
pbs_words = pbs['clean_articles'].str.split().str.len()

# get average words per article per source
fox_words.mean()
vox_words.mean()
pbs_words.mean()

# get 500 word sample of each 
fox.iloc[0]['clean_articles'][:500]
vox.iloc[0]['clean_articles'][:500]
pbs.iloc[0]['clean_articles'][:500]
