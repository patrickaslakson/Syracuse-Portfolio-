# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 00:37:45 2020

@author: Patrick
"""



### API KEY 
##https://newsapi.org/
## Example
## https://newsapi.org/v1/articles?
##source=bbc-news&sortBy=top&apiKey= f9fd15b8062345a29826a0a77639b5d1 

# $ pip install newsapi-python

## import libraries
import nltk
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
#from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import glob #using to loop through sub directories
from sklearn.feature_extraction import text  #to add own stop words
import re
from sklearn.model_selection import train_test_split
import random as rd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
#from mpl_toolkits.mplot3d import Axes3D 
## conda install python-graphviz
## restart kernel (click the little red x next to the Console)
#import graphviz 
from sklearn.metrics import confusion_matrix

from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO  
from IPython.display import Image  
## conda install pydotplus
#import pydotplus
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import seaborn as sns; sns.set()
#import tensorflow as tf
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string


import newsapi
from newsapi import NewsApiClient
import requests
import json
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

## GO HERE...GET API KEY...
## https://newsapi.org/
## https://newsapi.org/register
## a base URL is an endpoint
#BaseURL="https://newsapi.org/v1/articles"
BaseURL="https://newsapi.org/v2/top-headlines"

## https://newsapi.org/v1/articles?apiKey= f9fd15b8062345a29826a0a77639b5d1 &source=bbc-news&pageSize=85
######################################################## 
## WAY 1  
'''
URLPost = {'apiKey': 'f9fd15b8062345a29826a0a77639b5d1',
                    'source': 'bbc-news', 
                    'pageSize': 85,
                    'sortBy' : 'top',
                    'totalRequests': 75}

response1=requests.get(BaseURL, URLPost)
#print(response1)
#jsontxt = response1.json()
#print(jsontxt)
'''

# 'from=2019-11-20&'
####################################################

### WAY 2
url = ('https://newsapi.org/v2/everything?'
       'q=electric vehicles&'
       'sortBy=relevance&'
       'source=bbc-news&'
       'pageSize=100&'
       'apiKey=f9fd15b8062345a29826a0a77639b5d1')

print(url)

response2 = requests.get(url)
jsontxt2 = response2.json()
print(jsontxt2,"\n")
#####################################################

## Create a new csv file to save the headlines
MyFILE=open("EV.csv","w")
### Place the column names in - write to the first row
WriteThis="Author,Title,Headline\n"
MyFILE.write(WriteThis)
MyFILE.close()


## Open the file for append
MyFILE=open("EV.csv", "a")
for items in jsontxt2["articles"]:
    print(items)

###-------------------------------------------------------------              
    Author=items["author"]
    
    
    ## CLEAN the Title
    ##----------------------------------------------------------
    ##Replace punctuation with space
    # Accept one or more copies of punctuation         
    # plus zero or more copies of a space
    # and replace it with a single space
    Title=items["title"]
    Title=re.sub(r'[,.;@#?!&$\-\']+', ' ', Title, flags=re.IGNORECASE)
    Title=re.sub(r'\ +', ' ', Title, flags=re.IGNORECASE)
    Title=re.sub(r'\"', ' ', Title, flags=re.IGNORECASE)
    
    # and replace it with a single space
    ## NOTE: Using the "^" on the inside of the [] means
    ## we want to look for any chars NOT a-z or A-Z and replace
    ## them with blank. This removes chars that should not be there.
    Title=re.sub(r'[^a-zA-Z]', " ", Title, flags=re.VERBOSE)
    #Title=re.sub(r'([^\x00-\x7F])+', " ", Title, flags=re.VERBOSE)
    
    ##----------------------------------------------------------
    
    Headline=items["description"]
    Headline=re.sub(r'[,.;@#?!&$\-\']+', ' ', Headline, flags=re.IGNORECASE)
    Headline=re.sub(r'\ +', ' ', Headline, flags=re.IGNORECASE)
    Headline=re.sub(r'\"', ' ', Headline, flags=re.IGNORECASE)
    Headline=re.sub(r'[^a-zA-Z]', " ", Headline, flags=re.VERBOSE)
    Headline=re.sub(r'([^\x00-\x7F])+', " ", Headline, flags=re.VERBOSE)

    #print("Author: ", Author, "\n")
    #print("Title: ", Title, "\n")
    #print("Headline News Item: ", Headline, "\n\n")
    
    WriteThis=Author+ "," + Title + "," + Headline + "\n"
    
    MyFILE.write(WriteThis)
    
## CLOSE THE FILE
MyFILE.close()

#########################################################################


## The output looks like this:
##Author:  BBC News 

##Title:  Pope Francis addresses violence against women on Colombia visit 

##Headline News Item:  Pope Francis calls for respect for 
##"strong and influential" women during a five-day trip to Colombia. 
##--------------------------------------------------------
#FYI    
#do = jsontxt['articles'][0]["author"]
#print(do)

############### PROCESS THE FILE ######################
## https://stackoverflow.com/questions/21504319/python-3-csv-file-giving-unicodedecodeerror-utf-8-codec-cant-decode-byte-err
## Read to DF
BBC_DF=pd.read_csv("EV.csv", error_bad_lines=False)
print(BBC_DF.head())
# iterating the columns 
for col in BBC_DF.columns: 
    print(col) 
    
print(BBC_DF["Headline"])

### Tokenize and Vectorize the Headlines
## Create the list of headlines
HeadlineLIST=[]
for nextH in BBC_DF["Headline"]:
    HeadlineLIST.append(nextH)

print("The headline list is")
print(HeadlineLIST)

### Vectorize
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
MyCountV=CountVectorizer(input="content", lowercase=True, stop_words = "english")
 
MyDTM = MyCountV.fit_transform(HeadlineLIST)  # create a sparse matrix
print(type(MyDTM))
#vocab is a vocabulary list
vocab = MyCountV.get_feature_names()  # change to a list

MyDTM = MyDTM.toarray()  # convert to a regular array
print(list(vocab)[10:20])
ColumnNames=MyCountV.get_feature_names()
MyDTM_DF=pd.DataFrame(MyDTM,columns=ColumnNames)
print(MyDTM_DF)


#----------------------------------------------------

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
#######

#MyVectLDA_DH=CountVectorizer(input='filename')
##path="C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\DATA\\SmallTextDocs"
#Vect_DH = MyVectLDA_DH.fit_transform(ListOfCompleteFiles)
#ColumnNamesLDA_DH=MyVectLDA_DH.get_feature_names()
#CorpusDF_DH=pd.DataFrame(Vect_DH.toarray(),columns=ColumnNamesLDA_DH)
#print(CorpusDF_DH)

######

num_topics = 7

lda_model_DH = LatentDirichletAllocation(n_components=num_topics, 
                                         max_iter=1000, learning_method='online')
#lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
LDA_DH_Model = lda_model_DH.fit_transform(MyDTM_DF)


print("SIZE: ", LDA_DH_Model.shape)  # (NO_DOCUMENTS, NO_TOPICS)

# Let's see how the first document in the corpus looks like in
## different topic spaces
#print("First headline...")
#print(LDA_DH_Model[0])
#print("Sixth headline...")
#print(LDA_DH_Model[5])

#print(lda_model_DH.components_)


## implement a print function 
## REF: https://nlpforhackers.io/topic-modeling/
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic:  ", idx)
      
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])
                        ## gets top n elements in decreasing order
    

####### call the function above with our model and CountV
print_topics(lda_model_DH, MyCountV)



## Print LDA using print function from above
########## Other Notes ####################
#import pyLDAvis.sklearn as LDAvis
#import pyLDAvis
#import pyLDAvis.gensim 
## conda install -c conda-forge pyldavis
#pyLDAvis.enable_notebook() ## not using notebook
#panel = LDAvis.prepare(lda_model_DH, MyDTM_DF, MyCountV, mds='tsne')
#pyLDAvis.show(panel)
#panel = pyLDAvis.gensim.prepare(lda_model_DH, MyDTM, MyCountV, mds='tsne')
#pyLDAvis.show(panel)
##########################################################################

import matplotlib.pyplot as plt
import numpy as np

word_topic = np.array(lda_model_DH.components_)
word_topic = word_topic.transpose()

num_top_words = 8 ##
vocab_array = np.asarray(vocab)

#fontsize_base = 70 / np.max(word_topic) # font size for word with largest share in corpus
fontsize_base = 20

for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

plt.tight_layout()
plt.show()



