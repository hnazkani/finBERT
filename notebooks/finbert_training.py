#!/usr/bin/env python
# coding: utf-8

# # FinBERT Example Notebook
# 
# This notebooks shows how to train and use the FinBERT pre-trained language model for financial sentiment analysis.

# ## Modules 

# In[1]:


from pathlib import Path
import shutil
import os
import logging
import sys
sys.path.append('..')

from textblob import TextBlob
from pprint import pprint
from sklearn.metrics import classification_report

from transformers import AutoModelForSequenceClassification

from finbert.finbert import *
import finbert.utils as tools

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

project_dir = Path.cwd().parent
#pd.set_option('max_colwidth', -1)


# In[2]:


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.ERROR)


# ## Prepare the model

# ### Setting path variables:
# 1. `lm_path`: the path for the pre-trained language model (If vanilla Bert is used then no need to set this one).
# 2. `cl_path`: the path where the classification model is saved.
# 3. `cl_data_path`: the path of the directory that contains the data files of `train.csv`, `validation.csv`, `test.csv`.
# ---
# 
# In the initialization of `bertmodel`, we can either use the original pre-trained weights from Google by giving `bm = 'bert-base-uncased`, or our further pre-trained language model by `bm = lm_path`
# 
# 
# ---
# All of the configurations with the model is controlled with the `config` variable. 

# In[3]:


lm_path = project_dir/'models'/'language_model'/'finbertTRC2'
cl_path = project_dir/'models'/'classifier_model'/'finbert-sentiment'
cl_data_path = project_dir/'data'/'sentiment_data'


# ###  Configuring training parameters

# You can find the explanations of the training parameters in the class docsctrings. 

# In[4]:


# Clean the cl_path
try:
    shutil.rmtree(cl_path) 
except:
    pass

bertmodel = AutoModelForSequenceClassification.from_pretrained(lm_path,cache_dir=None, num_labels=3)


config = Config(   data_dir=cl_data_path,
                   bert_model=bertmodel,
                   num_train_epochs=4,
                   model_dir=cl_path,
                   max_seq_length = 48,
                   train_batch_size = 32,
                   learning_rate = 2e-5,
                   output_mode='classification',
                   warm_up_proportion=0.2,
                   local_rank=-1,
                   discriminate=True,
                   gradual_unfreeze=True)


# `finbert` is our main class that encapsulates all the functionality. The list of class labels should be given in the prepare_model method call with label_list parameter.

# In[5]:


finbert = FinBert(config)
finbert.base_model = 'bert-base-uncased'
finbert.config.discriminate=True
finbert.config.gradual_unfreeze=True


# In[6]:


finbert.prepare_model(label_list=['positive','negative','neutral'])


# ## Fine-tune the model

# In[7]:


# Get the training examples
train_data = finbert.get_data('train')


# In[8]:


model = finbert.create_the_model()


# ### [Optional] Fine-tune only a subset of the model
# The variable `freeze` determines the last layer (out of 12) to be freezed. You can skip this part if you want to fine-tune the whole model.
# 
# <span style="color:red">Important: </span>
# Execute this step if you want a shorter training time in the expense of accuracy.

# In[9]:


# This is for fine-tuning a subset of the model.

freeze = 6

for param in model.bert.embeddings.parameters():
    param.requires_grad = False
    
for i in range(freeze):
    for param in model.bert.encoder.layer[i].parameters():
        param.requires_grad = False


# ### Training

# In[10]:


trained_model = finbert.train(train_examples = train_data, model = model)


# ## Test the model
# 
# `bert.evaluate` outputs the DataFrame, where true labels and logit values for each example is given

# In[11]:


test_data = finbert.get_data('test')


# In[12]:


results = finbert.evaluate(examples=test_data, model=trained_model)


# ### Prepare the classification report

# In[13]:


def report(df, cols=['label','prediction','logits']):
    #print('Validation loss:{0:.2f}'.format(metrics['best_validation_loss']))
    cs = CrossEntropyLoss(weight=finbert.class_weights)
    loss = cs(torch.tensor(list(df[cols[2]])),torch.tensor(list(df[cols[0]])))
    print("Loss:{0:.2f}".format(loss))
    print("Accuracy:{0:.2f}".format((df[cols[0]] == df[cols[1]]).sum() / df.shape[0]) )
    print("\nClassification Report:")
    print(classification_report(df[cols[0]], df[cols[1]]))


# In[14]:


results['prediction'] = results.predictions.apply(lambda x: np.argmax(x,axis=0))


# In[15]:


report(results,cols=['labels','prediction','predictions'])


# ### Get predictions

# With the `predict` function, given a piece of text, we split it into a list of sentences and then predict sentiment for each sentence. The output is written into a dataframe. Predictions are represented in three different columns: 
# 
# 1) `logit`: probabilities for each class
# 
# 2) `prediction`: predicted label
# 
# 3) `sentiment_score`: sentiment score calculated as: probability of positive - probability of negative
# 
# Below we analyze a paragraph taken out of [this](https://www.economist.com/finance-and-economics/2019/01/03/a-profit-warning-from-apple-jolts-markets) article from The Economist. For comparison purposes, we also put the sentiments predicted with TextBlob.
# > Later that day Apple said it was revising down its earnings expectations in the fourth quarter of 2018, largely because of lower sales and signs of economic weakness in China. The news rapidly infected financial markets. Apple’s share price fell by around 7% in after-hours trading and the decline was extended to more than 10% when the market opened. The dollar fell by 3.7% against the yen in a matter of minutes after the announcement, before rapidly recovering some ground. Asian stockmarkets closed down on January 3rd and European ones opened lower. Yields on government bonds fell as investors fled to the traditional haven in a market storm.

# In[16]:


text = "Later that day Apple said it was revising down its earnings expectations in the fourth quarter of 2018, largely because of lower sales and signs of economic weakness in China. The news rapidly infected financial markets. Apple’s share price fell by around 7% in after-hours trading and the decline was extended to more than 10% when the market opened. The dollar fell by 3.7% against the yen in a matter of minutes after the announcement, before rapidly recovering some ground. Asian stockmarkets closed down on January 3rd and European ones opened lower. Yields on government bonds fell as investors fled to the traditional haven in a market storm."


# In[17]:


cl_path = project_dir/'models'/'classifier_model'/'finbert-sentiment'
model = AutoModelForSequenceClassification.from_pretrained(cl_path, cache_dir=None, num_labels=3)


# In[18]:


import nltk
nltk.download('punkt')


# In[19]:


result = predict(text,model)


# In[20]:


blob = TextBlob(text)
result['textblob_prediction'] = [sentence.sentiment.polarity for sentence in blob.sentences]
result


# In[21]:


print(f'Average sentiment is %.2f.' % (result.sentiment_score.mean()))


# Here is another example

# In[22]:


text2 = "Shares in the spin-off of South African e-commerce group Naspers surged more than 25% in the first minutes of their market debut in Amsterdam on Wednesday. Bob van Dijk, CEO of Naspers and Prosus Group poses at Amsterdam's stock exchange, as Prosus begins trading on the Euronext stock exchange in Amsterdam, Netherlands, September 11, 2019. REUTERS/Piroschka van de Wouw Prosus comprises Naspers’ global empire of consumer internet assets, with the jewel in the crown a 31% stake in Chinese tech titan Tencent. There is 'way more demand than is even available, so that’s good,' said the CEO of Euronext Amsterdam, Maurice van Tilburg. 'It’s going to be an interesting hour of trade after opening this morning.' Euronext had given an indicative price of 58.70 euros per share for Prosus, implying a market value of 95.3 billion euros ($105 billion). The shares jumped to 76 euros on opening and were trading at 75 euros at 0719 GMT."


# In[23]:


result2 = predict(text2,model)
blob = TextBlob(text2)
result2['textblob_prediction'] = [sentence.sentiment.polarity for sentence in blob.sentences]


# In[24]:


result2


# In[25]:


print(f'Average sentiment is %.2f.' % (result2.sentiment_score.mean()))


# In[ ]:




