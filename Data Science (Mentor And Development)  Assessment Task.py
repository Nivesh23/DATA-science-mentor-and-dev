#!/usr/bin/env python
# coding: utf-8

# # 3. Problem Solving
# - Write a short guidance note explaining feature selection techniques in machine learning to a hypothetical student struggling with the concept.
# 

# Hello,
# 
# I understand that you're struggling with the concept of feature selection, so let's break it down. At its core, feature selection is a process used in machine learning to select the most useful features (or inputs) for your model.
# 
# ## Why Feature Selection Matters 
# 
# Feature selection is crucial because:
# 
# 1. **Simplification:** It reduces the complexity of a model and makes it easier to interpret.
# 2. **Speeds up Learning:** Fewer data means faster training times.
# 3. **Prevents Overfitting:** Less redundant data means less opportunity to make decisions based on noise.
# 4. **Improves Accuracy:** Less misleading data means modeling accuracy improvement.
# 
# ## Main Techniques for Feature Selection
# 
# There are several techniques commonly used for feature selection:
# 
# ### 1. Univariate Selection
# Univariate selection uses statistical tests to select those features that have the strongest relationship with the output variable.
# 
# For example, chi-squared test can be applied between each categorical input variable and class output variable - variables with highest chi-squared statistics are selected.
# 
# ### 2. Recursive Feature Elimination
# In this method, an estimator/model is trained on initial set of features and importance of each feature is obtained either through `coef_` or `feature_importances_` attribute like in case of logistic regression or random forest respectively then least important features are pruned from current set iteratively till we reach desired number of top n_features_to_select which contribute most towards predicting target variable .
# 
# ### 3. Principle Component Analysis (PCA)
# PCA involves linear algebra operations which results into principal components explaining maximum variance within dataset . These components act as new derived/transformed variables replacing original ones while retaining essence/most information embedded within original dataset but bear in mind PCA doesn't consider output label while performing transformations hence its unsupervised method .
# 
# ### 4.Correlation Matrix with Heatmap
# Correlation matrix gives correlation scores between pair-wise attributes implying how closely changes in one variable corresponds to change in another – higher absolute value indicates stronger correlation & possible redundancy among them hence weaker ones could be dropped using certain threshold cut-off .
# Remember Pearson/Spearman/Kendall methods only applicable when dealing continuous/categorical type attributes respectively .
# 
# Note: While these techniques can provide good starting points, they do not replace domain knowledge about what factors should influence predictions made by your model.
# 
# 
# Don't hesitate to ask if you have any questions!
# 
# 

# ### Assessment Questions:

# # 1. Explain how you would handle missing data in a given dataset and provide a code snippet demonstrating this.
# 

# In[ ]:


import pandas as pd


# In[3]:


df=pd.read_csv("House_data.csv")


# In[9]:


df


# In[6]:


df.sample(5)


# In[8]:


df.shape


# In[10]:


df.isnull()


# In[11]:


df.isnull().sum()


# In[12]:


df.isnull().sum().sum()


# ### Filling null values

# In[14]:


df2=df.fillna(value=0)


# In[15]:


df2


# In[18]:


df2.isnull().sum().sum()


# In[20]:


df3=df1.fillna(value=5)


# #filling with previous value

# In[21]:


df4=df.fillna(method='pad')
df4


# In[24]:


df4.isnull().sum()


# In[25]:


#filling with next value


# In[26]:


df5=df.fillna(method='bfill')


# In[27]:


df5


# In[28]:


df6=df.fillna(method='pad',axis=1)#column wise


# In[29]:


df6


# In[31]:


df7=df.fillna(method='bfill',axis=1)#column wise


# In[32]:


df7


# In[33]:


#filling different values in null in different column


# In[36]:


df8=df.fillna({'society':'abcd','balcony':'defg'})


# In[37]:


df8


# In[45]:


#filling na value with mean
df9=df.fillna(value=df['balcony'].mean())


# In[44]:


df9


# In[46]:


#filling na value with max
df10=df.fillna(value=df['balcony'].max())
df10


# ### Drop na method

# In[47]:


df11=df.dropna()


# In[48]:


df11


# In[50]:


df11.isnull().sum()


# In[51]:


df11=df.dropna(how='all')
df11


# In[52]:


df12=df.dropna(how='any')


# In[53]:


df12


# In[56]:


import numpy as np
df13=df.replace(to_replace=np.nan,value=875465)


# In[57]:


df13


# In[59]:


df13=df.replace(to_replace=3.0,value=5.0)
df13


# # Interpolate

# In[67]:


df['balcony']=df['balcony'].interpolate(method='linear')
df


# In[61]:





# # 2. Prepare a high-level lesson plan for an introductory session on deep learning.

# Lesson Plan for an Introductory Session on Deep Learning
# 
# Title: Introduction to Deep Learning
# 
# **Objectives**
# By the end of this session, students will:
# - Understand what deep learning is and its applications.
# - Understand how deep learning differs from traditional machine learning.
# - Be introduced to basic concepts such as neural networks (including different types), activation functions, backpropagation etc.
# 
# **Materials Needed**
# Presentation slides, PC with Python environment set up (preferably Google Colab or Jupyter notebook), Internet connection.
# 
# *Lesson Outline*
# 
# 1. **Introduction (~15 minutes)**
#     - What is AI? Brief overview
#     - Machine Learning vs Deep Learning 
# 
# 2. **Deep Dive into Deep Learning (~30 minutes)**
#     - Neural Networks: Basics and Architecture
#         - Input layer, hidden layers, output layer
#         - Neurons/nodes
#     - Different Types of Neural Networks 
#         - Feedforward Neural Network
#         - Convolutional Neural Network (CNN)
#         - Recurrent Neural Network (RNN)
# 
# 3. **Key Elements of Deep Learning (~20 minutes)**
#     - Activation Functions: Sigmoid, ReLU etc.
#     - Cost Function & Optimization: Gradient Descent 
#     - Backpropagation Algorithm 
# 
# 4. **Applications of Deep Learning (~10 Minutes)** 
#    Present a few examples where deep learning excels like image recognition, natural language processing etc.
# 
# 5. **Hands-On Activity using Tensorflow/Keras in Python Environment (~25 Minutes)**  
#   Simple code along exercise implementing a feedforward network on a simple dataset like MNIST dataset for digit recognition
# 
# 6. **Recap and Q&A Session(~20 Minutes)**  
# 

# # 3. How would you troubleshoot a machine learning model whose performance isn't as expected? Discuss your approach briefly.

# #3.Troubleshooting a Machine Learning Model Whose Performance Isn't As Expected**
# 
# When dealing with underperforming models there are several steps you can take:
# 
# 1. ***Check Data Quality*** : First ensure that your data has been processed correctly – missing values handled appropriately, categorical variables encoded properly etc.
# 
# 2. ***Feature Engineering*** : Consider adding new features or removing irrelevant ones which might be causing noise in the model's predictions.
# 
# 3.. ***Try Different Models*** : If one algorithm isn’t working well try others; it may be case that chosen model doesn't fit well to problem at hand.
# 
# 4.. ***Tune Hyperparameters*** : Most ML algorithms have hyperparameters that control their behavior – tuning these might help improve performance
# 
# 5.. ***Cross Validation*** : Use cross-validation techniques to get better estimate of model performance by dividing training data into k subsets(folds).
# 
# 6.. ***Ensemble Methods*** : Combining predictions from multiple models can often yield more robust results than single individual model approach .
# 
# 7.. ***Analyze Errors**: Look at specific instances where your model performs poorly – understanding why certain errors occur can provide insight into potential improvements.
# 
# 
# Remember always start with simpler approaches first before moving onto more complex methods!
# 

# 

# # 4. Explain in simple terms what Natural Language Processing (NLP) is and its real-world applications.

# Language is a way that humans have been using for communicating with one another
# since the beginning of time. The term ‘natural language’ refers to language that
# has naturally evolved over time due to repeated use by humans. In essence, natural
# language is referred to as the language humans use to communicate with one another.
# Natural language processing, often abbreviated as NLP, refers to the field of
# programming computers to allow the processing and analysis of natural language.
# From something as basic as a computer program to count the number of words
# in a piece of text, to something more complex such as a program that can serve
# replies to questions asked by humans or translate between languages, all qualify as
# NLP. Essentially, regardless of the difficulty level, any task that involves a computer
# dealing with language through a program qualifies as natural language processing.
# Knowing about the range of applications helps us understand the impact of NLP.
# 
# Consider the following example. You are cooking in the kitchen and want your voice
# assistants, such as Alexa or Google Home, to turn on your TV.
# 

# You: Turn on the Living Room TV
# 
# TV turns on
# 
# 
# You: Play the soccer match on NBC Sports 11.
# Match starts playing on your TV
# 
# 
# You: Pause TV
# 
# TV pauses your video
# 
# 
# You: At what temperature should I bake vegetables?
# 
# 
# ‘400 degrees Fahrenheit is the perfect temperature for most vegetables
# for a crispy exterior and a tender interior.’
# 
# 
# You: Play TV
# 
# 
# TV resumes your paused vide

# NLP applications
# The advanced applications of NLP that are discussed and implemented in Section
# V include the following.
# 1. Named-entity recognition: Named entity recognition (NER) is a form of natural
# language processing and is also known as entity extraction, entity identification,
# or entity chunking. This technique identifies segments of key information within
# a piece of text and categorizes the segments into predefined categories such as
# person name, location, date, timestamp, organization name, percentages, codes,
# numbers, and more. See Figure 1.1 for an example.
# FIGURE 1.1 An example of named-entity recognition.
# 2. Keyphrase extraction: Key-phrase extraction is a textual information processing
# task concerned with the automatic extraction of representative and characteristic phrases from a document that express all the key aspects of its content.
# Keyphrases aim to represent a succinct conceptual summary of a text document. They find use in various applications such as digital information management systems for semantic indexing, faceted search, document clustering,
# and classification [129]. See Figure 1.2 for an example.
# FIGURE 1.2 An example of keyphrase extraction.
# 3. Topic modeling: Topic modeling is the process of identifying different topics
# from a set of documents by detecting patterns of words and phrases within
# them as seen in Figure 1.3. Topic modeling finds applications in document
# clustering, text organization, information retrieval from unstructured text, and
# feature selection [24].
# 8  Natural Language Processing in the Real-World
# FIGURE 1.3 Topic modeling.
# 4. Text similarity: Text similarity is a popular NLP application that finds use in
# systems that depend on finding documents with close affinities. A popular example is content recommendations seen on social media platforms. Ever noticed
# that when you search for a particular topic, your next-to-watch recommended
# list gets flooded with very similar content? Credit goes to text similarity algorithms, among some other data points that help inform user interest and
# ranking.
# 5. Text classification: Text classification refers to classifying text into user-defined
# categories. This can be something as basic as binary labels to hundreds and
# thousands of categories. Examples include categorizing social media content
# into topics and consumer complaint categorization in customer service.
# 6. Text summarization: Long blobs of text such as articles, papers, or documents
# are condensed into a summary that aims to retain vital information using text
# summarization techniques. Google News1, the Inshorts app2, and various other
# news aggregator apps take advantage of text summarization algorithms.
# 7. Language detection and translation: Detection of language from text refers to
# language detection. The process of translating text from one language to another is language translation. There exist many pre-trained models for numerous language tasks that can be used right out of the box by practitioners. Most
# models are trained on a particular text language. Such models don’t perform
# as well if used on text of a different language. In such cases, practitioners often
# resort to language detection and translation techniques. Such techniques also
# find use in language translation tools to help people communicate in non-native
# languages.

# In[ ]:




