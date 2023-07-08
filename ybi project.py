#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('Movies Recommendation.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df_features = df[['Movie_Genre','Movie_Keywords','Movie_Tagline','Movie_Cast','Movie_Director']].fillna('')


# In[8]:


df_features.shape


# In[9]:


df_features


# In[10]:


x = df_features['Movie_Genre']+''+df_features['Movie_Keywords']+''+df_features['Movie_Tagline']+''+df_features['Movie_Cast']+''+df_features['Movie_Director']
x


# In[11]:


x.shape


# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[13]:


tfidf= TfidfVectorizer()


# In[14]:


x = tfidf.fit_transform(x)


# In[15]:


x.shape


# In[16]:


print(x)


# In[17]:


from sklearn.metrics.pairwise import cosine_similarity


# In[18]:


similarity_score = cosine_similarity(x)


# In[19]:


similarity_score 


# In[20]:


similarity_score.shape


# In[26]:


Favourite_Movie_Name = input('enter your favourite movie name : ')


# In[27]:


All_Movies_Title_List = df['Movie_Title'].tolist()


# In[28]:


import difflib


# In[29]:


Movie_Recommendation = difflib.get_close_matches(Favourite_Movie_Name, All_Movies_Title_List )
print(Movie_Recommendation)


# In[30]:


Close_match = Movie_Recommendation[0]
print(Close_match)


# In[31]:


Index_of_Close_match_movie = df[df.Movie_Title ==Close_match]['Movie_ID'].values[0]
print(Index_of_Close_match_movie )


# In[32]:


Recommendation_score = list(enumerate(similarity_score[Index_of_Close_match_movie]))
print(Recommendation_score)


# In[33]:


len(Recommendation_score)


# In[34]:


sorted_similar_movies = sorted(Recommendation_score, key = lambda x:x[1],reverse = True)
print(sorted_similar_movies)


# In[35]:


print('Top 30 movies suggested for you:\n')

i = 1


for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = df[df.index == index]['Movie_Title'].values[0]
    if (i < 31):  
        print(i, '.', title_from_index)
        i += 1
    else:
        break


# In[38]:


import difflib

Movie_Name = input('Enter your favorite movie name: ')

list_of_all_titles = df['Movie_Title'].tolist()

Find_Close_Match = difflib.get_close_matches(Movie_Name, list_of_all_titles)

Close_Match = Find_Close_Match[0]
Index_of_Movie = df[df.Movie_Title == Close_Match]['Movie_ID'].values[0]
Recommendation_Score = list(enumerate(similarity_score[Index_of_Movie]))
sorted_similar_movies = sorted(Recommendation_Score, key=lambda x: x[1], reverse=True)

print("Top 10 Movies suggested for you: \n")
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = df[df.Movie_ID == index]['Movie_Title'].values
    if i < 11:
        print(i, '.', title_from_index)
        i += 1


# In[ ]:





# In[ ]:




