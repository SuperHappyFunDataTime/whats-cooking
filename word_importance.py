#-----------------------------
#
#  This script runs a Random Forest
#  on the recipe data by WORDS (not ingredients),
#  and then spits out the feature importance
#
#-----------------------------

# Import libraries
import numpy as np
import random
import logging
import json
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from scipy import sparse
import matplotlib.pyplot as plt

# Set Working Directory
os.chdir('/home/nick/Documents/Kaggle/Recipes/')

##----Format Logging-----
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

##----Load Data-----
with open('train.json') as data_file:    
    train = json.load(data_file)

##----ETL----
# Create joined recipe string
# Create documents
for entry in train:
    entry['ingredient_string'] = ' '.join(entry['ingredients'])
    
#-----Create list of ingredients-----
ingredient_list = [x['ingredient_string'] for x in train]
target = [x['cuisine'] for x in train]

#---Split Train set into train-train and train-test---
# Not sure we need the following because we are doing a RF
#  on the whole training set.  But we can always split as follows...
indices = range(len(ingredient_list))
percent = 0.8
random.shuffle(indices)
how_many = int(np.round(percent * len(train)))
train_train_indices = indices[0:how_many]
train_test_indices = indices[how_many:]

train_train = [x for ix, x in enumerate(ingredient_list) if ix in train_train_indices]
train_test = [x for ix, x in enumerate(ingredient_list) if ix in train_test_indices]

train_train_target = [x for ix, x in enumerate(target) if ix in train_train_indices]
train_test_target = [x for ix, x in enumerate(target) if ix in train_test_indices]

#----Bag of words transform-----
vectorizer = CountVectorizer(analyzer = 'word',
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = None)
# trained_features = vectorizer.fit_transform(train_train)
trained_features = vectorizer.fit_transform(ingredient_list)

#-----Train Random Forest-----
recipe_forest = RandomForestClassifier(n_estimators = 100)
recipe_forest = recipe_forest.fit(trained_features, target)

#----Feature Importance-----
importances = recipe_forest.feature_importances_
ind_sort = np.argsort(importances)[::-1]

importances = importances[ind_sort]
ingredient_names = vectorizer.get_feature_names()
ingredient_names = [ingredient_names[x] for x in ind_sort]

# Plot feature importances
plt.bar(range(len(importances)), importances)
