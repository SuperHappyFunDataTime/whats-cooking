# This file creates the bag-of-words matrix,
#  then attempts a random forest predictor on the matrix
#  getting an accuracy on the 20% hold out set of ~74.5%
#
# Then a word2vec model is fit to create "better" features,
#  but only getting an accuracy on the 20% hold out set of ~71%
#
import numpy as np
import random
import nltk
import re
import logging
import json
import os
from sklearn.feature_extraction.text import CountVectorizer
import nltk.data
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models, similarities, matutils
from gensim.models import word2vec
from collections import defaultdict
from scipy import sparse

os.chdir('/home/nick/Documents/Kaggle/Recipes/')

##----Format Logging-----
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

##-----NLTK Data----
# nltk.download()

##----Import Data-----
with open('train.json') as data_file:    
    train = json.load(data_file)

with open('test.json') as data_file:
    test = json.load(data_file)

# Create documents
for entry in train:
    entry['ingredient_string'] = ' '.join(entry['ingredients'])

#-----Create list of ingredients-----
ingredient_list = [x['ingredient_string'] for x in train]
target = [x['cuisine'] for x in train]

#---Split Train set into train-train and train-test---
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
                             max_features = 2000)
trained_features = vectorizer.fit_transform(train_train)

##-----Train Random Forest-----
recipe_forest = RandomForestClassifier(n_estimators = 500)
recipe_forest = recipe_forest.fit(trained_features, train_train_target)

##----Predict on Random Forest-----
test_features = vectorizer.transform(train_test)
test_prediction = recipe_forest.predict(test_features)

# accuracy
def get_accuracy(prediction, actual):
    assert len(prediction) == len(actual)
    pred_list = list(prediction)
    actual_list = list(actual)
    num_correct = sum([1 for x,y in zip(pred_list, actual_list) if x == y])
    accuracy = float(num_correct)/float(len(actual))
    return(accuracy)

rf_accuracy = get_accuracy(test_prediction, train_test_target)

# Break up strings:
recipe_ingredient_list_train = [x.lower().split() for x in train_train]
recipe_ingredient_list_test = [x.lower().split() for x in train_test]

##-----Train Word2Vec-----
num_features = 500   # Word Vector Dimensionality
min_word_count = 2  # Minimum Word Count
num_workers = 4      # Number of Threads to Run (parallel processing)
context = 6          # Context Window Size
downsampling = 1e-3  # Downsampling for frequent words

# Train model.  This can take a while. To check, type 'top -o cpu' in bash.
w2v_model = word2vec.Word2Vec(recipe_ingredient_list_train,
                              workers = num_workers,
                              size = num_features,
                              min_count = min_word_count,
                              window = context,
                              sample = downsampling)
                              
                              
# When done training, pull all of model into memory
w2v_model.init_sims(replace=True)

##-----Create recipe feature from ingredient features-----
# The following function takes in a ingredient list and outputs a feature.
def make_features(words, model, num_features):
    # Intialize feature
    featureVec = np.zeros((num_features,), dtype='float32')
    # nwords is the count of words that are in our models vocabulary
    nwords=0.0
    # Here is our models vocabulary in set form
    model_names_set = set(model.index2word)
    
    # Loop though word list and add up feature
    for word in words:
        if word in model_names_set:
            nwords = nwords + 1.0
            featureVec = np.add(featureVec, model[word])
    # Divide by number of words in list (to get average)
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return(featureVec)

# Create vectors for each recipe:
recipe_features_train = []
for recipe in recipe_ingredient_list_train:
    recipe_features_train.append(make_features(recipe,
                                               w2v_model,
                                               num_features))
    
recipe_features_test = []
for recipe in recipe_ingredient_list_test:
    recipe_features_test.append(make_features(recipe,
                                              w2v_model,
                                              num_features))


##-----Train Random Forest-----
w2v_forest = RandomForestClassifier(n_estimators = 250)
w2v_forest = w2v_forest.fit(recipe_features_train, train_train_target)

# Make Predictions
w2v_test_prediction = w2v_forest.predict(recipe_features_test)

# Get Accuracy
w2v_accuracy = get_accuracy(w2v_test_prediction, train_test_target)
