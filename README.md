# Kaggle - Spooky Identification

## app.py: 
Flask application with a single endpoint that accepts text from the Front-End, fits the classifier in .pkl file 

## hyperparametertuning.py:
A class that contains methods and attributes to send appropriate dictionaries to perform GridSearchCV 

## main.py: 
The main function from which the vector space model creations, parameter tuning, GridSearchCV and finally cross validation takes place

## nltk.txt: 
Contains necessary information for certain modules from nltk library to be downloaded

## requirements.txt: 
Contains all the necessary libraries with its appropriate versions to be downloaded

## spooky_author_model.pkl: 
The pickle file that contains the best classifier

## vectorspace.py: 
A class that contains necessary methods and attributes to create vector space models
