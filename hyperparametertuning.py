"""
hyperparametertuning.py

(C) 2018 by Abhishek Babuji <abhishekb2209@gmail.com>

Contains methods to return a pipeline object and a dictionary containing
classifier parameters
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

class HyperParameterTuning:
    """
    Contains methods to return a pipeline object and a dictionary containing
    classifier parameters
    """

    def __init__(self, classifier, vectorizer):
        """
        Args:
            classifier (One of 6 sklearn classifier objects): 'logreg', 'svm', 'nb',
                                                              'knn', 'xgboost', 'randomforests'
            vectorizer (CountVectorizer or TfidfVectorizer): Type of vector space model


        Returns:
            pipeline (sklearn pipeline object): Returns a pipeline object which is used
                                                by GridSearchCV
            model_params[self.classifier] (dict): Returns a dictionary of parameters
                                                  for the specified type of classifier

        """

        self.classifier = classifier
        self.vectorizer = vectorizer

    def get_pipeline(self):
        """
        Args:

            classifier (One of 6 sklearn classifier objects): 'logreg', 'svm', 'nb',
                                                              'knn', 'xgboost', 'randomforests'
            vectorizer (CountVectorizer or TfidfVectorizer): Type of vector space model


        Returns:
            pipeline (sklearn pipeline object): Returns a pipeline object which is
                                                used by GridSearchCV
            model_params[self.classifier] (dict): Returns a dictionary of parameters
                                                  for the specified type of classifier
        """

        classifier_objects = {'logreg': LogisticRegression(),
                              'svm': SVC(),
                              'knn': KNeighborsClassifier(),
                              'xgboost': GradientBoostingClassifier(),
                              'randomforests': RandomForestClassifier(),
                              'nb': MultinomialNB()}
        pipeline = Pipeline([('vect', self.vectorizer),
                             ('clf', classifier_objects[self.classifier])])

        return pipeline

    def get_params(self):
        """
        Args:
            self


        Returns:
            model_params[self.classifier] (dict): Returns a dictionary of parameters for the
                                                  specified type of classifier

        """
        model_params = {'logreg': {'clf__C': (1, 10, 100), 'clf__penalty': ('l1', 'l2')},
                        'svm': {'clf__C': (1, 10, 100),
                                'clf__kernel': ('linear', 'poly', 'rbf', 'sigmoid')},
                        'knn': {'clf__n_neighbors': (5, 10, 50, 100)},
                        'xgboost': {'clf__n_estimators': (100, 500, 1000)},
                        'randomforests': {'clf__n_estimators': (100, 500, 1000)},
                        'nb': {'clf__alpha': (0, 1), 'clf__fit_prior': (True, False)}}
        return model_params[self.classifier]
