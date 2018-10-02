"""
main.py

(C) 2017 by Abhishek Babuji <abhishekb2209@gmail.com>

Contains methods to read in the csv file, create combination of datasets
with reductions and weighting schemes and rune and fit the model
to different datasets using 6 classifiers
"""

import itertools
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from hyperparametertuning import HyperParameterTuning
from vectorspace import VectorSpace


def tune_fit_model(transformed_data, parameter_desc, train):
    """
    Args:
        transformed_data (list of lists): Every list, has two components.
                                          1. vectorizer: which is either CountVectorizer or
                                             TfidfVectorizer
                                          2. reduced_data: has the training set with
                                             reduction such as stemming or lemmatization
                                             applied

        parameter_desc (list of lists): For each list in transformed_data, the corresponding list
                                        in the same index have description about the data so it's
                                        easy to plot graphs or sort once model accuracies have been
                                        calculated

        train (Pandas DataFrame): contains the training data including the labels

    Returns:
         None
    """

    count = 0
    params = []
    for vectorizer, reduced_data in transformed_data:

        for classifier in {'nb', 'svm', 'logreg', 'knn',
                           'xgboost', 'randomforests'}:
            print("Data: ", parameter_desc[count])
            print("Classfier: ", classifier)
            x_train, x_test, y_train, y_test = train_test_split(reduced_data,
                                                                train['author'],
                                                                stratify=train['author'],
                                                                test_size=0.25)
            hyperparam_instance = HyperParameterTuning(classifier, vectorizer)
            search = GridSearchCV(hyperparam_instance.get_pipeline(),
                                  param_grid=hyperparam_instance.get_params(),
                                  cv=5, scoring='accuracy',
                                  n_jobs=-2)
            search.fit(x_train, y_train)
            y_pred = search.predict(x_test)
            print("Validation accuracy", accuracy_score(y_test, y_pred))
            print("Best parameter (CV score=%0.3f):" % search.best_score_)
            print(search.best_params_)
            param = [parameter_desc[count], classifier, search.best_score_,
                     accuracy_score(y_test, y_pred), search.best_params_]
            params.append(param)
            print(param)
            print()
        count += 1

    param_df = pd.DataFrame(params, columns=['Dataset', 'Classifier', 'Training accuracy',
                                             'Validation accuracy', 'Classifier object'])
    param_df.to_csv("/Users/abhishekbabuji/Desktop/ModelPerformances.csv")


def main():
    """
    The main function that reads in the train.csv from my local file system
    """

    train = pd.read_csv("/Users/abhishekbabuji/Downloads/all/train.csv")
    datasets = [[train['text']], ['TF', 'TFIDF'], ['stem', 'lemmatize', None], ['english', None],
                [(1, 1), (1, 2), (2, 2)]]
    dataset_combination = list(itertools.product(*datasets))

    transformed_data = []
    parameter_desc = []
    count = 0

    for dataset_params in dataset_combination:
        count += 1
        print("Transformed data no. ", count)
        model = VectorSpace(dataset_params[0], dataset_params[1], dataset_params[2],
                            dataset_params[3], dataset_params[4])
        vectorizer, reduced_data = model.create_vec_space()

        transformed_data.append([vectorizer, reduced_data])
        parameter_desc.append([dataset_params[1], dataset_params[2],
                               dataset_params[3], dataset_params[4]])
        print("Vector space transformation applied with parameters: ",
              dataset_params[1], dataset_params[2], dataset_params[3], dataset_params[4])
        print()

    tune_fit_model(transformed_data, parameter_desc, train)


if __name__ == '__main__':
    main()
