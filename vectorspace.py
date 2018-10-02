"""
vectorspace.py

(C) 2017 by Abhishek Babuji <abhishekb2209@gmail.com>

Creates vector space models for combination of weighting factors like TF and TF-IDF
and reduction techniques like Stemming and Lemmatization
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer


class VectorSpace:
    """
    Creates vector space model for training data with specifications of weighting factors,
    reductions, stop words and ngram combination
    """

    def __init__(self, train, weighting_factor=None, reduction=None,
                 stop_words=None, ngrams=None):

        """

        Args:
            train (Pandas DataFrame): the training data
            weighting_factor (Optional argument, None by default, str otherwise)):
                                can take 'TF' or 'IDF'
            reduction (Optional argument, None by default, str otherwise):
                                can take 'stem' or 'lemmatize'
            stop_words (Optional argument, None by default, str otherwise):
                                can take 'english'
            ngrams (Optional argument, None by default, tuple otherwise):
                                can take (1, 1), (1, 2) or (2,2)

        Returns:
            vectorizer (CountVectorizer or TfidfVectorizer object)
            train (Pandas DataFrame): the training data with reduction applied (if any)

        """
        print("Parameters recieved: ", weighting_factor, reduction, stop_words, ngrams)
        self.train = train
        self.weighting_factor = weighting_factor
        self.stop_words = stop_words
        self.ngrams = ngrams
        self.reduction = reduction

    def lemmatize_sentences(self, sentence):
        """

        Args:
            sentence (str): A single sentence from a Pandas DataFrame
                            and applied the reduction (if any)

        Returns:
            lemmatized_tokens (str): A single sentence from a Pandas DataFrame
                            with the reduction applied (if any)

        """
        lemmatizer = WordNetLemmatizer()
        tokens = sentence.split()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

    def stem_sentences(self, sentence):

        """

        Args:
            sentence (str): A single sentence from a Pandas DataFrame
                            and applied the reduction (if any)

        Returns:
            stemmed_tokens (str): A single sentence from a Pandas DataFrame
                            with the reduction applied (if any)

        """

        porter_stemmer = PorterStemmer()
        tokens = sentence.split()
        stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)

    def apply_reduction(self):

        """

        Args:
            self

        Returns:
            train (Pandas DataFrame): Returns the train data instance with the reduction
                                      applied (if any)

        """

        if self.reduction == 'stem':
            print("Performing reduction: stemming")
            self.train = self.train.apply(self.stem_sentences)
        elif self.reduction == 'lemmatize':
            print("Performing reduction: lemmatization")
            self.train = self.train.apply(self.lemmatize_sentences)
        return self.train

    def tf_vectorizer(self):

        """

        Args:
            self

        Returns:
            vectorizer (CountVectorizer object)
            train (Pandas DataFrame): the training data with reduction applied (if any)
        """

        self.train = self.apply_reduction()
        print("Returning CountVectorizer object with parameters: ", self.stop_words, self.ngrams)
        vectorizer = CountVectorizer(stop_words=self.stop_words, ngram_range=self.ngrams)
        return vectorizer, self.train

    def tfidf_vectorizer(self):

        """

        Args:
            self

        Returns:
            vectorizer (TfidfVectorizer object)
            train (Pandas DataFrame): the training data with reduction applied (if any)
        """

        self.train = self.apply_reduction()
        print("Returning TfidfVectorizer object with parameters: ", self.stop_words, self.ngrams)
        vectorizer = TfidfVectorizer(stop_words=self.stop_words, ngram_range=self.ngrams)
        return vectorizer, self.train

    def create_vec_space(self):

        """

        Args:
            self

        Returns:
            vectorizer (TfidfVectorizer object)

        """

        if self.weighting_factor == 'TF':
            return self.tf_vectorizer()
        return self.tfidf_vectorizer()
