import typing as t
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from transformers import BertTokenizer
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

from .features import FeaturesDict


class Representation:
    def __init__(self, corpus: t.List[str], corpus_test: t.List[str], rep: str):
        self.corpus = corpus
        self.corpus_test = corpus_test
        self.rep = rep
        self.vectorizer = None
        self.X_train = None
        self.X_test = None
        self.tokenizer = None
        
    def get_vectors(self):
        if self.rep == 'bow':
            return self.bag_of_words()
        elif self.rep == 'fot':
            return self.feature_of_text()
        elif self.rep == 'wp':
            return self.word_pieces()  
        elif self.rep == 'emb':
            return self.embedding()
        elif self.rep == 'bo2':
            return self.bag_of_2_grams()  
        elif self.rep == 'bo3':
            return self.bag_of_3_grams()   
        
    def bag_of_words(self) -> t.List[np.array]:
        self.vectorizer = CountVectorizer()
        self.X_train = self.vectorizer.fit_transform(self.corpus).toarray()
        self.X_test = self.vectorizer.transform(self.corpus_test).toarray()
        return self.X_train, self.X_test
    
    def bag_of_2_grams(self) -> t.List[np.array]:
        self.vectorizer = CountVectorizer(ngram_range=(1, 2))
        self.X_train = self.vectorizer.fit_transform(self.corpus).toarray()
        self.X_test = self.vectorizer.transform(self.corpus_test).toarray()
        return self.X_train, self.X_test
    
    def bag_of_3_grams(self) -> t.List[np.array]:
        self.vectorizer = CountVectorizer(ngram_range=(1, 3))
        self.X_train = self.vectorizer.fit_transform(self.corpus).toarray()
        self.X_test = self.vectorizer.transform(self.corpus_test).toarray()
        return self.X_train, self.X_test
    
    def feature_of_text(self):
        self.vectorizer = DictVectorizer()
        nlp = spacy.load('pl_core_news_lg')
        X_dict = FeaturesDict(nlp).get(self.corpus)
        X_test_dict = FeaturesDict(nlp).get(self.corpus_test)
        self.X_train = self.vectorizer.fit_transform(X_dict).toarray()
        self.X_test = self.vectorizer.transform(X_test_dict).toarray()
        return self.X_train, self.X_test
    
    # bag of word pieces
    def word_pieces(self):
        tokenizer = BertTokenizer.from_pretrained('dkleczek/bert-base-polish-cased-v1')
        train_set = [' '.join(tokenizer.tokenize(text)) for text in self.corpus]
        test_set = [' '.join(tokenizer.tokenize(text)) for text in self.corpus_test]
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(train_set)
        self.X_train = self.vectorizer.transform(train_set)
        self.X_test  = self.vectorizer.transform(test_set)
        return self.X_train, self.X_test
    
    def embedding(self):
        num_words = 1000 #800 basic #500 embed_basic # 1000 lstm # 1000 glove
        maxlen = len(self.corpus[0].split()) #100
        self.tokenizer = Tokenizer(num_words=num_words)
        self.tokenizer.fit_on_texts(self.corpus)
        X_train = self.tokenizer.texts_to_sequences(self.corpus)
        X_test = self.tokenizer.texts_to_sequences(self.corpus_test)      
        self.X_train = pad_sequences(X_train, maxlen=maxlen)
        self.X_test = pad_sequences(X_test, maxlen=maxlen)
        return self.X_train, self.X_test
        