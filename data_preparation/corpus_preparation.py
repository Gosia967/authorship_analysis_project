import numpy as np
import typing as t
from random import sample, choice

from .data_preparation import DataPreparation

class CorpusPreparation(DataPreparation):
    
        
    # returns train_set, train_classes, test_set, test_classes  
    # randomly part of each book in traning set and part in test set
    def prepare_corpus_mixed_books(self, train_size: int, test_size: int) -> t.List[np.array]:
        train_set = []
        train_classes = []
        test_set = []
        test_classes = []
        for author in self.books_text:
            author_pars = set([par for book in self.books_text[author] for par in book])
            if len(author_pars) >= train_size + test_size:
                author_train_set = set(sample(author_pars, train_size))
                author_pars -= author_train_set
                author_test_set = set(sample(author_pars, test_size))
                train_set += list(author_train_set)
                train_classes += [author] * train_size
                test_set += list(author_test_set)
                test_classes += [author] * test_size
        return [np.array(train_set), np.array(train_classes), np.array(test_set), np.array(test_classes)]
                 
    # returns train_set, train_classes, test_set, test_classes  
    # given book either in traning set or in test set
    def prepare_corpus_divide_books(self, train_size: int, test_size: int) -> t.List[np.array]:
        train_set = []
        train_classes = []
        test_set = []
        test_classes = []
        for author in self.books_text:
            author_train_pars, author_test_pars = self._divide_books_for_sets(self.books_text[author], test_size)
            if len(author_train_pars) >= train_size and len(author_test_pars) >= test_size:
                author_train_set = set(sample(author_train_pars, train_size))
                author_test_set = set(sample(author_test_pars, test_size))
                train_set += list(author_train_set)
                train_classes += [author] * train_size
                test_set += list(author_test_set)
                test_classes += [author] * test_size
        return [np.array(train_set), np.array(train_classes), np.array(test_set), np.array(test_classes)]                                                                         
    def _divide_books_for_sets(self, books_text: list, test_size: int):
        test_pars = set()
        while len(test_pars) < test_size and len(books_text)>0:
            book = choice(books_text)
            test_pars.update(book)
            books_text.remove(book)
        train_pars = set([par for book in books_text for par in book])
        return train_pars, test_pars
              