import typing as t
import math
import spacy
import os

from sys import platform

from .preparation_utils import Splitter, Reader, Preprocessing, authors_abbr

class DataPreparation:
    def __init__(
        self, 
        books_epoch_list: t.List[t.Tuple[str, str]] = None
    ):
        self.books_epoch_list = books_epoch_list
        self.books_text = {}
        
    def get_books(self) -> t.List[t.List[str]]:
        path = os.path.join('..', 'books_downloads', 'authors')
        for author, title in sum(self.books_epoch_list, []):
            book = Reader().read_book_by_author_and_title(path, author, title)
            if len(book) > 0:
                if author in self.books_text:
                    self.books_text[author].append(book)
                else:
                    self.books_text[author] = [book]
            
    def books_preprocessing(self, prep_func_names: t.List[str], pos: t.List[str] = []):
        nlp = spacy.load('pl_core_news_lg')
        nlp.max_length = 2500000
        for author in self.books_text:
            processed_books = []
            for book_text in self.books_text[author]:
                preprocessing = Preprocessing(book_text, nlp)
                for pf_name in prep_func_names:
                    pf = getattr(preprocessing, pf_name)
                    if pf_name == 'POS_leave_only':
                        pf(pos)
                    else:
                        pf()
                processed_books.append(preprocessing.text)
                print(author, flush=True)
            self.books_text[author] = processed_books
                    
        
    def split_books(self, par_len: int = None):
        for author in self.books_text:
            self.books_text[author] = [Splitter().split_book(book, par_len) for book in self.books_text[author]]
                    
    def limit_to_n_most_prolific_authors(self, limit_n: int = math.inf):
        counts = [author for author in self._count_authors_pars()]
        most_prolific_authors = []
        for books_epoch in self.books_epoch_list:
            authors_list = [author for (author, title) in books_epoch]
            authors_set = sorted(list(set(authors_list)))
            most_prolific_authors += [author for author in counts if author in authors_set][:limit_n]
        self.books_text = {author:self.books_text[author] for author in self.books_text if author in most_prolific_authors}
              
    def _count_authors_pars(self) -> dict:
        par_nums = {author:len([par for book in self.books_text[author] for par in book]) for author in self.books_text}
        print(par_nums)
        return dict(sorted(par_nums.items(), key=lambda item: item[1], reverse=True))
    
    def read_corpus_form_frozen_dataset(self, authors, prep_descr, kinds_descr):
        authors_a = authors_abbr(authors)
        dir_path = 'set_'+prep_descr+'_'+kinds_descr+'_'+'_'.join(authors_a)
        for author in authors:
            self.books_text[author] = []
            book_number = 0
            file_path = os.path.join('..', 'datasets', dir_path, author, str(book_number)+'.txt')
            while os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    book = file.read()
                    self.books_text[author].append(book)
                book_number += 1
                file_path = os.path.join('..', 'datasets', dir_path, author, str(book_number)+'.txt')
