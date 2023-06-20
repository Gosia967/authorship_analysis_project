import numpy as np
import os

from random import sample, choice

from .data_preparation import DataPreparation

class ArpaPreparation(DataPreparation):
    
    def create_train_txt_corpus_for_arpa(self, train_size_words: int):       
        path = os.path.join('..','arpa_files','train_corpuses')      
        for author in self.books_text:
            corpus = ''
            while len(corpus.split()) < train_size_words and len(self.books_text[author]) > 0:
                book = choice(self.books_text[author])
                if len(corpus.split()) + len(book.split()) <= train_size_words:
                    corpus += book
                    self.books_text[author].remove(book)
                else:
                    missing_words_num = train_size_words - len(corpus.split())
                    book_words = book.split()
                    corpus += ' '.join(book_words[:missing_words_num])
                    corpus += '\n'
                    rest_of_book = ' '.join(book_words[missing_words_num:])
                    self.books_text[author].remove(book)
                    if train_size_words < 100000:
                        self.books_text[author].append(rest_of_book)
            self._save_to_file(path, author, corpus)
     
        
    def get_test_corpus(self, test_size: int, par_len: int):
        test_set = []
        test_classes = []
        for author in self.books_text:
            books = ''.join(self.books_text[author])
            paragraphs = self._get_paragraphs(books, par_len)
            if len(paragraphs) >= test_size:
                author_set = list(set(sample(paragraphs, test_size)))
                test_set += author_set
                test_classes += [author] * len(author_set)
        return [np.array(test_set), np.array(test_classes)]
                    
            
    def _get_paragraphs(self, book: str, par_len: int):
        if not par_len:
            return [p for p in book.split('. ') if len(p)>0] 
        words = book.split()
        if len(words) == 0:
            return []
        sections = [words[i:i+par_len] for i in range(0, len(words) - par_len, par_len)] if len(words)>par_len else [[words]]
        return [' '.join(sec) for sec in sections]
        
    def _save_to_file(self, path: str, author: str, corpus: str) -> None:
        path = os.path.join(path, author + '.txt')
        with open(path, 'w') as f:
            f.write(corpus)
             