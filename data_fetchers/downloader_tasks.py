import os
from data_fetchers.txt_book_fetcher import TXTBookFetcher
from data_fetchers.books import BookSet
from data_fetchers.text_adjuster import TextAdjuster
from data_fetchers.get_polish_authors import get_polish_authors_list


def download_book_txt(author: str, title: str):
    bookset = BookSet()
    bookset.fetch()
    book = bookset.get_book_by_title_and_author(title, author)
    txt_fetcher = TXTBookFetcher(book)
    downloaded_text = txt_fetcher.fetch_txt()
    adjuster = TextAdjuster()
    book_text = adjuster.cut_footer(downloaded_text)
    if adjuster.detect_polish(book_text):
        return book_text


def list_all_books_by_author(author: str):
    bookset = BookSet()
    bookset.fetch()
    return [book.title for book in bookset.get_all_books_by_author(author)]
    
def download_all_books_by_author(author: str, path: str):
    books_titles = list_all_books_by_author(author)
    for title in books_titles:
        book_txt = download_book_txt(author, title)
        if book_txt:
            save_txt_in_file(book_txt, title, author, path)
        
        
def save_txt_in_file(txt: str, title: str, author: str, path: str):
    adjuster = TextAdjuster()
    adjusted_title = adjuster.adjust_title_for_filename(title)
    file_path = os.path.join(path, author, adjusted_title + '.txt')
    with open(file_path, 'w',  encoding="utf-8") as f:
        f.write(txt)
        
def get_last_downloaded_author_id(path: str):
    authors = get_polish_authors_list()
    for i in range(len(authors) - 1):
        if not os.path.isdir(os.path.join(path, authors[i+1])):
            return i
                   
        
def download_all(path):
    authors = get_polish_authors_list()
    last_author_id = get_last_downloaded_author_id(path)
    for i in range(last_author_id, len(authors)):
        author_path = os.path.join(path, authors[i])
        try:
            os.mkdir(author_path)
            download_all_books_by_author(authors[i], path)
        except:
            pass
        finally:
            if len(os.listdir(author_path)) == 0:
                os.rmdir(author_path)
       

path = os.path.join('..', 'books_downloads', 'authors')
download_all(path)