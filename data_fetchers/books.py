import typing as t
from data_fetchers.json_info_fetcher import BooksFetcher
from data_fetchers.get_polish_authors import get_polish_authors_list

ACCEPTED_KINDS = ["Dramat", "Liryka", "Epika"]


class Book:
    def __init__(
        self, title: str, author: str, epochs: t.List[str], href: str, kind: str
    ) -> None:
        self.title = title
        self.author = author
        self.epochs = epochs
        self.href = href
        self.kind = kind


class BookSet:
    def __init__(self) -> None:
        self.shelf: t.Dict[(str, str), Book] = dict()
        self.polish_author_list: list = get_polish_authors_list()

    def put_book(self, book: Book) -> None:
        identifier = (book.title, book.author)
        self.shelf[identifier] = book
        
    def get_book_by_title_and_author(self, title: str, author: str) -> Book:
        identifier = (title, author)
        return self.shelf[identifier]
    
    def list_of_books_to_author_title_list(self, books_list: t.List[Book]) -> t.List[t.Tuple[str]]:
        return [(book.author, book.title) for book in books_list]
    
    def get_all_books_by_author(self, author: str) -> t.List[Book]:
        return [self.shelf[book] for book in self.shelf if self.shelf[book].author == author]
    
    def get_books_by_authors_list(self, authors: t.List[str]) -> t.List[Book]:
        return [self.shelf[book] for book in self.shelf if self.shelf[book].author in authors]
    
    def get_books_by_authors_list_kinds(self, authors: t.List[str], kinds: t.List[str]) -> t.List[Book]:
        return [self.shelf[book] for book in self.shelf if self.shelf[book].author in authors and self.shelf[book].kind in kinds]
    
    def get_all_books_in_kind_by_author(self, kind: str, author: str) -> t.List[Book]:
        return [self.shelf[book] for book in self.shelf if self.shelf[book].author == author and self.shelf[book].kind == kind]
    
    def get_all_books_in_epoch_in_kind_by_author(self, epoch: str, kind: str, author: str) -> t.List[Book]:
        return [self.shelf[book] for book in self.shelf if self.shelf[book].author == author and self.shelf[book].kind == kind and epoch in self.shelf[book].epochs]
    
    def get_all_books_in_kind(self, kind: str) -> t.List[Book]:
        return [self.shelf[book] for book in self.shelf if self.shelf[book].kind == kind]
    
    def get_all_books_in_epoch_in_kind(self, epoch: str, kind: str) -> t.List[Book]:
        return [self.shelf[book] for book in self.shelf if self.shelf[book].kind == kind and epoch in self.shelf[book].epochs]
    
    def get_books_epochs_list(self, epochs_kind_list: list) -> t.List[t.List[t.Tuple[str]]]:
        epoch_dict = {}
        for epoch, kind in epochs_kind_list:
            if epoch in epoch_dict:
                epoch_dict[epoch]+=[(book.author, book.title) for book in self.get_all_books_in_epoch_in_kind(epoch, kind)]
            else:
                epoch_dict[epoch]=[(book.author, book.title) for book in self.get_all_books_in_epoch_in_kind(epoch, kind)]
        books_epoch_list = [epoch_dict[epoch] for epoch in epoch_dict]
        return books_epoch_list

    def fetch(self) -> None:
        books_fetcher = BooksFetcher()
        book_josn_list = books_fetcher.fetch()
        for book_json in book_josn_list:
            author = book_json["author"]
            title = book_json["title"]
            epochs = self._get_epochs_list(book_json["epoch"])
            href = book_json["href"]
            kind = book_json["kind"]
            if self._accepted_book(kind, author):
                self.put_book(
                    Book(
                        title=title, author=author, epochs=epochs, href=href, kind=kind
                    )
                )

    def _accepted_book(self, kind: str, author: str) -> bool:
        return kind in ACCEPTED_KINDS and author in self.polish_author_list

    def _get_epochs_list(self, epochs: str) -> t.List[str]:
        return epochs.split(", ")
