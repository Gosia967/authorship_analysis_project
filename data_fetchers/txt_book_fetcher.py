import requests
from data_fetchers.books import Book


class TXTBookFetcher:
    def __init__(self, book: Book) -> None:
        self.book = book
        self.details = None

    def _fetch_details(self) -> None:
        response = requests.get(self.book.href)
        self.details = response.json()

    def fetch_txt(self):
        self._fetch_details()
        txt_url = self.details["txt"]
        if txt_url:
            response = requests.get(txt_url)
            return response.text
