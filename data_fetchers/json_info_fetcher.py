import requests


class JSONInfoFetcher:
    def __init__(self, url: str) -> None:
        self.url = url
        self.json_data = None

    def fetch(self) -> list:
        response = requests.get(self.url)
        self.data = response.json()
        return self.data

    def get_list(self, key: str):
        if self.data:
            return [item[key] for item in self.data]
        return []


class AuthorsFetcher(JSONInfoFetcher):
    def __init__(self) -> None:
        self.url = "https://wolnelektury.pl/api/authors/"

    def get_names(self):
        return self.get_list("name")


class EpochsFetcher(JSONInfoFetcher):
    def __init__(self) -> None:
        self.url = "https://wolnelektury.pl/api/epochs/"

    def get_names(self):
        return self.get_list("name")


class BooksFetcher(JSONInfoFetcher):
    def __init__(self) -> None:
        self.url = "https://wolnelektury.pl/api/books/"
