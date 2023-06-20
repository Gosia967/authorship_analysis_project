import pathlib
import os


def get_polish_authors_list():
    path = str(pathlib.Path(__file__).parent.resolve())
    path = os.path.join(path, 'polish_authors.txt')
    f = open(path, "r", encoding="utf-8")
    authors_list = [author[:-1] for author in f.readlines()]
    f.close()
    return authors_list
