from data_fetchers.json_info_fetcher import AuthorsFetcher, EpochsFetcher
from data_fetchers.books import BookSet, ACCEPTED_KINDS


def get_authors_txt():
    path = "authors.txt"
    fetcher = AuthorsFetcher()

    f = open(path, "a", encoding="utf-8")
    fetcher.fetch()
    names_list = fetcher.get_names()
    for name in names_list:
        print(name)
        f.write(name + "\n")
    f.close()


def list_num_of_books_by_polish_authors_in_epochs():
    bookset = BookSet()
    epochs_fetcher = EpochsFetcher()
    epochs_fetcher.fetch()
    epochs_names_list = epochs_fetcher.get_names()
    bookset.fetch()
    epochs_authors_dict = {epoch: dict() for epoch in epochs_names_list}
    for book in bookset.shelf:
        for epoch in book.epochs:
            epoch_dict = epochs_authors_dict[epoch]
            if book.author in epoch_dict:
                epoch_dict[book.author] += 1
            else:
                epoch_dict[book.author] = 1
    return epochs_authors_dict


def beautiful_sorted_print(epochs_authors_dict):
    for epoch in epochs_authors_dict:
        sorted_dict = dict(
            sorted(epochs_authors_dict[epoch].items(), key=lambda item: item[1])
        )
        print(epoch)
        for author in sorted_dict:
            if sorted_dict[author] > 1:
                print(f"    {author} : {sorted_dict[author]}")


def list_num_of_books_by_polish_authors_in_epochs_kind():
    bookset = BookSet()
    epochs_fetcher = EpochsFetcher()
    epochs_fetcher.fetch()
    epochs_names_list = epochs_fetcher.get_names()
    bookset.fetch()
    kind_epochs_authors_dict = {
        kind: {epoch: dict() for epoch in epochs_names_list} for kind in ACCEPTED_KINDS
    }
    for kind in ACCEPTED_KINDS:
        for identifier in bookset.shelf:
            book = bookset.shelf[identifier]
            if book.kind == kind:
                for epoch in book.epochs:
                    epoch_dict = kind_epochs_authors_dict[kind][epoch]
                    if book.author in epoch_dict:
                        epoch_dict[book.author] += 1
                    else:
                        epoch_dict[book.author] = 1
    return kind_epochs_authors_dict

def list_books_by_polish_authors_in_epochs_kind():
    bookset = BookSet()
    epochs_fetcher = EpochsFetcher()
    epochs_fetcher.fetch()
    epochs_names_list = epochs_fetcher.get_names()
    bookset.fetch()
    kind_epochs_authors_dict = {
        kind: {epoch: dict() for epoch in epochs_names_list} for kind in ACCEPTED_KINDS
    }
    for kind in ACCEPTED_KINDS:
        for identifier in bookset.shelf:
            book = bookset.shelf[identifier]
            if book.kind == kind:
                for epoch in book.epochs:
                    epoch_dict = kind_epochs_authors_dict[kind][epoch]
                    if book.author in epoch_dict:
                        epoch_dict[book.author].append(book.title)
                    else:
                        epoch_dict[book.author] = [book.title]
    return kind_epochs_authors_dict


def beautiful_sorted_kind_print(kind_epochs_authors_dict):
    for kind in kind_epochs_authors_dict:
        print(kind)
        for epoch in kind_epochs_authors_dict[kind]:
            sorted_dict = dict(
                sorted(
                    kind_epochs_authors_dict[kind][epoch].items(),
                    key=lambda item: item[1],
                )
            )
            print("    " + epoch)
            for author in sorted_dict:
                if sorted_dict[author] > 1:
                    print(f"        {author} : {sorted_dict[author]}")


