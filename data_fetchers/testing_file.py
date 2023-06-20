from data_fetchers.json_info_fetcher import JSONInfoFetcher


def test_funtion():
    url = "https://wolnelektury.pl/api/authors/"
    fetcher = JSONInfoFetcher(url)
    data = fetcher.fetch()
    print(len(data))
    print(fetcher.get_dict())


test_funtion()
