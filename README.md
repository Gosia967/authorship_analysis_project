# Badanie metod ustalania autorstwa tekstu (kod projektu)
### Autorka: Małgorzata Maciejewska
### O kodzie projektu

Folder 'training_models' zawiera notebooki z testami do każdego rozdziału pracy dotyczącego klasyfikacji. Każda grupa klasyfikatorów jest testowana w dwóch notatnikach: jeden umożliwa testowanie klasyfikatorów na pojednyczym zbiorze danych, tworzenie nowych zbiorów danych i ewentualnie wyjaśnialności, a drugi służy do seryjnego testowania klasyfikatorów i uzyskania wyników (w formacie tabel latexowych i macierzy pomyłek).
Wyniki zapisywane są w folderach 'results' (tabele latexowe) i 'figures' (pliki graficzne).

### Instalacja środowiska i dodatkowe pliki
- Kod wymaga instalacji pythonowych pakietów. W rekomendowanym środowisku udostępnianym przez google drive powinno wystarczyć doinstalowanie pakietów z pliku 'requrements.txt', pełna wersja pakietów znajduje się w pliku 'requirements_google_full_version.txt'.
- Testowanie sieci z embeddingiem GloVe wymaga pobrania pliku w formacie txt https://github.com/sdadas/polish-nlp-resources#glove (w formacie zip 626MB) i umieszczenia go w katalogu training_models/embedding jako glove_100_3_polish.txt.
- Tworzenie własnych plików w formacie ARPA (narzędziem lmplz) wymaga pobrania i zbudowania plików biblioteki kenlm. W katalogu rodzica katalogu 'authorship_analysis' należy uruchomić (instrukcja znajduje się też na stornie https://kheafield.com/code/kenlm/):
```
wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
cd kenlm
mkdir build
cd build
cmake ..
make -j2
```
- Testowanie sieci z embeddingiem GloVe wymaga pobrania pliku w formacie txt https://github.com/sdadas/polish-nlp-resources#glove (w formacie zip 626MB) i umieszczenia go w katalogu training_models/embedding jako glove_100_3_polish.txt.
- Kod pozwala na tworzenie własnych zbiorów danych, składających się z książek innych autorów niż opisanych w tekście pracy, w tym celu należy uruchomić 'data_fetchers\downloader_tasks.py' (pobranie wszystkich książek z Wolnych Lektur do folderu 'book_downloads').
