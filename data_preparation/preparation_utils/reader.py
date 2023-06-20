import os
import data_fetchers.text_adjuster as adj

class Reader():
    def __init__(self):
        self.adjuster = adj.TextAdjuster()
    
    def read(self, path_to_read: str) -> str:
        if os.path.exists( path_to_read):
            with open(path_to_read, 'r', encoding="utf-8") as f:
                return self.adjuster.remove_komentarz(f.read())
        return ''           
            
    def read_book_by_author_and_title(self, path: str, author: str, title: str) -> str:
        file_path = self.get_file_path_with_author_title(path, author, title)
        return self.read(file_path)
    
    def get_file_path_with_author_title(self, path: str, author: str, title: str):
        adjusted_title = self.adjuster.adjust_title_for_filename(title)
        return os.path.join(path, author, adjusted_title + '.txt')
