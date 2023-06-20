class Splitter():
    def split_book_for_paragraphs(self, book_text: str):
        paragraphs = book_text.split('\n')
        return [p for p in paragraphs if len(p)>5 and not p.startswith('ISBN')]
    
    def split_book_for_sentences(self, book_text: str):
        ...
        
    def split_book_for_n_elements_sections(self, book_text: str, par_len: int): 
        words = book_text.split()
        sections = [words[i:i+par_len] for i in range(0, len(words) - par_len, par_len)] if len(words)>par_len else [words]
        if len(sections[-1]) < 0.1 * par_len and len(sections) > 1:
            sections = sections[:-1]
        return [' '.join(sec) for sec in sections]
        
    def split_book(self, book_text: str, par_len: int = None):
        if not par_len:           
            return self.split_book_for_paragraphs(book_text)
        else:
            return self.split_book_for_n_elements_sections(book_text, par_len)
           