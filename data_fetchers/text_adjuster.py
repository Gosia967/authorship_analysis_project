import langid

class TextAdjuster:
    FOOTER_STRATSWITH = "Ta lektura, podobnie jak tysiące innych, dostępna jest na stronie wolnelektury.pl."
    FOOTER_DASHES = "-----"
    FORBIDDEN_SIGNS = ['*', '\\', '/', '?', '|', '>', '<', ':', '"']
    
    def cut_footer(self, txt: str):
        if txt:
            txt = txt.partition(self.FOOTER_STRATSWITH)[0]
            cut_dashes = txt.rpartition(self.FOOTER_DASHES)
            if cut_dashes[2] in ['', '\n', '\n\n', '\n\n\n']:
                return cut_dashes[0]
            return txt
        
    def detect_polish(self, txt: str):
        if txt:
            return langid.classify(txt)[0] == 'pl'
    
    def adjust_title_for_filename(self, title):
        for sign in self.FORBIDDEN_SIGNS:
            title = title.replace(sign, '')
        return title

    def remove_komentarz(self, text: str):
        x = list(filter(
            lambda x: x>=0,
            [
                text.find('\nKomentarz'), 
                text.find('\nOmówienie')
            ]))
        if len(x)>0:
            return text[:min(x)]
        return text
