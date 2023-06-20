class Preprocessing():
    def __init__(self, text:str, nlp):
        self.text = text
        self.nlp = nlp
        
    # descr: p
    def remove_punctuation(self):
        t = ''
        doc = self.nlp(self.text)
        for token in doc:
            if token.pos_ != 'PUNCT':
                t += token.text_with_ws
            else:
                t += ' '
        self.text = t
            
    # descr: s    
    def remove_stop_words(self):
        t = ''
        doc = self.nlp(self.text)
        for token in doc:
            if token.text not in self.nlp.Defaults.stop_words:
                t += token.text_with_ws
        self.text = t
        
    # descr: m
    def lemmatize_text(self):
        t = ''
        doc = self.nlp(self.text)
        for token in doc:
            t += token.lemma_ + ' '
        self.text = t
        
    # descr: o
    # POS tags: https://universaldependencies.org/u/pos/
    def POS_leave_only(self, pos):
        t = ''
        doc = self.nlp(self.text)
        for token in doc:
            if token.pos_ in pos or token.pos_ in ['PUNCT', 'SPACE']:
                t += token.text_with_ws
        self.text = t
        
    # descr: l
    def lower_text(self):
        self.text = self.text.lower()
    
    # descr: a
    def anonymize(self):
        doc = self.nlp(self.text)
        t = ''
        prev_end = 0
        for ent in doc.ents:
            t += self.text[prev_end:ent.start_char]+ ent.label_
            prev_end = ent.end_char
        t += self.text[prev_end:]
        self.text = t
