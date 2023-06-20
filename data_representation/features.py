import re
import inspect


class FeaturesDict:
    def __init__(self, nlp):
        self.nlp = nlp
    
    def get(self, corpus):
        return [self._get_vector(str(text)) for text in corpus]
        
    def _get_vector(self, text: str):
        fg = FeaturesGetter(text, self.nlp)
        attrs = (getattr(fg, name) for name in dir(fg))
        methods = filter(inspect.ismethod, attrs)
        return {method.__name__:method() for method in methods if not method.__name__.startswith('__')}
        

class FeaturesGetter:

    def __init__(self, text: str, nlp) -> None:
        self.text = text
        fgu = FeaturesGetterUtils(text, nlp)
        self.sentences = fgu.split_text_for_sentences()
        self.pos = fgu.get_pos()
        self.lemma = fgu.get_lemma()
    
    def chars_num(self) -> int:
        return len(self.text)
    
    def words_num(self) -> int:       
        return len(self.text.split()) 
    
    def avg_chars_in_word(self) -> float: 
        return self.chars_num()/self.words_num()
    
    def sentences_num(self) -> int:    
        return len(self.sentences)
    
    def avg_words_in_sen(self) -> float: 
        return self.words_num()/self.sentences_num()
    
    def avg_chars_in_sen(self) -> float: 
        return self.chars_num()/self.sentences_num()
    
    def commas_num(self) -> int:
        return len(re.findall('[,]', self.text))
    
    def punct_num(self) ->int:      
        return len(self.pos['PUNCT']) if 'PUNCT' in self.pos else 0
    
    def avg_commas_in_sen(self) -> float: 
        return self.commas_num()/self.sentences_num()
    
    def avg_punct_in_sen(self) -> float: 
        return self.punct_num()/self.sentences_num()
      
    def different_words_ratio(self) -> float:
        return len(set(self.lemma))/max(len(self.lemma), 1)
    
    def adj_num(self) -> int:
        return len(self.pos['ADJ']) if 'ADJ' in self.pos else 0
        
    def verb_num(self) -> int:
        return len(self.pos['VERB']) if 'VERB' in self.pos else 0
        
    def noun_num(self) -> int:
        return len(self.pos['NOUN']) if 'NOUN' in self.pos else 0
        
    def conj_num(self) -> int:
        num = len(self.pos['CCONJ']) if 'CCONJ' in self.pos else 0
        num += len(self.pos['SCONJ']) if 'SCONJ' in self.pos else 0
        return num 
        
    def adv_num(self) -> int:
        return len(self.pos['ADV']) if 'ADV' in self.pos else 0
    
    def pron_num(self) -> int:
        return len(self.pos['PRON']) if 'PRON' in self.pos else 0
        
    def proper_noun_num(self) -> int:
        return len(self.pos['PROPN']) if 'PROPN' in self.pos else 0
    
    def adj_noun_ratio(self) -> float:
        return self.adj_num()/max(self.noun_num(), 1)  
        
    def propn_noun_ratio(self) -> float:
        return self.proper_noun_num()/max(self.noun_num(), 1) 
        
    def pron_noun_ratio(self) -> float:
        return self.pron_num()/max(self.noun_num(), 1) 
        
    def noun_verb_ratio(self) -> float:
        return self.noun_num()/max(self.verb_num(), 1) 
        
    def adv_verb_ratio(self) -> float:
        return self.adv_num()/max(self.verb_num(), 1) 
        
    def avg_adj_in_sen(self) -> float:
        return self.adj_num()/max(self.sentences_num(), 1) 
    
    def avg_noun_in_sen(self) -> float:
        return self.noun_num()/max(self.sentences_num(), 1) 
        
    def avg_verb_in_sen(self) -> float:
        return self.verb_num()/max(self.sentences_num(), 1) 
        
    def avg_proper_noun_in_sen(self) -> float:
        return self.proper_noun_num()/max(self.sentences_num(), 1) 
    
    
class FeaturesGetterUtils:
    
    def __init__(self, text: str, nlp):
        self.text = text
        self.nlp = nlp
        self.doc = self.nlp(text)
    
    def split_text_for_sentences(self) -> list:
        separators_re = '[!] [A-Z]|[?] [A-Z]|[.] [A-Z]|[.]\n|[?]\n|[!]\n'
        sep_list = re.findall(separators_re, self.text)
        sentences_list = re.split(separators_re, self.text)
        for i in range(len(sentences_list)-1):
            sentences_list[i]+=sep_list[i][0]
            if sep_list[i][1] == ' ':
                sentences_list[i+1]=sep_list[i][2]+sentences_list[i+1]   
        return sentences_list if len(sentences_list[-1])>0 else sentences_list[:-1]
    
    def get_pos(self) -> dict:
        pos_dict = {}
        for token in self.doc:
            if token.pos_ in pos_dict:
                pos_dict[token.pos_].append(token)
            else:
                pos_dict[token.pos_] = [token]
        return pos_dict
    
    def get_lemma(self):
        return [token.lemma_ for token in self.doc if token.pos_ != 'PUNCT']
    