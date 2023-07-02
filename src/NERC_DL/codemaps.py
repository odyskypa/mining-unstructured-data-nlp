
import string
import re

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet')

from dataset import *

class Codemaps :
    # --- constructor, create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(self, data, maxlen=None, suflen=None, DB_drug_list=None, DB_brand_list=None, DB_group_list=None, hsdb_list=None) :

        if isinstance(data,Dataset) and maxlen is not None and suflen is not None:
            self.__create_indexs(data, maxlen, suflen, DB_drug_list, DB_brand_list, DB_group_list, hsdb_list)

        elif type(data) == str and maxlen is None and suflen is None:
            self.__load(data)

        else:
            print('codemaps: Invalid or missing parameters in constructor')
            exit()

            
    # --------- Create indexs from training data
    # Extract all words and labels in given sentences and 
    # create indexes to encode them as numbers when needed
    def __create_indexs(self, data, maxlen, suflen, DB_drug_list, DB_brand_list, DB_group_list, hsdb_list) :

        lemmatizer = WordNetLemmatizer()

        self.maxlen = maxlen
        self.suflen = suflen
        words = set([])
        lc_words = set([])
        sufs = set([])
        sufs3 = set([])
        sufs4 = set([])
        sufs6 = set([])
        pref3 = set([])
        pref4 = set([])
        pref5 = set([])
        pref6 = set([])
        labels = set([])
        pos_tags= set([])
        lemmas = set([])
        caps = set([])
        single_caps = set([])
        punctuations = set([])  # New set to store punctuations
        digits = set([])  # New set to store digits
        drugs = set([])
        brands = set([])
        groups = set([])
        hsdbs = set([])

        for s in data.sentences() :
            for t in s :
                words.add(t['form'])
                sufs.add(t['lc_form'][-self.suflen:])
                sufs3.add(t['lc_form'][-3:])
                sufs4.add(t['lc_form'][-4:])
                sufs6.add(t['lc_form'][-6:])
                pref3.add(t['lc_form'][:3])
                pref4.add(t['lc_form'][:4])
                pref5.add(t['lc_form'][:5])
                pref6.add(t['lc_form'][:6])
                labels.add(t['tag'])
                lc_words.add(t['lc_form'])
                pos_tag = nltk.pos_tag([t['form']])[0][1]

                # Map POS tag to WordNet POS tag
                if pos_tag.startswith('J'):
                    wn_pos = wordnet.ADJ
                elif pos_tag.startswith('V'):
                    wn_pos = wordnet.VERB
                elif pos_tag.startswith('N'):
                    wn_pos = wordnet.NOUN
                elif pos_tag.startswith('R'):
                    wn_pos = wordnet.ADV
                else:
                    wn_pos = wordnet.NOUN  # Noun by default
                pos_tags.add(wn_pos)
                lemma = lemmatizer.lemmatize(t['form'], wn_pos)
                lemmas.add(lemma)

                if re.search(r'[A-Z]{2,}', t['form']):
                    caps.add(t['form'])
                if re.search(r'^[A-Z]$', t['form']):
                    single_caps.add(t['form'])
                if re.search(r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]', t['form']):
                    punctuations.add(t['form'])
                if re.search(r'\d', t['form']):
                    digits.add(t['form'])

                if t['form'] in DB_drug_list:
                    drugs.add(t['form'])
                if t['form'] in DB_brand_list:
                    brands.add(t['form'])
                if t['form'] in DB_group_list:
                    groups.add(t['form'])
                if t['form'] in hsdb_list:
                    hsdbs.add(t['form'])

        self.word_index = {w: i+2 for i,w in enumerate(list(words))}
        self.word_index['PAD'] = 0 # Padding
        self.word_index['UNK'] = 1 # Unknown words

        self.suf_index = {s: i+2 for i,s in enumerate(list(sufs))}
        self.suf_index['PAD'] = 0  # Padding
        self.suf_index['UNK'] = 1  # Unknown suffixes5

        self.suf3_index = {s: i + 2 for i, s in enumerate(list(sufs3))}
        self.suf3_index['PAD'] = 0  # Padding
        self.suf3_index['UNK'] = 1  # Unknown suffixes3

        self.suf4_index = {s: i + 2 for i, s in enumerate(list(sufs4))}
        self.suf4_index['PAD'] = 0  # Padding
        self.suf4_index['UNK'] = 1  # Unknown suffixes4

        self.suf6_index = {s: i + 2 for i, s in enumerate(list(sufs6))}
        self.suf6_index['PAD'] = 0  # Padding
        self.suf6_index['UNK'] = 1  # Unknown suffixes6

        # self.pre3_index = {pre3: i + 2 for i, pre3 in enumerate(list(pref3))}
        # self.pre3_index['PAD'] = 0  # Padding
        # self.pre3_index['UNK'] = 1  # Unknown suffixes
        #
        # self.pre4_index = {pre4: i + 2 for i, pre4 in enumerate(list(pref4))}
        # self.pre4_index['PAD'] = 0  # Padding
        # self.pre4_index['UNK'] = 1  # Unknown suffixes
        #
        # self.pre5_index = {pre5: i + 2 for i, pre5 in enumerate(list(pref5))}
        # self.pre5_index['PAD'] = 0  # Padding
        # self.pre5_index['UNK'] = 1  # Unknown suffixes
        #
        # self.pre6_index = {pre6: i + 2 for i, pre6 in enumerate(list(pref6))}
        # self.pre6_index['PAD'] = 0  # Padding
        # self.pre6_index['UNK'] = 1  # Unknown suffixes

        self.lc_index = {lc: i + 2 for i, lc in enumerate(list(lc_words))}
        self.lc_index['PAD'] = 0  # Padding
        self.lc_index['UNK'] = 1  # Unknown lowercase

        self.pos_index = {pos: i + 2 for i, pos in enumerate(list(pos_tags))}
        self.pos_index['PAD'] = 0  # Padding
        self.pos_index['UNK'] = 1  # Unknown pos

        self.lemmas_index = {lemma: i + 2 for i, lemma in enumerate(list(lemmas))}
        self.lemmas_index['PAD'] = 0  # Padding
        self.lemmas_index['UNK'] = 1  # Unknown lemmas

        self.caps_index = {cap: i + 2 for i, cap in enumerate(list(caps))}
        self.caps_index['PAD'] = 0  # Padding
        self.caps_index['UNK'] = 1  # Unknown capital letter patterns

        self.single_caps_index = {cap: i + 2 for i, cap in enumerate(list(single_caps))}
        self.single_caps_index['PAD'] = 0  # Padding
        self.single_caps_index['UNK'] = 1  # Unknown single capital letter patterns

        self.punctuations_index = {punc: i + 2 for i, punc in enumerate(list(punctuations))}
        self.punctuations_index['PAD'] = 0  # Padding
        self.punctuations_index['UNK'] = 1  # Unknown punctuations

        self.digits_index = {digit: i + 2 for i, digit in enumerate(list(digits))}
        self.digits_index['PAD'] = 0  # Padding
        self.digits_index['UNK'] = 1  # Unknown digits

        self.drugs_index = {drug: i + 2 for i, drug in enumerate(list(drugs))}
        self.drugs_index['PAD'] = 0  # Padding
        self.drugs_index['UNK'] = 1  # Unknown digits

        self.brands_index = {brand: i + 2 for i, brand in enumerate(list(brands))}
        self.brands_index['PAD'] = 0  # Padding
        self.brands_index['UNK'] = 1  # Unknown digits

        self.groups_index = {group: i + 2 for i, group in enumerate(list(groups))}
        self.groups_index['PAD'] = 0  # Padding
        self.groups_index['UNK'] = 1  # Unknown digits

        self.hsdbs_index = {hsdb: i + 2 for i, hsdb in enumerate(list(hsdbs))}
        self.hsdbs_index['PAD'] = 0  # Padding
        self.hsdbs_index['UNK'] = 1  # Unknown digits

        self.label_index = {t: i+1 for i,t in enumerate(list(labels))}
        self.label_index['PAD'] = 0 # Padding
        
    ## --------- load indexs ----------- 
    def __load(self, name) : 
        self.maxlen = 0
        self.suflen = 0
        self.word_index = {}
        self.suf_index = {}
        self.label_index = {}

        with open(name+".idx") as f :
            for line in f.readlines(): 
                (t,k,i) = line.split()
                if t == 'MAXLEN' : self.maxlen = int(k)
                elif t == 'SUFLEN' : self.suflen = int(k)                
                elif t == 'WORD': self.word_index[k] = int(i)
                elif t == 'SUF': self.suf_index[k] = int(i)
                elif t == 'LABEL': self.label_index[k] = int(i)
                            
    
    ## ---------- Save model and indexs ---------------
    def save(self, name) :
        # save indexes
        with open(name+".idx","w") as f :
            print ('MAXLEN', self.maxlen, "-", file=f)
            print ('SUFLEN', self.suflen, "-", file=f)
            for key in self.label_index : print('LABEL', key, self.label_index[key], file=f)
            for key in self.word_index : print('WORD', key, self.word_index[key], file=f)
            for key in self.suf_index : print('SUF', key, self.suf_index[key], file=f)


    ## --------- encode X from given data ----------- 
    def encode_words(self, data) :

        lemmatizer = WordNetLemmatizer()
        # encode and pad sentence words
        Xw = [[self.word_index[w['form']] if w['form'] in self.word_index else self.word_index['UNK'] for w in s] for s in data.sentences()]
        Xw = pad_sequences(maxlen=self.maxlen, sequences=Xw, padding="post", value=self.word_index['PAD'])

        # encode and pad suffixes
        Xs = [[self.suf_index[w['lc_form'][-self.suflen:]] if w['lc_form'][-self.suflen:] in self.suf_index else self.suf_index['UNK'] for w in s] for s in data.sentences()]
        Xs = pad_sequences(maxlen=self.maxlen, sequences=Xs, padding="post", value=self.suf_index['PAD'])

        Xs3 = [[self.suf3_index[w['lc_form'][-3:]] if w['lc_form'][-3:] in self.suf3_index else
               self.suf3_index['UNK'] for w in s] for s in data.sentences()]
        Xs3 = pad_sequences(maxlen=self.maxlen, sequences=Xs3, padding="post", value=self.suf3_index['PAD'])

        Xs4 = [[self.suf4_index[w['lc_form'][-4:]] if w['lc_form'][-4:] in self.suf4_index else
               self.suf4_index['UNK'] for w in s] for s in data.sentences()]
        Xs4 = pad_sequences(maxlen=self.maxlen, sequences=Xs4, padding="post", value=self.suf4_index['PAD'])

        Xs6 = [[self.suf6_index[w['lc_form'][-6:]] if w['lc_form'][-6:] in self.suf6_index else
               self.suf6_index['UNK'] for w in s] for s in data.sentences()]
        Xs6 = pad_sequences(maxlen=self.maxlen, sequences=Xs6, padding="post", value=self.suf6_index['PAD'])

        # encode and pad prefixes
        # Xp3 = [[self.pre3_index[w['lc_form'][:3]] if w['lc_form'][:3] in self.pre3_index else
        #        self.pre3_index['UNK'] for w in s] for s in data.sentences()]
        # Xp3 = pad_sequences(maxlen=self.maxlen, sequences=Xp3, padding="post", value=self.pre3_index['PAD'])
        #
        # Xp4 = [[self.pre4_index[w['lc_form'][:4]] if w['lc_form'][:4] in self.pre4_index else
        #         self.pre4_index['UNK'] for w in s] for s in data.sentences()]
        # Xp4 = pad_sequences(maxlen=self.maxlen, sequences=Xp4, padding="post", value=self.pre4_index['PAD'])
        #
        # Xp5 = [[self.pre5_index[w['lc_form'][:5]] if w['lc_form'][:5] in self.pre5_index else
        #         self.pre5_index['UNK'] for w in s] for s in data.sentences()]
        # Xp5 = pad_sequences(maxlen=self.maxlen, sequences=Xp5, padding="post", value=self.pre5_index['PAD'])
        #
        # Xp6 = [[self.pre6_index[w['lc_form'][:6]] if w['lc_form'][:6] in self.pre6_index else
        #         self.pre6_index['UNK'] for w in s] for s in data.sentences()]
        # Xp6 = pad_sequences(maxlen=self.maxlen, sequences=Xp6, padding="post", value=self.pre6_index['PAD'])

        # encode and pad lowercase
        Xlc = [[self.lc_index[w['lc_form']] if w['lc_form'] in self.lc_index else
               self.lc_index['UNK'] for w in s] for s in data.sentences()]
        Xlc = pad_sequences(maxlen=self.maxlen, sequences=Xlc, padding="post", value=self.lc_index['PAD'])

        # encode and pad caps
        Xcs = [[self.caps_index[w['form']] if w['form'] in self.caps_index else self.caps_index['UNK']
               for w in s] for s in data.sentences()]
        Xcs = pad_sequences(maxlen=self.maxlen, sequences=Xcs, padding="post", value=self.caps_index['PAD'])

        Xsc = [[self.single_caps_index[w['form']] if w['form'] in self.single_caps_index else self.single_caps_index['UNK']
              for w in s] for s in data.sentences()]
        Xsc = pad_sequences(maxlen=self.maxlen, sequences=Xsc, padding="post", value=self.single_caps_index['PAD'])

        Xpu = [[self.punctuations_index[w['form']] if w['form'] in self.punctuations_index else self.punctuations_index[
            'UNK']
                for w in s] for s in data.sentences()]
        Xpu = pad_sequences(maxlen=self.maxlen, sequences=Xpu, padding="post", value=self.punctuations_index['PAD'])

        Xd = [[self.digits_index[w['form']] if w['form'] in self.digits_index else self.digits_index['UNK']
               for w in s] for s in data.sentences()]
        Xd = pad_sequences(maxlen=self.maxlen, sequences=Xd, padding="post", value=self.digits_index['PAD'])

        Xdr = [[self.drugs_index[w['form']] if w['form'] in self.drugs_index else self.drugs_index['UNK']
               for w in s] for s in data.sentences()]
        Xdr = pad_sequences(maxlen=self.maxlen, sequences=Xdr, padding="post", value=self.drugs_index['PAD'])

        Xgs = [[self.groups_index[w['form']] if w['form'] in self.groups_index else self.groups_index['UNK']
               for w in s] for s in data.sentences()]
        Xgs = pad_sequences(maxlen=self.maxlen, sequences=Xgs, padding="post", value=self.groups_index['PAD'])

        Xbs = [[self.brands_index[w['form']] if w['form'] in self.brands_index else self.brands_index['UNK']
               for w in s] for s in data.sentences()]
        Xbs = pad_sequences(maxlen=self.maxlen, sequences=Xbs, padding="post", value=self.brands_index['PAD'])

        Xhsdb = [[self.hsdbs_index[w['form']] if w['form'] in self.hsdbs_index else self.hsdbs_index['UNK']
               for w in s] for s in data.sentences()]
        Xhsdb = pad_sequences(maxlen=self.maxlen, sequences=Xhsdb, padding="post", value=self.hsdbs_index['PAD'])

        # encode and pad pos tags and lemmas
        Xlem = []
        Xpos = []
        for s in data.sentences():
            sentence_tags = []
            sentence_lemmas = []
            for w in s:
                pos_tag = nltk.pos_tag([w['form']])[0][1]
                if pos_tag.startswith('J'):
                    wn_pos = wordnet.ADJ
                elif pos_tag.startswith('V'):
                    wn_pos = wordnet.VERB
                elif pos_tag.startswith('N'):
                    wn_pos = wordnet.NOUN
                elif pos_tag.startswith('R'):
                    wn_pos = wordnet.ADV
                else:
                    wn_pos = wordnet.NOUN  # Noun by default

                lemma = lemmatizer.lemmatize(w['form'], wn_pos)

                if wn_pos in self.pos_index:
                    sentence_tags.append(self.pos_index[wn_pos])
                else:
                    sentence_tags.append(self.pos_index['UNK'])

                if lemma in self.lemmas_index:
                    sentence_lemmas.append(self.lemmas_index[lemma])
                else:
                    sentence_lemmas.append(self.lemmas_index['UNK'])

            Xpos.append(sentence_tags)
            Xlem.append(sentence_lemmas)

        Xpos = pad_sequences(maxlen=self.maxlen, sequences=Xpos, padding="post", value=self.pos_index['PAD'])
        # Xlem = pad_sequences(maxlen=self.maxlen, sequences=Xlem, padding="post", value=self.lemmas_index['PAD'])

        # return encoded sequences
        #return [Xw,Xs,Xlc,Xs3,Xs4,Xs6,Xpos,Xlem, Xp3,Xp4,Xp5,Xp6, Xcs, Xsc, Xpu, Xd, Xdr, Xbs, Xgs, Xhsdb]
        return [Xw,Xs,Xlc,Xs3,Xs4,Xs6,Xpos, Xcs, Xsc, Xpu, Xd, Xdr, Xbs, Xgs, Xhsdb]
        # return [Xw,Xs,Xlc,Xs3,Xs4,Xs6,Xpos,Xlem, Xp3,Xp4,Xp5,Xp6, Xcs, Xsc, Xpu, Xd]

    
    ## --------- encode Y from given data ----------- 
    def encode_labels(self, data) :
        # encode and pad sentence labels 
        Y = [[self.label_index[w['tag']] for w in s] for s in data.sentences()]
        Y = pad_sequences(maxlen=self.maxlen, sequences=Y, padding="post", value=self.label_index["PAD"])
        return np.array(Y)

    ## -------- get word index size ---------
    def get_n_words(self) :
        return len(self.word_index)
    ## -------- get suf index size ---------
    def get_n_sufs(self) :
        return len(self.suf_index)

    def get_n_sufs3(self) :
        return len(self.suf3_index)

    def get_n_sufs4(self) :
        return len(self.suf4_index)

    def get_n_sufs6(self) :
        return len(self.suf6_index)

    ## -------- get pref index size ---------
    def get_n_prefs3(self) :
        return len(self.pre3_index)

    def get_n_prefs4(self) :
        return len(self.pre4_index)

    def get_n_prefs5(self) :
        return len(self.pre5_index)

    def get_n_prefs6(self) :
        return len(self.pre6_index)

    def get_n_caps(self) :
        return len(self.caps_index)

    def get_n_signle_caps(self) :
        return len(self.single_caps_index)

    def get_n_punctuation(self) :
        return len(self.punctuations_index)

    def get_n_digits(self) :
        return len(self.digits_index)

    def get_n_drugs(self) :
        return len(self.drugs_index)

    def get_n_brands(self) :
        return len(self.brands_index)

    def get_n_groups(self) :
        return len(self.groups_index)

    def get_n_hsdbs(self) :
        return len(self.hsdbs_index)

    ## -------- get label index size ---------
    def get_n_labels(self) :
        return len(self.label_index)

    ## -------- get index for given word ---------
    def word2idx(self, w) :
        return self.word_index[w]
    ## -------- get index for given suffix --------
    def suff2idx(self, s) :
        return self.suff_index[s]
    ## -------- get index for given label --------
    def label2idx(self, l) :
        return self.label_index[l]
    ## -------- get label name for given index --------
    def idx2label(self, i) :
        for l in self.label_index :
            if self.label_index[l] == i:
                return l
        raise KeyError

    ## -------- get lc word index size ---------
    def get_lc_n_words(self):
        return len(self.lc_index)

    ## -------- get pos index size ---------
    def get_n_pos(self):
        return len(self.pos_index)

    ## -------- get pos index size ---------
    def get_n_lemmas(self):
        return len(self.lemmas_index)

