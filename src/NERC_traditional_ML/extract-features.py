#! /usr/bin/python3

import sys
import re
from os import listdir
import time

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import nltk

nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

## --------- tokenize sentence -----------
## -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    ## word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(txt):
        ## keep track of the position where each token should appear, and
        ## store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset + len(t) - 1))
        offset += len(t)

    ## tks is a list of triples (word,start,end)
    return tks


## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans):
    (form, start, end) = token
    for (spanS, spanE, spanT) in spans:
        if start == spanS and end <= spanE:
            return "B-" + spanT
        elif start >= spanS and end <= spanE:
            return "I-" + spanT

    return "O"


## --------- Feature extractor -----------
## -- Extract features for each token in given sentence

def extract_features(tokens):
    # for each token, generate list of features and add it to the result
    result = []
    for k in range(0, len(tokens)):
        tokenFeatures = [];
        t = tokens[k][0]

        tokenFeatures.append("form=" + t)
        tokenFeatures.append("suf3=" + t[-3:])

        if k > 0:
            tPrev = tokens[k - 1][0]
            tokenFeatures.append("formPrev=" + tPrev)
            tokenFeatures.append("suf3Prev=" + tPrev[-3:])
        else:
            tokenFeatures.append("BoS")

        if k < len(tokens) - 1:
            tNext = tokens[k + 1][0]
            tokenFeatures.append("formNext=" + tNext)
            tokenFeatures.append("suf3Next=" + tNext[-3:])
        else:
            tokenFeatures.append("EoS")

        result.append(tokenFeatures)

    return result

def ext_drug_bank(drug_bank_dir):

    with open(drug_bank_dir, 'r', encoding="utf8") as f:
        data = f.read()

        entries = data.split('\n')

        DB_drug_list = []
        DB_brand_list = []
        DB_group_list = []

        # Loop through each entry
        for n, entry in enumerate(entries):
            if n + 1 == len(entries):
                continue

            item_type = entry.split('|')[1]
            item = entry.split('|')[0]
            if item_type == 'drug':
                DB_drug_list.append(item)
            elif item_type == 'brand':
                DB_brand_list.append(item)
            elif item_type == 'group':
                DB_group_list.append(item)

    return DB_drug_list, DB_brand_list, DB_group_list

def ext_HSDB(hsdb_dir):

    with open(hsdb_dir, 'r', encoding="utf8") as f:
        data = f.read()
        hsdb_list = data.split('\n')

    return hsdb_list

def extract_features1(tokens, DB_drug_list, DB_brand_list, DB_group_list, hsdb_list):
    result = []
    # Initialize the WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()

    for k in range(len(tokens)):
        tokenFeatures = []
        t = tokens[k][0]

        tokenFeatures.append("form=" + t)
        tokenFeatures.append("lower=" + t.lower())
        tokenFeatures.append("length=" + str(len(t)))


        if t in DB_drug_list:
            tokenFeatures.append("DB_drug")
        if t in DB_brand_list:
            tokenFeatures.append("DB_brand")
        if t in DB_group_list:
            tokenFeatures.append("DB_group")
        if t in hsdb_list:
            tokenFeatures.append("hsdb")

        if k == 0:
            tokenFeatures.append("BoS")
        elif k == len(tokens) - 1:
            tokenFeatures.append("EoS")

        tokenFeatures.append("suf3=" + t[-3:])
        tokenFeatures.append("suf4=" + t[-4:])
        tokenFeatures.append("suf5=" + t[-5:])
        tokenFeatures.append("suf6=" + t[-6:])

        # add the string "OC" if there is one capital letter in the word
        # or add the string "MTOC" if there are more than one capital letters in the word
        if sum(1 for c in t if c.isupper()) == 1:
            tokenFeatures.append("OC")
        elif sum(1 for c in t if c.isupper()) > 1:
            tokenFeatures.append("MTOC")

        # 5. add the string "POSC" if there is presence of special characters in the word,
        # like numbers, dashes, pantuaction marks, etc.
        if any(c in string.punctuation or c.isdigit() for c in t):
            tokenFeatures.append("POSC")

        tokenFeatures.append("pre3=" + t[:3])
        tokenFeatures.append("pre4=" + t[:4])
        tokenFeatures.append("pre5=" + t[:5])
        tokenFeatures.append("pre6=" + t[:6])

        # part of speach (POS) tag of the word
        pos_tag = nltk.pos_tag([t])[0][1]
        tokenFeatures.append("POS=" + pos_tag)

        pos = pos_tag[0].lower() if pos_tag and pos_tag[0].lower() in {'n', 'v', 'a', 'r', 's'} else 'n'
        lemma = lemmatizer.lemmatize(t, pos)
        tokenFeatures.append("lemma=" + lemma)

        if k > 1:  # We are after the second word of the sentence,
            # take the same features of the previous twp words
            for x in [1, 2]:
                t_prev = tokens[k - x][0]
                if x == 1:
                    annot = "prev"
                else:
                    annot = "prev_2"

                tokenFeatures.append(f"form_{annot}=" + t_prev)
                tokenFeatures.append(f"lower_{annot}=" + t_prev.lower())
                tokenFeatures.append(f"length_{annot}=" + str(len(t_prev)))

                if t_prev in DB_drug_list:
                    tokenFeatures.append(f"DB_drug_{annot}")
                if t_prev in DB_brand_list:
                    tokenFeatures.append(f"DB_brand_{annot}")
                if t_prev in DB_group_list:
                    tokenFeatures.append(f"DB_group_{annot}")
                if t_prev in hsdb_list:
                    tokenFeatures.append(f"hsdb_{annot}")

                tokenFeatures.append(f"suf3_{annot}=" + t_prev[-3:])
                tokenFeatures.append(f"suf4_{annot}=" + t_prev[-4:])
                tokenFeatures.append(f"suf5_{annot}=" + t_prev[-5:])
                tokenFeatures.append(f"suf6_{annot}=" + t_prev[-6:])

                if sum(1 for c in t_prev if c.isupper()) == 1:
                    tokenFeatures.append(f"OC_{annot}")
                elif sum(1 for c in t_prev if c.isupper()) > 1:
                    tokenFeatures.append(f"MTOC_{annot}")
                if any(c in string.punctuation or c.isdigit() for c in t_prev):
                    tokenFeatures.append(f"POSC_{annot}")

                tokenFeatures.append(f"pre3_{annot}=" + t_prev[:3])
                tokenFeatures.append(f"pre4_{annot}=" + t_prev[:4])
                tokenFeatures.append(f"pre5_{annot}=" + t_prev[:5])
                tokenFeatures.append(f"pre6_{annot}=" + t_prev[:6])

                pos_tag = nltk.pos_tag([t_prev])[0][1]
                tokenFeatures.append(f"POS_{annot}=" + pos_tag)

                pos = pos_tag[0].lower() if pos_tag and pos_tag[0].lower() in {'n', 'v', 'a', 'r', 's'} else 'n'

                lemma_prev = lemmatizer.lemmatize(t_prev, pos)
                tokenFeatures.append(f"lemma_{annot}=" + lemma_prev)

        elif k == 1:
            t_prev = tokens[k - 1][0]

            tokenFeatures.append("form_prev=" + t_prev)
            tokenFeatures.append("lower_prev=" + t_prev.lower())
            tokenFeatures.append("length_prev=" + str(len(t_prev)))

            if t_prev in DB_drug_list:
                tokenFeatures.append(f"DB_drug_prev")
            if t_prev in DB_brand_list:
                tokenFeatures.append(f"DB_brand_prev")
            if t_prev in DB_group_list:
                tokenFeatures.append(f"DB_group_prev")
            if t_prev in hsdb_list:
                tokenFeatures.append(f"hsdb_prev")

            tokenFeatures.append("suf3_prev=" + t_prev[-3:])
            tokenFeatures.append("suf4_prev=" + t_prev[-4:])
            tokenFeatures.append("suf5_prev=" + t_prev[-5:])
            tokenFeatures.append("suf6_prev=" + t_prev[-6:])

            if sum(1 for c in t_prev if c.isupper()) == 1:
                tokenFeatures.append("OC_prev")
            elif sum(1 for c in t_prev if c.isupper()) > 1:
                tokenFeatures.append("MTOC_prev")

            if any(c in string.punctuation or c.isdigit() for c in t_prev):
                tokenFeatures.append("POSC_prev")

            tokenFeatures.append("pre3_prev=" + t_prev[:3])
            tokenFeatures.append("pre4_prev=" + t_prev[:4])
            tokenFeatures.append("pre5_prev=" + t_prev[:5])
            tokenFeatures.append("pre6_prev=" + t_prev[:6])

            pos_tag = nltk.pos_tag([t_prev])[0][1]
            tokenFeatures.append("POS_prev=" + pos_tag)

            pos = pos_tag[0].lower() if pos_tag and pos_tag[0].lower() in {'n', 'v', 'a', 'r', 's'} else 'n'

            lemma_prev = lemmatizer.lemmatize(t_prev, pos)
            tokenFeatures.append("lemma_prev=" + lemma_prev)

        if k < len(tokens) - 2:  # We are before the second last word of the sentence,
            # take the same features of the next two words
            for x in [1, 2]:

                t_next = tokens[k + x][0]
                if x == 1:
                    annot = "next"
                else:
                    annot = "next_2"

                tokenFeatures.append(f"form_{annot}=" + t_next)
                tokenFeatures.append(f"lower_{annot}=" + t_next.lower())
                tokenFeatures.append(f"length_{annot}=" + str(len(t_next)))

                if t_next in DB_drug_list:
                    tokenFeatures.append(f"DB_drug_{annot}")
                if t_next in DB_brand_list:
                    tokenFeatures.append(f"DB_brand_{annot}")
                if t_next in DB_group_list:
                    tokenFeatures.append(f"DB_group_{annot}")
                if t_next in hsdb_list:
                    tokenFeatures.append(f"hsdb_{annot}")

                tokenFeatures.append(f"suf3_{annot}=" + t_next[-3:])
                tokenFeatures.append(f"suf4_{annot}=" + t_next[-4:])
                tokenFeatures.append(f"suf5_{annot}=" + t_next[-5:])
                tokenFeatures.append(f"suf6_{annot}=" + t_next[-6:])

                if sum(1 for c in t_next if c.isupper()) == 1:
                    tokenFeatures.append(f"OC_{annot}")
                elif sum(1 for c in t_next if c.isupper()) > 1:
                    tokenFeatures.append(f"MTOC_{annot}")

                if any(c in string.punctuation or c.isdigit() for c in t_next):
                    tokenFeatures.append(f"POSC_{annot}")

                tokenFeatures.append(f"pre3_{annot}=" + t_next[:3])
                tokenFeatures.append(f"pre4_{annot}=" + t_next[:4])
                tokenFeatures.append(f"pre5_{annot}=" + t_next[:5])
                tokenFeatures.append(f"pre6_{annot}=" + t_next[:6])

                pos_tag = nltk.pos_tag([t_next])[0][1]
                tokenFeatures.append(f"POS_{annot}=" + pos_tag)

                pos = pos_tag[0].lower() if pos_tag and pos_tag[0].lower() in {'n', 'v', 'a', 'r', 's'} else 'n'
                lemma_next = lemmatizer.lemmatize(t_next, pos)
                tokenFeatures.append(f"lemma_{annot}=" + lemma_next)

        elif k == len(tokens) - 2:

            t_next = tokens[k + 1][0]

            tokenFeatures.append("form_next=" + t_next)
            tokenFeatures.append("lower_next=" + t_next.lower())
            tokenFeatures.append("length_next=" + str(len(t_next)))

            if t_next in DB_drug_list:
                tokenFeatures.append(f"DB_drug_next")
            if t_next in DB_brand_list:
                tokenFeatures.append(f"DB_brand_next")
            if t_next in DB_group_list:
                tokenFeatures.append(f"DB_group_next")
            if t_next in hsdb_list:
                tokenFeatures.append(f"hsdb_next")

            tokenFeatures.append("suf3_next=" + t_next[-3:])
            tokenFeatures.append("suf4_next=" + t_next[-4:])
            tokenFeatures.append("suf5_next=" + t_next[-5:])
            tokenFeatures.append("suf6_next=" + t_next[-6:])

            if sum(1 for c in t_next if c.isupper()) == 1:
                tokenFeatures.append("OC_next")
            elif sum(1 for c in t_next if c.isupper()) > 1:
                tokenFeatures.append("MTOC_next")

            if any(c in string.punctuation or c.isdigit() for c in t_next):
                tokenFeatures.append("POSC_next")

            tokenFeatures.append("pre3_next=" + t_next[:3])
            tokenFeatures.append("pre4_next=" + t_next[:4])
            tokenFeatures.append("pre5_next=" + t_next[:5])
            tokenFeatures.append("pre6_next=" + t_next[:6])

            pos_tag = nltk.pos_tag([t_next])[0][1]
            tokenFeatures.append("POS_next=" + pos_tag)

            pos = pos_tag[0].lower() if pos_tag and pos_tag[0].lower() in {'n', 'v', 'a', 'r', 's'} else 'n'
            lemma_next = lemmatizer.lemmatize(t_next, pos)
            tokenFeatures.append("lemma_next=" + lemma_next)

        result.append(tokenFeatures)
    return result

def extract_features_final(tokens, DB_drug_list, DB_brand_list, DB_group_list, hsdb_list):
    result = []
    # Initialize the WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()

    for k in range(len(tokens)):
        tokenFeatures = []
        t = tokens[k][0]

        tokenFeatures.append("form=" + t)
        tokenFeatures.append("lower=" + t.lower())
        tokenFeatures.append("length=" + str(len(t)))


        if t in DB_drug_list:
            tokenFeatures.append("DB_drug")
        if t in DB_brand_list:
            tokenFeatures.append("DB_brand")
        if t in DB_group_list:
            tokenFeatures.append("DB_group")
        if t in hsdb_list:
            tokenFeatures.append("hsdb")

        if k == 0:
            tokenFeatures.append("BoS")
        elif k == len(tokens) - 1:
            tokenFeatures.append("EoS")

        tokenFeatures.append("suf3=" + t[-3:])
        tokenFeatures.append("suf4=" + t[-4:])
        tokenFeatures.append("suf5=" + t[-5:])
        tokenFeatures.append("suf6=" + t[-6:])

        # add the string "OC" if there is one capital letter in the word
        # or add the string "MTOC" if there are more than one capital letters in the word
        if sum(1 for c in t if c.isupper()) == 1:
            tokenFeatures.append("OC")
        elif sum(1 for c in t if c.isupper()) > 1:
            tokenFeatures.append("MTOC")

        # 5. add the string "POSC" if there is presence of special characters in the word,
        # like numbers, dashes, pantuaction marks, etc.
        if any(c in string.punctuation or c.isdigit() for c in t):
            tokenFeatures.append("POSC")

        tokenFeatures.append("pre3=" + t[:3])
        tokenFeatures.append("pre4=" + t[:4])
        tokenFeatures.append("pre5=" + t[:5])
        tokenFeatures.append("pre6=" + t[:6])

        # part of speach (POS) tag of the word
        pos_tag = nltk.pos_tag([t])[0][1]
        tokenFeatures.append("POS=" + pos_tag)

        pos = pos_tag[0].lower() if pos_tag and pos_tag[0].lower() in {'n', 'v', 'a', 'r', 's'} else 'n'
        lemma = lemmatizer.lemmatize(t, pos)
        tokenFeatures.append("lemma=" + lemma)

        if k > 0:
            t_prev = tokens[k - 1][0]

            tokenFeatures.append("length_prev=" + str(len(t_prev)))

            if t_prev in DB_drug_list:
                tokenFeatures.append(f"DB_drug_prev")

            tokenFeatures.append("suf4_prev=" + t_prev[-4:])

            if sum(1 for c in t_prev if c.isupper()) == 1:
                tokenFeatures.append("OC_prev")
            elif sum(1 for c in t_prev if c.isupper()) > 1:
                tokenFeatures.append("MTOC_prev")

            if any(c in string.punctuation or c.isdigit() for c in t_prev):
                tokenFeatures.append("POSC_prev")

            pos_tag = nltk.pos_tag([t_prev])[0][1]
            tokenFeatures.append("POS_prev=" + pos_tag)

        if k < len(tokens) - 2:  # We are before the second last word of the sentence,
            # take the same features of the next two words
            t_next = tokens[k + 2][0]

            if sum(1 for c in t_next if c.isupper()) == 1:
                tokenFeatures.append(f"OC_next_2")
            elif sum(1 for c in t_next if c.isupper()) > 1:
                tokenFeatures.append(f"MTOC_next_2")

        result.append(tokenFeatures)
    return result


## --------- MAIN PROGRAM -----------
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --

# directory with files to process
datadir = sys.argv[1]
drug_bank_dir = sys.argv[2]
hsdb_dir = sys.argv[3]

DB_drug_list, DB_brand_list, DB_group_list = ext_drug_bank(drug_bank_dir)
hsdb_list = ext_HSDB(hsdb_dir)

st = time.time()
# process each file in directory
for f in listdir(datadir):

    # parse XML file, obtaining a DOM tree
    tree = parse(datadir+"/"+f)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value   # get sentence id
        spans = []
        stext = s.attributes["text"].value   # get sentence text
        entities = s.getElementsByTagName("entity")
        for e in entities:
            # for discontinuous entities, we only get the first span
            # (will not work, but there are few of them)
            (start, end) = e.attributes["charOffset"].value.split(";")[0].split("-")
            typ = e.attributes["type"].value
            spans.append((int(start), int(end), typ))


        # convert the sentence to a list of tokens
        tokens = tokenize(stext) # here we have a list containing for each word of a sentence the following set (word, offset-start, offset-end)
        # extract sentence features
        # features = extract_features(tokens)
        # features = extract_features1(tokens, DB_drug_list, DB_brand_list, DB_group_list, hsdb_list)
        # get the start time

        features = extract_features_final(tokens, DB_drug_list, DB_brand_list, DB_group_list, hsdb_list)

        # print features in format expected by crfsuite trainer
        for i in range(0, len(tokens)):
            # see if the token is part of an entity
            tag = get_tag(tokens[i], spans)
            print(sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

        # blank line to separate sentences
        print()

# get the end time
# et = time.time()
#
# # get the execution time
# elapsed_time = et - st
# sys.stderr.write(f'Execution time: {elapsed_time} seconds\n')