import math
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd
import spacy

# Load the xx_sent_ud_sm model for sentence segmentation and xx_ent_wiki_sm for lemmatization
nlp_sent = spacy.load('xx_sent_ud_sm')
nlp_lem = spacy.load('xx_ent_wiki_sm')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

#Tokenizer function. You can add here different preprocesses.
def preprocess0(sentence, labels):
    '''
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting, 
    tokenization or sentence splitting.

    Input: Sentence in string format
    Output: Preprocessed sentence either as a list or a string
    '''
    # Place your code here
    # Keep in mind that sentence splitting affectes the number of sentences
    # and therefore, you should replicate labels to match.
    return sentence,labels

#Tokenizer function. You can add here different preprocesses.
def preprocess(sentences, labels):
    '''
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting, 
    tokenization or sentence splitting.

    Input: Sentence in string format
    Output: Preprocessed sentence either as a list or a string
    '''
    # Split the input sentences into individual sentences using the xx_sent_ud_sm model
    preprocessed_sentences = []
    preprocessed_labels = []
    for i,j in zip(sentences.index, labels.index):
        doc = nlp_sent(sentences[i])
        for i, sent in enumerate(doc.sents):
            # Lowercase the text
            sent_text_lower = sent.text.lower()
            
            # Remove punctuation
            sent_text = sent_text_lower.translate(str.maketrans('', '', string.punctuation))
            
            # Lemmatize the text using the xx_ent_wiki_sm model
            sent_doc = nlp_lem(sent_text)
            sent_lemmas = [token.lemma_ if token.lemma_ != '' else token.text for token in sent_doc]
            
            # Strip Lemmas and Append the preprocessed sentence and corresponding label to the results
            preprocessed_sentences.append(' '.join(sent_lemmas).strip())
            preprocessed_labels.append(labels[j])
    preprocessed_sentences = pd.Series(preprocessed_sentences)
    preprocessed_labels = pd.Series(preprocessed_labels)
    
    # Return the preprocessed sentences and replicated labels
    return preprocessed_sentences, preprocessed_labels



def preprocess1(sentences, labels):
    '''
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting, 
    tokenization or sentence splitting.

    Input: Sentence in string format
    Output: Preprocessed sentence either as a list or a string
    '''
    # Split the input sentences into individual sentences using NLTK sentence tokenizer
    preprocessed_sentences = []
    preprocessed_labels = []
    for i,j in zip(sentences.index, labels.index):
        sent_list = sent_tokenize(sentences[i])
        for sent in sent_list:
            # Lowercase the text
            sent_text_lower = sent.lower()
            
            # Remove punctuation
            sent_text = sent_text_lower.translate(str.maketrans('', '', string.punctuation))
            
            # Tokenize the text using NLTK word tokenizer
            sent_tokens = word_tokenize(sent_text)
            
            # Lemmatize the text using NLTK WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
            sent_lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in sent_tokens]
            
            # Append the preprocessed sentence and corresponding label to the results
            sent_filtered = [word for word in sent_lemmas]
            preprocessed_sentences.append(' '.join(sent_filtered).strip())
            preprocessed_labels.append(labels[j])
    
    # Convert the preprocessed sentences and labels into pandas Series
    preprocessed_sentences = pd.Series(preprocessed_sentences)
    preprocessed_labels = pd.Series(preprocessed_labels)
    
    # Return the preprocessed sentences and replicated labels
    return preprocessed_sentences, preprocessed_labels


def get_wordnet_pos(word):
    '''
    Map POS tag to first character used by WordNetLemmatizer
    '''
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_splitting_four_char_based(sentence, labels):
    '''
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting,
    tokenization or sentence splitting.

    Input: Sentence in string format
    Output: Preprocessed sentence either as a list or a string
    '''
    # Place your code here
    new_sentences = []
    new_labels = []
    labels_list = labels.tolist()
    num_substrings = 4
    for index, s in enumerate(sentence):
        n, r = divmod(len(s), num_substrings)
        parts = [s[(i * n) + min(i, r):((i + 1) * n) + min(i + 1, r)] for i in range(num_substrings)]
        for part in parts:
            new_sentences.append(part)
            new_labels.append(labels_list[index])

    new_sentences_series = pd.Series(new_sentences)
    new_labels_series = pd.Series(new_labels)
    return new_sentences_series, new_labels_series

def preprocess_splitting_four(sentence, labels):
    '''
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting, 
    tokenization or sentence splitting.

    Input: Sentence in string format
    Output: Preprocessed sentence either as a list or a string
    '''
    # Place your code here
    # Split each sentence into exactly 4 parts
    new_sentences = []
    new_labels = []
    labels_list = labels.tolist()
    for index, s in enumerate(sentence):
        words = s.split()
        n = len(words)
        parts = []
        for i in range(0, n, math.ceil(n / 4)):
            part = words[i:i + math.ceil(n / 4)]
            parts.append(part)
        for part in parts:
            new_sentences.append(' '.join(part))
            new_labels.append(labels_list[index])

    new_sentences_series = pd.Series(new_sentences)
    new_labels_series = pd.Series(new_labels)
    return new_sentences_series, new_labels_series