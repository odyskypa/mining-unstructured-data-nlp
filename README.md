# Natural Language Processing with Traditional Machine Learning (ML) and Deep Learning (DL)
Project of Mining Unstructed Data (MUD) Course for Master in Data Science Program of Universitat Politècnica de Catalunya (UPC)
***
The project is divided into 3 separate parts:

1. [Language Detection](./src/language_detection)
2. [Name Entity Recognition and Classification (NERC)](./src/NERC_traditional_ML) of pharmaceutical products and [Drug-Drug Interaction Detection and Classification (DDI)](./src/DDI_traditional_ML) with traditional Machine Learning (ML) techniques
3. [NERC](./src/NERC_DL) of pharmaceutical products and [DDI](./src/DDI_DL) with state-of-the-art Deep Learning (DL) techniques

Both solutions of part `2.` and `3.` make use of the `BIO tagging approach`.

Below one can find information about the 3 parts of the project.

***
### Language Detection
In this part of the project an analysis of the performance of several preprocessing techniques and Machine Learning models for `Language Detection` task was accomplished. Word and character tokenization was tested and compared, as well as several different vocabulary sizes.

#### Preprocessing Steps
Preprocessing includes the usage of `NTLK` library, for `tokenization`, `lemmatization` and `POS tagging` of words or characters. Additionally, `PCA` was implemented for dimensionality reduction.

#### Classifiers Implemented
* Naive Bayes (NB)
* k-Nearest Neighbors (kNN)
* Decision Tree Classifier
* Support Vector Machine (SVM)

The source code used for this part of the project can be found [here](./src/language_detection)

***
### NERC and DDI with Traditional ML Techniques
In this part of the project, `NERC` and `DDI` tasks were performed by utilizing traditional ML approaches.

#### NERC with Traditional ML Techniques
In order to recognize and classify the pharmaceutical entities of the available data that were extracted from biomedical texts, experiments were performed with two different classifiers: `Naive Bayes (NB)` and `Conditional Random Fields (CRF)`.

Before applying the classifiers, the most imprortant stage of this part of the project was the `generation of the features`. The features used are the following:

* Form of the token (form=)
* Lowercased form of the token (lower=)
* Length of the token (length=)
* A set of flags indicating whether the token matches certain words from four different lists (DB_drug_list, DB_brand_list, DB_group_list, and hsdb_list)
* Whether the token is at the beginning or end of a sentence (BoS, EoS)
* Suffixes of length 3, 4, 5, and 6 of the token (e.g. suf3=, suf4=, etc.)
* A set of flags indicating whether the token has special characters (POSC), such as punctuation or digits, or occurrences of one capital letter (OC) or more (MTOC).
* Prefixes of length 3, 4, 5, and 6 of the token (e.g. pre3=, pre4=, etc.)
* POS tag of the token, as determined by the NLTK library (POS=).
* Lemma of the token, as determined by the WordNet lemmatizer (lemma=).
* The same features as above for the previous one or two tokens, depending on the position of the current token in the sentence (e.g. form_prev_1=, form_prev_2, form_next_1, form_next_2, etc.). In this case, checks were performed in order to validate if a token is at the start or at the end  of a sentence in order to avoid exceeding the sentence length or trying to select tokens with indexes like -1. All different cases are taken into account.

One important thing that needs to be clarified here is that the result of the extract-features.py script that is generating the features used for training the classifiers, is making use of the `B-I-O schema tagging`. `B-I-O schema tagging`, also known as the IOB (Inside-Outside-Beginning) tagging scheme, `is a commonly used method for NER`. In the B-I-O schema, each word in a text is tagged with a label that indicates whether it is part of an entity or not, and if so, which type of entity it belongs to.

Once all the above-mentioned features were calculated, `feature-importance` analysis was performed in order to conclude to the final set of features for training the classifiers. Graphs about feature importance per class of the analysis can be found in [feature-graphs folder](./feature-graphs). One example for the `B-drug` class is depicted in the figure below:

![](./feature-graphs/max-score-B-drug.png)

#### DDI with Traditional ML Techniques

***
### NERC and DDI with Deep Learning

#### NERC with Deep Learning

#### DDI with Deep Learning
