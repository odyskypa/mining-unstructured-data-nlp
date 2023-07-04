# Language Detection, Name-Entity Recognition and Classification, Drug-Drug Interaction Detection and Classification
Project of Mining Unstructed Data (MUD) Course for Master in Data Science Program of Universitat Politècnica de Catalunya (UPC)
***
The project is divided into 3 separate parts:

1. [Language Detection](./src/language_detection)
2. [Name Entity Recognition and Classification (NERC)](./src/NERC_traditional_ML) of pharmaceutical products and [Drug-Drug Interaction Detection and Classification (DDI)](./src/DDI_traditional_ML) with traditional Machine Learning (ML) techniques
3. [NERC](./src/NERC_DL) of pharmaceutical products and [DDI](./src/DDI_DL) with state-of-the-art Deep Learning (DL) techniques

Both solutions of part `2.` and `3.` make use of the `BIO tagging approach`. Also, part `2` and `3` consist of the same `training/dev/test` data and only the solution's implementation changes.

All the experiments, results and conclusions of each individual part of the project can be found in the documents located at the [docs folder](./docs).

Below one can find information about the 3 parts of the project.

For getting access to the data of the project, contact me via email @`odykypar@gmail.com`

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
In more detail, inside the given sentences that exist in the corpora, there are four types of entities that need to be recognized and subsequently classified, namely `drug`, `brand`, `group`, and `drug_n`. In order to recognize and classify the pharmaceutical entities of the available data that were extracted from biomedical texts, experiments were performed with two different classifiers: `Naive Bayes (NB)` and `Conditional Random Fields (CRF)`.

Before applying the classifiers, the most imprortant stage of this part of the project was the `generation of the features`. The features used are the following:

* Form of the token (`form=`)
* Lowercased form of the token (`lower=`)
* Length of the token (`length=`)
* A set of flags indicating whether the token matches certain words from four different lists (`DB_drug_list`, `DB_brand_list`, `DB_group_list`, and `hsdb_list`)
* Whether the token is at the beginning or end of a sentence (`BoS`, `EoS`)
* Suffixes of length 3, 4, 5, and 6 of the token (e.g. `suf3=`, `suf4=`, etc.)
* A set of flags indicating whether the token has special characters (`POSC`), such as punctuation or digits, or occurrences of one capital letter (`OC`) or more (`MTOC`).
* Prefixes of length 3, 4, 5, and 6 of the token (e.g. `pre3=`, `pre4=`, etc.)
* POS tag of the token, as determined by the NLTK library (`POS=`).
* Lemma of the token, as determined by the WordNet lemmatizer (`lemma=`).
* The same features as above for the previous one or two tokens, depending on the position of the current token in the sentence (e.g. `form_prev_1=`, `form_prev_2`, `form_next_1`, `form_next_2`, etc.).
  * In this case, checks were performed in order to validate if a token is at the start or at the end  of a sentence in order to avoid exceeding the sentence length or trying to select tokens with indexes like -1. All different cases are taken into account.

One important thing that needs to be clarified here is that the result of the `extract-features.py` script that is generating the features used for training the classifiers, is making use of the `B-I-O schema tagging`. `B-I-O schema tagging`, also known as the IOB (Inside-Outside-Beginning) tagging scheme, `is a commonly used method for NER`. In the B-I-O schema, each word in a text is tagged with a label that indicates whether it is part of an entity or not, and if so, which type of entity it belongs to.

Consequently, the `B-I-O tags` are being used by the classification algorithms to detect if a word in the sentence is an entity, and then to classify those entities into one of the following target labels: `B-drug`, `B-brand`, `B-group`, `B-drug_n`, `I-drug`, `I-brand`, `I-group`, `I-drug_n` and `O`.

Once all the above-mentioned features were calculated, `feature-importance` analysis was performed in order to conclude to the final set of features for training the classifiers. Graphs about feature importance per class of the analysis can be found in [feature-graphs folder](./feature-graphs). One example for the `B-drug` class is depicted in the figure below:

![](./feature-graphs/max-score-B-drug.png)

Finally, hyper-parameter tuning of the 2 mentioned classifiers took place.

The source code used for this part of the project can be found [here](./src/NERC_traditional_ML)

#### DDI with Traditional ML Techniques

With the `DDI Corpus`, which is a semantically annotated corpus of documents describing drug-drug interactions from the DrugBank[^1] database and MedLine[^2] abstracts, we face the goal of detecting when there is an interaction and when not, and classifying the existing drug-drug interactions into `advice`, `effect`, `mechanism`, and `int`.

In this case, the performed experiments made use of two ML algorithms, `Naive Bayes` and `Linear Support Vector Machine (Linear SVM)`.

The input to the `feature extractor` is not the sentence itself, but a `dependency tree` with added individual features about every node/word, generated with `CoreNLP`[^3]. CoreNLP (Core Natural Language Processing) is a popular open-source software library developed by Stanford University that provides a suite of natural language processing tools for a variety of tasks, including tokenization, part-of-speech (POS) tagging, named entity recognition, parsing, sentiment analysis, and coreference resolution.

The CoreNLP dependency tree object contained, for every word/node in a sentence, a dictionary with the following relevant information:

* Literal word 	
* Lemma of the word
* Part-of-speech (POS) tagging
* Address of head node
* Relations/dependencies with other nodes
* Offsets in the sentence

And then from there, with the help of the deptree.py module we could also obtain or infer:

* Least Common Subsumer (LCS) between words
* Paths in the dependency tree (labeled as we wanted)
* Head entity of a node
* Who is drug entity

It is important to mention here, that this part of the project is focused again in the `feature selection` process, for the task of DDI.

The `set of the initial features` was composed of:

* `lib`: lemma of the first token after e1 entity head
* `wib`: lemma of the first token after e2 entity head
* `lpig`: lemma and PoS tag of the first token after drug1 entity head
* `eib`: True/False. Presence of a third drug entity between e1 and e2.
* `path`: path with lemmas and relations from e1 head to LCS to e2 head
* `path1`: path up with lemmas and relations from e1 head to LCS
* `path2`: path down with lemmas and relations from LCS to e2 head 

Where the head token for each gold entity refers to the token in the dependency parse tree that corresponds to the main noun or verb that represents the entity. 

Additionally, a set of tests was performed to assess the effects of adding the rest of a particular token information (word, lemma, PoS tag, and combinations).

Also, features like `path1_nodes`, `path2_nodes`, `path1_edges`, `path1_nodes`, `path_edges` and `path_edges` were created which are based purely on words and PoS tags from the dependency relations. The addition of these features gave us a `3% increase on average in the macro F1` in the tests we performed. Two examples of these new features are:
```
 path2_nodes=inhibitor<inhibitor<NN<metabolism< metabolism<NN<theophylline<theophylline<NN
 path_edges=conj<<<indir<conj<<<indir
```
  
Finally, by analyzing the meaning behind the four different types of interactions to see if there was any pattern or possible indicators in the sentences that we could use in our favor to differentiate between them, rather than more analytical approaches as presented before. Here is the definition of every interaction type:

* `Advice`: DDI in which a recommendation or advice regarding the concomitant use of two drugs involved in them is described.
   * e.g.:  "Interactions may be expected, and UROXATRAL should NOT be used in combination with other alpha-blockers."
* `Effect`: DDI in which the effect of the drug-drug interaction is described.
   * e.g.: “Aspirin and Paracetamol help decrease temperature.”
* `Mechanism`: The mechanism of interaction can be pharmacodynamic (the effects of one drug are changed by the presence of another drug at its site of action, for example, "alcohol potentiates the depressor effect of barbiturates") pharmacokinetic (the processes by which drugs are absorbed, distributed, metabolized and excreted are affected, for example, ("induced the metabolism of", "increased the clearance of'). As already noted, a pharmacodynamic relationship between entities must be considered type effect.
   * e.g.: Grepafloxacin, like other quinolones, may inhibit the metabolism of caffeine and theobromine.
* `int`: the sentence simply states that an interaction occurs and does not provide any information about the interaction.
   * e.g.: The interaction of omeprazole and ketoconazole has been established.

The presence of certain `clue verbs before, between, and after` the `drug entities` could be useful information for labeling the different types of interactions.

The list of key terms for the `clue lemmas list` is the following:

```
 clue_lemmas = ["affect", "effect", "diminish", "produce", "increase", "result", "decrease", "induce", "enhance", "lower", "cause", "interact", "interaction", "shall", "caution", "advise", "reduce", "prolong", "not"]
```

The source code used for this part of the project can be found [here](./src/DDI_traditional_ML)

***
### NERC and DDI with Deep Learning
In this part of the project, the problems of NERC and DDI which needed to be addressed are exactly the same as in the previous section. The main difference of the solution, is the introduction of `Deep Learning` techniques. Special attention has been paid to the implementation of `Word2Vec`, `GloVe` and `BiLSTM` for the NERC task, while `CNN`, `BiLSTM`, `Tranformer` and `BERT` architectures implemented for the DDI task.

#### NERC with Deep Learning

The overall architecture of the neural network of this solution is defined using the `Keras functional API`. The model is constructed by specifying the input layers and the subsequent flow of data through the `embedding`, `bidirectional LSTM (BiLSTM)`, `hidden`, and `output layers`.

Compared to the traditional ML approach followed before, this approach focuses on both using appropriate word embeddings to train the network in collaboration with new input features to improve results within the learning algorithm.

To begin with, the following techniques will be tested to further enhance the performance of the network: 
* Adding custom input embedding layers with: 
   * `lowercase` words, 
   * their `POS tags`, 
   * their `lemmas`, 
   * `suffixes` and `prefixes` of length 3 to 6
   * `words` that `contain one capital letter`, `multiple capital letters`, `digits`, and `punctuation` respectively
* Moreover, four additional input layers have been tested, which contain:
   * `words` that are `included in the external sources` provided (DrugBank.txt, HSDB.txt), as well as
   * a `frozen layer` containing `pre-trained embeddings from Stanford (GloVe)`.

##### Input Layers
* `inptW`: layer for word indices representing the main words in the sentences.
* `inptS`: layer for suffix of length 5 indices representing the suffixes of the words.
* `inptLc`: lowercase word indices, which represent the lowercase versions of the words.
* `inptS3`, `inptS4`, `inptS6`: layers for suffix indices of different lengths (3, 4, and 6 characters), capturing variations in word endings.
* `inptPos`: part-of-speech tag indices representing the grammatical category of the words.
* `inptCs`: capitalization input layer indicating whether each word contains more than one capitalized letter or not.
* `inptSC`: single capitalization indices indicating whether each word contains a single capital letter.
* `inptPU`: punctuation input layer representing the presence or absence of punctuation marks in the words.
* `inptD`: digit indices layer indicating the presence or absence of digits in the words.
* `inptDr`: indices layer representing the presence or absence of drug-related terms in the words.
* `inptBr`: representing the presence or absence of brand-related terms in the words.
* `inptGr`: include the presence or absence of group-related terms in the words.
* `inptHs`: indices that include the presence or absence of HSDB (Hazardous Substances Data Bank) terms in the words.

##### Embedding Layers
* `embW`: embeddings for word indices with an output dimension of 100.
* `embS`: embedding layer for suffix indices with an output dimension of 50.
* `embLc`: lowercase word indices with an output dimension of 100.
* `embS3`, `embS4`, `embS6`: suffixes of different lengths (3, 4, and 6 characters) with an output dimension of 50.
* `embPos`: part-of-speech tag embedding layer with an output dimension of 50.
* `embCs`: capitalization embedding layer indicating whether each word contains more than one capitalized letter or not with an output dimension of 50.
* `embSC`: embeddings for single capitalization indices with an output dimension of 50.
* `embPU`: punctuation indices with an output dimension of 50.
* `embD`: embedding layer for digit indices with an output dimension of 50.
* `embDr`: drug-related term indices embeddings with an output dimension of 50.
* `embBr`: brand-related term indices with an output dimension of 50.
* `embGr`: embeddings for group-related term indices with an output dimension of 50.
* `embHs`: HSDB term indices embedding layer with an output dimension of 50.

##### Pre-trained GloVe Word Embeddings
The GloVe word embeddings capture semantic relationships between words based on their co-occurrence statistics in a large corpus of text. The embeddings are loaded from the `glove.6B.300d.txt` file (found [here](https://nlp.stanford.edu/projects/glove/)), which contains word vectors of dimensionality 300.

##### BiLSTM Layer
After embedding the input features, a `BiLSTM layer` is employed to **capture contextual information and sequential dependencies within the text**. The BiLSTM layer consists of `300 hidden units` and is set to `return sequences`. Additionally, as an **activation function** the `hyperbolic tangent function (tanh)` was used. Although `ReLU activation function` was tested as well in this part of the architecture but the results were worse compared to Tanh.

##### Hidden Layer
A dense hidden layer follows the BiLSTM layer. This layer serves as a non-linear transformation to further extract and encode relevant features from the contextual representations provided by the BiLSTM. The intention of adding a hidden layer is to allow the network to learn more complex patterns included in the data. However, the difference in the final results is not so important. The hidden layer was tested with different configurations, including tries of 50 and 200 units (neurons) and usage of tanh and ReLU activation functions.

###### Output Layer
Finally, a `time-distributed dense layer` is applied to produce the output predictions for each position in the input sequences. The **number of units in this layer corresponds to the number of unique labels** in the dataset. The activation function used in the output layer is a `softmax function`, which yields a **probability distribution among the target labels**, enabling the classification of each word in the input sequence.


The source code used for this part of the project can be found [here](./src/NERC_DL)

#### DDI with Deep Learning
In this part of the project the following solution was developed for the DDI task.

##### CNN and BiLSTM
`Convolutional layers` are powerful for learning local patterns in data, while `Bidirectional LSTMs` are effective at capturing long-term dependencies in sequential data, and in both directions. Because of that, there are many ways of attempting to combine them to obtain enhanced results. We acknowledged the three alternatives:

* CNN’s output as the input to the BiLSTM. It may allow the BiLSTM to learn features from the input data that have been learned by the CNN
* LSTM’s output as the input to the CNN . It may allow the BiLSTM to learn features from the input data that have been learned by the CNN.
* The CNN and BiLSTM operate on the input data independently, and their outputs are concatenated and passed to maybe a fully connected layer.

##### Early stopping
It was realized that the model did overfit way before arriving at the default number of `10` epochs in some tests. That is, the validation loss and validation accuracy started to get worse than expected. The solution introduced for that was to `stop training always a fixed 10 epochs` and instead implement an early stopping mechanism with the patience of 5 epochs over the validation loss reduction.

##### Input and Embedding Layers
By following the results of the previous parts of the project, it was decided to add:
* `PoS`, `lemma`, and `lowercase` spaces of the sentence words
The immediate impact of their addition was low, but we believe that in effect it led to future improvements since most of the time we added further complexity, the results improved steadily. Every `embedding trainable instance` has a size of `100`.

##### Transformer and BERT
`Transformers` can be suitable for drug-drug interaction classification in sentences because they **effectively capture contextual information and long-range dependencies** within the text. Also, as **opposed to RNN-based approaches**, parallelization of processes is not an issue and can lead to the creation of more complex models. Putting apart temporarily the CNN and LSTM model, it was decided to **check the behavior of a model with** only the `word embeddings and a default transformer design, with multi-attention heads of size 2, embedding size 100, and the use of positional encoding integrated`.

From this model, it was decided to add trainable parameters complexity and other embeddings to see how the results evolved. Tries with embedding size of `100`, `200`, and `512` and for different head and dense layers dimensionality values took place. Results for different transformers' complexity were quite similar, with F1 scores far from the best ones throughout the project. It is believed that the main reason for this result is due to the lack of a bigger dataset or the ability to properly parallelize the pipeline, which with the limited resources it was impossible to take advantage of, compared to CNN + LSTM approaches.

Another test implememented was to add `pre-trained word embeddings` to the model. `BERT (Bidirectional Encoder Representations from Transformers)` is a powerful **pre-trained contextual representation system built on bi-directionality**. The `BERT` system was used to obtain the embeddings, but even with deactivated training for the BERT parameters, it was too much for the Collab environment to handle. The alternative was to use the applicable GloVe embeddings that we had applied in the NERC task here as well and see the impact.

##### Dense layers
Finally, different model designs were tested by adding or removing dense layers. Also the `ReLu` activation function for all the hidden layers was tested as well, except the last layer which is implementing the `softmax` activation. Including `max-pooling` or `global average pooling transformations` between was not thoroughly tested and can be something interesting for further iterations.

The source code used for this part of the project can be found [here](./src/DDI_DL)

[^1]: [https://go.drugbank.com/](https://go.drugbank.com/)
[^2]: [https://www.nlm.nih.gov/medline/medline_overview.html](https://www.nlm.nih.gov/medline/medline_overview.html)
[^3]: [https://stanfordnlp.github.io/CoreNLP/](https://stanfordnlp.github.io/CoreNLP/)
