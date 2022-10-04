# Text Sentiment Analysis: A Collection of Notes and Examples

This repository collects/links projects related to text sentiment analysis, as well as some high level notes on the techniques used in the field.

:warning: Important notes, first of all:

[![Unfinished](https://img.shields.io/badge/status-unfinished-orange)](https://shields.io/#your-badge)

- This is an on-going project; I will extend the content as far as I have time for it.
- In some cases, I will use the code from other public tutorials/posts rightfully citing the source.
- In addition to the examples, the links in the section [Interesting Links](#interesting-links) are very useful resources for those interested in the topic.

## Introduction

The techniques used in [sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis) are able to efficiently condense raw sequences of words to one valuable scalar which covers the spectrum from `negative` to `positive` (or an equivalent one). Then, that scalar can become an important feature in more complex models that output important business predictions. In my personal experience, that mapping from a text string to a number is very handy in plenty of businesses that work with tabular datasets that contain free text fields.

This repository serves two purposes related to the topic of **sentiment analysis**:

1. This is a document for my future self which collects techniques that I come across.
2. I present a primer on the basics of Natural Language Processing (NLP) for pragmatic engineers that specialize on other fields, but the Universe gifts them with valuable text data.

Overview of Contents:

- [Text Sentiment Analysis: A Collection of Notes and Examples](#text-sentiment-analysis-a-collection-of-notes-and-examples)
  - [Introduction](#introduction)
  - [How to Use This](#how-to-use-this)
    - [Dependencies](#dependencies)
  - [Natural Language Processing Pipeline: General Notes](#natural-language-processing-pipeline-general-notes)
    - [Preprocessing](#preprocessing)
    - [Preprocessing with SpaCy](#preprocessing-with-spacy)
    - [Beyond the Vocabulary: Word Vectors, Bags and Sequences](#beyond-the-vocabulary-word-vectors-bags-and-sequences)
    - [Vectorization with Scikit-Learn](#vectorization-with-scikit-learn)
    - [Sentiment Analysis: A Binary Classification Problem](#sentiment-analysis-a-binary-classification-problem)
  - [Recurrent Neural Networks: General Notes](#recurrent-neural-networks-general-notes)
    - [Quick Usage Examples of LSTMs in Pytorch](#quick-usage-examples-of-lstms-in-pytorch)
  - [List of Examples + Description Points](#list-of-examples--description-points)
  - [Improvements, Next Steps](#improvements-next-steps)
  - [Interesting Links](#interesting-links)
  - [Authorship](#authorship)

## How to Use This

I collect, either

- links to other repositories, mine or others',
- notebooks,
- or code stored in folders.

All of them are referenced/indexed in the section [List of Examples + Description Points](#list-of-examples--description-points).

1. Select one example from the section [List of Examples + Description Points](#list-of-examples--description-points). If there are additional instructions, they should be in the folder where the example is.
2. If there is an `Open in Colab` button anywhere, you can use it :smile:.
3. If you'd like to run the code locally on your device install the [dependencies](#dependencies) and run the main file in it; often, the main file will be a notebook.

Additionally, the following sections give high-level explanations of the overall processes and tools necessary in Natural Language Processing (NLP):

- [Natural Language Processing Pipeline: General Notes](#natural-language-processing-pipeline-general-notes)
- [Recurrent Neural Networks: General Notes](#recurrent-neural-networks-general-notes)

### Dependencies

You should create a python environment (e.g., with [conda](https://docs.conda.io/en/latest/)) and install the dependencies listed in the [requirements.txt](requirements.txt) file of each example. If there is no such file in a folder example, the one in the root level should work.

A short summary of commands required to have all in place is the following; however, as mentioned, **each example might have its own specific dependency versions**:

```bash
conda create -n text-sent python=3.6
conda activate text-sent
conda install pytorch -c pytorch 
conda install pip
pip install -r requirements.txt
```

## Natural Language Processing Pipeline: General Notes

:construction: To be done.

This section introduces very briefly the most essential tasks carried out for text preprocessing; if you require more information, you can check [my NLP Guide](https://github.com/mxagar/nlp_guide).

### Preprocessing

Raw text needs to be processed to transform words into numerical vectors. The following image summarizes usual steps that are performed to that end.

IMAGE

First, the text string needs to be properly split into basic units of meaning or **tokens**, i.e., we perform the **tokenization**. A figure from [Spacy: Linguistic Features](https://spacy.io/usage/linguistic-features) gives a good example of that process:

![Tokenization (from the Spacy website)](./assets/tokenization.png)

Quite often the token form of the word converted to a vector; however, that token can be further processed to more basic forms with **stemming and lemmatization**:

- Stemming: `runs --> run (stem)`
- Lemmatization: `was --> be (lemma)`

Additionally, several **properties of the word** can be identified, if required:

- Part of Speech (POS) or morphology: is it a noun, verb, adverb, punctuation, etc.?
- Syntactic function (dependency): is it the subject, a determinant, etc.?
- Named Entity Recognition (NER): is this a company name, a money quantity, etc.?

Next, typically ***stop-words* are removed**; these words that appear with high frequency and do not convey much information, e.g., "the", "of", etc. Similarly, punctuation symbols might be also candidates to be removed.

And finally, the **vocabulary** is built. This is an object which collects all possible tokens/words in a set; optimally, we should index all the tokens/words so that:

- given an index, we know its word/token text form,
- and given a word/token text form, we know its index.

Even though the preprocessing might seem a lot of effort, I've seen quite often to reduce the work to this **simplified recipe**:

1. Remove punctuation: `replace()` with `''` any character contained in the `string.punctuation` module.
2. Quick and dirty tokenization, i.e., basically split text strings: `text.lower().split()`.
3. Create a vocabulary with a `set()` and a `dict()`.

```python
# ...
text =  """This story is about surfing.
        Catching waves is fun!
        Surfing is a popular water sport.
        """
# Build vocabulary (dictionary)
vocab = {}
x = text.lower().split()
for word in x:
    if word in vocab:
        continue
    else:
        vocab[word]=i
        i+=1
# Vocabulary size
n = len(vocab)
```

However, if you'd like to squeeze the maximum amount of information from the text, you can have a look at the next section, which summarizes how to easily perform all the preprocessing steps I mentioned using [SpaCy](https://spacy.io/usage).

### Preprocessing with SpaCy

In the following, I provide some lines which show how to use [SpaCy](https://spacy.io/usage) to perform advanced text preprocessing. To figure out how to install SpaCy, look at the [SpaCy installation site](https://spacy.io/usage); and if you'd like more information on how to use the library, check my [my NLP Guide](https://github.com/mxagar/nlp_guide).

```python
import spacy

# We load our English model
nlp = spacy.load('en_core_web_sm')

# Create a _Doc_ object:
# the nlp model processes the text 
# and saves it structured in the Doc object
# u: Unicode string (any symbol, from any language)
doc = nlp(u'Tesla is looking at buying U.S. startup for $6 million')

# Print each token separately
# Tokens are word representations, unique elements
# Note that spacy does a lot of identification work already
# $ is a symbol, U.S. is handled as a word, etc.
for token in doc:
    # token.text: raw text
    # token.pos_: part of speech: noun, verb, punctuation... (MORPHOLOGY)
    # token.dep_: syntactic dependency: subject, etc. (SYNTAX)
    # token.lemma_: lemma
    # token.is_stop: is the token a stop word?
    print(token.text, token.pos_, token.dep_)

# Loop in sentences
for sent in doc.sents:
    print(sent)

# Print the set of SpaCy's default stop words
print(nlp.Defaults.stop_words)

# Named entities (NER)
for ent in doc.ents:
    print(ent.text, ent.label_, str(spacy.explain(ent.label_)))
```

### Beyond the Vocabulary: Word Vectors, Bags and Sequences

The vocabulary is basically a bidirectionally indexed list of words/tokens; bidirectional, because we can get from and index its text form and vice versa.

With the vocabulary defined, we can represent each word in two forms:

1. As **one-hot encoded sparse vectors** of size `(n,)`, being `n` the number of words in the vocabulary.
2. As compressed vectors from an **embedding** of size `(m,)`, with `m < n`.

In a sparse representation, a word is a vector of zeroes except in the index which corresponds to the text form in the vocabulary, where the vector element value is 1.

In a compressed representation we don't have sparse one-hot encoded vectors, but vectors of much smaller sizes that contain float values in all their cells. We can create those vector spaces in different ways; one option is to compress sparse vectors to a latent space (e.g., with an autoencoder):

`[0, 1, 0, ..., 0] (n: vocabulary size) -> [0.2, 0.5, ..., 0.1] (m: latent word vector space)`

Typical sizes are `n = 70,000`, `m = 300`.

There are also **semantic embeddings**, which preserve the meaning relationships of words encoded as spatial relationships in the embedding spaces; thus, we can perform algebraic operations on the words:

`vector(queen) ~ vector(king) - vector(man) + vector(woman)`

Word vector similarities can be measured with their cosine similarity.

On the other hand, texts formed by words can be represented in the following general forms:

1. As **bags of words** (one-hot encoded).
2. As **sequences of word vectors** (usually compressed vectors).

The bag of words representation does not take into account the order in which the words appear in the text; with it we create a **Document Term Matrix** (DTM) of size `N x n`, being `N` the number of documents/texts and `n` the size of the vocabulary. In the cells of the DTM we can have:

- whether a word appears in a document/text,
- how many times it appears in that document/text,
- or the frequency of the word in that document considering also the rest of the documents.

![Document Term Matrix](./assets/document_term_matrix.png)

Typically, the latter option works best. It consists in weighting the term frequency in each text with the inverse frequency of appearance in all the texts: **TF-IDF, Term Frequency, Inverse Document Frequency**. This approach takes into account the importance of a word in the whole corpus, i.e., the complete text dataset. As such, stop words get a lower weight, thus less importance.

The TFIDF formulas are as follows (although everything is automatically computed):

- $\textrm{tfidf}(t,d,D) = \textrm{tf}(t,d) \cdot \textrm{idf}(t,D)$

- $\textrm{tf}(t,d) = \textrm{count}(t \in d)$: `count (term t in document d)`

- $\textrm{idf}(t,D) = \log \frac{N}{| \{d \in D \, : t \in d \} |}$: `log (total documents N / documents which contain term t)`

Sequences of word vectors are typically used with neural networks, either Recurrent Neural Networks (RNN) or Transformers.

### Vectorization with Scikit-Learn

In the following, a very simple text vectorization snippet is shown with [Scikit-Learn](https://scikit-learn.org/stable/); in it, texts are vectorized as bags of words. Note that:

- Texts are transformed into feature vectors stacked in the rows, as it is always done with the `X` matrix in other cases.
- We don't need to create any vocabulary with this approach, it is done automatically.
- We can use the produced `X_train_tfidf` in any classifier!

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('./data/sms_spam.csv')
# label: 1 (spam) / 0 (not spam)
# message: SMS texts that can be spam

from sklearn.model_selection import train_test_split

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=42)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape # (3733, 7082)

tfidf_vect = TfidfVectorizer()
X_train_tfidf = tfidf_vect.fit_transform(X_train)
X_train_tfidf.shape # (3733, 7082)
```

### Sentiment Analysis: A Binary Classification Problem

## Recurrent Neural Networks: General Notes

:construction: To be done.

### Quick Usage Examples of LSTMs in Pytorch

## List of Examples + Description Points

:construction: To be done.

## Improvements, Next Steps

:construction: To be done.

## Interesting Links

- [My notes and code](https://github.com/mxagar/deep_learning_udacity) on the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101).
- [NLP Guide](https://github.com/mxagar/nlp_guide).
- [Emotion Classification](https://en.wikipedia.org/wiki/Emotion_classification).

## Authorship

Mikel Sagardia, 2022.  
No guarantees.

You are free to use this project, but please link it back to the original source.