# Text Sentiment: A Collection of Notes and Examples

This repository collects/links projects related to text sentiment analysis, as well as some high level notes on the techniques used in the field.

:warning: Important notes, first of all:

[![Unfinished](https://img.shields.io/badge/status-unfinished-orange)](https://shields.io/#your-badge)

- This is a document for my future self.
- This is an on-going project; I will extend the content as far as I have time for it.
- In some cases, I will use the code from other public tutorials/posts rightfully citing the source.
- In addition to the examples, the links in the section [Interesting Links](#interesting-links) are very useful resources for those interested in the topic.

## Introduction

From a pragmatic perspective, [sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis) is one of the most practical topics in Natural Language Processing (NLP) for me at the moment, because the techniques used in it are able to efficiently condense raw sequences of words to one valuable scalar *automatically*. Then, that scalar can become an important feature in more complex models that output important business insights.

## How to Use This

1. Go to the desired example folder from the section [List of Examples + Description Points](#list-of-examples--description-points). You should have brief instructions in each folder.
2. If there is an `Open in Colab` button anywhere, you can use it :smile:.
3. If you'd like to run the code locally on your device install the [dependencies](#dependencies) and run the main file in it; often, the main file will be a notebook that takes care of all.

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
## Recurrent Neural Networks: General Notes

:construction: To be done.

## List of Examples + Description Points

:construction: To be done.

## Improvements, Next Steps

:construction: To be done.

## Interesting Links

:construction: To be done.

## Authorship

Mikel Sagardia, 2022.  
No guarantees.

You are free to use this project, but please link it back to the original source.