---
layout: post
title: Hierarchical neural network language model
abstract: In hierarchical neural network language model, words are classified into several word classes according to their freqency. There are two output layers for words and classes respectively. The algorithms for assigning words with classes are also included.
---


### 1. Introduction

the frequency of each word learned from training data is $$f_i$$, all words in vocabulary are sorted in descend order according to their frequency, and all words will be classified into $$n$$ classes.


### 2. Algorithms for Class Assignment

#### 2.2 Algorithm 2:
calcuate the total frequency of all words:

$$F = \sum_{i=1}^{V}f_i$$

then assign class index to words one by one according to accumulation probabilty of each word. For the $$t$$th word, if its accumulation probability satisfies the following condition:

$$\frac{i}{n}<\sum_{j=1}^{t}f_j\leq\frac{i+1}{n}$$

its class index will be set as $$i$$.

#### 2.3 Algorithm 3:

calcuate the total frequency of all words:

$$F = \sum_{i=1}^{V}f_i$$

then calculate the summary of the square probability of all words:

$$dF = \sum_{i=1}^{V}\sqrt{\frac{f_i}{F}}$$

then assign class index to words one by one according to accumulation probabilty of each word. For the $$t$$th word, if its accumulation probability satisfies the following condition:

$$\frac{i}{n}<\sum_{j=1}^{t}\frac{\sqrt{f_t/F}}{dF}\leq\frac{i+1}{n}$$

its class index will be set as $$i$$.
