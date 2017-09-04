---
layout: post
title: Class based neural network language model
abstract: The idea of word classes has been used extensively in language modeling for improving perplexities or increasing speed. In this post, some researches about introducing word classes into neural network language modeling will be described, and a extension of word classes, hierarchical neural network language model, will also be included.
---

#### 1. INTRODUCTION
The amount of computation required for neural network language models mainly lies in their output layers in which the number of nodes is equal to the size of vocabulary, usually serval ten thousands. if the size of output layer can be reduced, the amount of computation will decreased. Hierarchical neural network language modeling is proposed by [Bengio et al. (2005)](http://www.gatsby.ucl.ac.uk/aistats/fullpapers/208.pdf) as a speed-up technique for both training and test of neural network language model, by which the size of output layer is reduced significantly. The idea of hierarchical neural network language modeling is original from class-based language models ([Goodman, 2001](http://www2.denizyuret.com/ref/goodman/goodman2001.pdf)). In hierarchical neural network language models (Figure 1), words in vocabulary are classified into different classes, and the conditional probability of a word $$w_t$$ in word sequence $$w_1w_2{\dots}w_T$$ is represented as:

$$
P(w_t{\mid}w_1w_2{\dots}w_{t-1})\;=\;P(w_t{\mid}w_1w_2{\dots}w_{t-1}, c_i)P(c_i{\mid}w_1w_2{\dots}w_{t-1})
$$

where $$c_i$$ is the word class $$w_t$$ blongs to. The hierarchical nerual network language model has two output layers: one for the probability of words and the other for the probability of classes. At each step, only the probability of words which share the same class with target word and probability of all classes will be calculated. By this way, the time for both training and test will decrease greatly.

<div class="thumbnail">
    <img src="/images/hnnlms/rnnlm-class.png">
    <div class="caption">
        <p class="text-center">Figure 1. Architecture of hierarchical neural network language model</p>
    </div>
</div>

#### 2. ALGORITHMS FOR CLASS ASSIGNMENT
One of the key points to build a hierarchical neural network langauge model is to classify words into classes, and several algorithms have been raised for this. In [Bengio et al. (2005)](http://www.gatsby.ucl.ac.uk/aistats/fullpapers/208.pdf), words are clustered using WordNet. [Mikolov et al. (2011)](http://mirlab.org/conference_papers/International_Conference/ICASSP%202011/pdfs/0005528.pdf) proposed two algorithms to classify words according to their frequencies in training data set.

#### 2.1. Algorithm 1
The simple way to assign word classes is to cluster words uniformly and randomly. 

#### 2.2. Algorithm 2
When building vocabulary from training data set, the frequency of each word in training data set is counted. The sum of all words' frequenies is:

$$
F = \sum_{i=1}^{K}f_i
$$

where, $$K$$ is the size of vocabulary, $$f_i\;(i=1,2,\dots, K)$$ is the frequency of each word. Before classifing words into classed, all the words are sorted in descend order according to their frequencies. Then assign class index to words one by one according to the accumulation probabilty of each word. For the $$t$$th word, it will be classified into $$i$$th class if its accumulation probability satisfies the following criterion:

$$\frac{i}{N}<\sum_{j=1}^{t}f_j\leq\frac{i+1}{N}$$

where $$N$$ is the total number of word classes whose optimal value is around $$\sqrt{K}$$.

The word classes established using this algorithm is not uniform. The words with higher frequencies are classified into smaller word classes, and words with lower frequencies are classified into larger ones. In this way, the amount of computation is reduced further.

#### 2.3. Algorithm 3
This algorithm is almost same as the previous one, The only difference is that square probability is used to  the sum of the square probability of all words is:

$$dF = \sum_{i=1}^{V}\sqrt{\frac{f_i}{F}}$$

For the $$t$$th word, it will be classified into word class $$i$$ if its accumulation square probability satisfies the following condition:

$$\frac{i}{n}<\sum_{j=1}^{t}\frac{\sqrt{f_j/F}}{dF}\leq\frac{i+1}{n}$$

With this algorithm, the difference among word classes is enlarged. The word classes of words with higher frequencies become smaller, and the ones of words with lower frequencies become bigger.

#### 3. COMPARISON OF DIFFERENT ALGORITHM
This speed-up technique can decrease the calculated amount of neural network language model, but it also casuses a bit of degradation in model's performance. In order to explore the advantages and disadvantages of above three algorithms for word class assignment, experiments are performed on hierarchical neural network language models with these algorithms. The experimental results are showed in Table 1:

Algorithm    | PPL    | Time/s
-------------|:------:|:-----:
Uniform      | 253.23 |  1
Algorithm 01 | 253.23 |  1
Algorithm 02 | 73.2   |  2
Algorithm 03 | 73.2   |  3

#### 4. CONCLUSION
