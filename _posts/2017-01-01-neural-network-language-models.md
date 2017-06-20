---
layout: post
title: Neural Network Languge Models
abstract: Different architectures of neural network language models are decribed in this post, including feedforward neural network language model, recurrent neural network model and its long short term memory version. Variants of these nerual network language models will also be discussed, like using different activation functions, with or without direct connections between input and output layers and adopting bias terms or not. General issues about training and testing neural network language models are also included in this post.
---

### 1. Introduction
Generally, well designed language model (LM) makes a critical difference in various natural language processing (NLP) tasks, including speech recognition, machine translation, semantic extraction and etc. Language modeling, therefore, has been the research focus in NLP field all the time, and a large number of sound research results have been published for decades. N-gram based language modeling, a non-paramtric approach, is used to be state of the art, but now a paramtric approach, neural network language modeling is considered to show the best performance, and become the most commonly used language modeling technique.

Although some previous attempts $$\citep{miikk_1991,schmidhuber_1996,xu_2000}$$, details on which are provided in \citet{schwenk_2007}, have been made to apply artificial neural network into language modeling, neural network language modeling has attracted researches' attention only after \citet{bengio_2001, bengio_2003a}. Neural network language models did not show prominent advantage over other language models until Mikolov investigates recurrent neural network for language modeling \citet{mikolov_2010, mikolov_2011, mikolov_2012}. After more than a decade's research, there are imporvements, marginal or critical, over NNLMs. However, as metioned in \citet{goodman_2001}, these techniques should be studied together and language model built with the combination of all these techniques may show better performance than models with one or several of them. Furthermore, some techniques are raised to solve the same problems of NNLMs, but the results are not comparable because of different experimental setups. Another significant problem is that while most studies focus on the achievements of NNLMs, the limits of NNLMs are seldom studied thoroughly. \citet{jozefowicz_2016} attempts to explore the limits of neural network language modeling, but they only try to deal with two problems of training NNLMs: corpus and vocabulary sizes rather than presenting an exhausted explorations on the limits of NNLMs.

### 2. Neural Network Language Models, NNLMs
The goal of neural network language models are to learn the probability function of word sentences in a language, therefore they are also termed as neural probabilistic lanuage models or neural statistic lanuage models. The probability of a word sequence $$w_1w_2...w_k$$ can be represented by the production of the conditional probability of each word given all the previous ones:

$$P\;=\;\prod^{k}_{t=1}P(w_t|w_1...w_{t-1})$$

Statistic language models all take this chain rule as base. [Any Assumption?]

#### 2.1 Feedforward Neural Network Language Model, FNNLM
The architecture of the original feedforward neural network language model (FNNLM) in \citet{bengio_2001, bengio_2003a} is showed as in Figure 1. In Bengio's model, a vocabulary is pre-built from a training data set, and the order of words in it is fixed. So each word in this vocabulary has a unique index, just like a fixed order word list. The object of FNNLM is to evaluate the probabilty $$P(w_t|w_{1}...w_{t-1})$$, but, for lack of an effective representation of history, FNNLM follows the assumption of N-gram approach that the probabilty of a word in a word sequence depends on only the direct $$(n-1)$$ predecessor word:

$$P(w_t|w_1...w_{t-1})\;\approx\;P(w_t|w_{t-n+1}...w_{t-1})$$

For word $$w_t$$ in word sequence $$w_1w_2...w_k$$, its direct previous words $$w_{t-n+1},...,w_{t-1}$$ are used to predict its conditional probabilty. In FNNLM, words $$w_{t-n+1},...,w_{t-1}$$ are projected linearly into feature vectors using a shared matrix $$C\in{\mathbb{R}^{V\times{m}}}$$ according to their index in the vocabulary, where $$V$$ is the size of the vocabulary and $$m$$ is the feature vector's size. In fact, each row of projection maxtrix $$C$$ is a feature vector of a word. The input $$x\in\mathbb{R}^{n_i}$$ of FNNLM is formed using the concatenation of the feacture vectors of words $$w_{t-n+1},...,w_{t-1}$$, where $$n_i\;=\;m\times(n-1)$$ is the size of input layer in feedforward neural network model. The feedforward neural network model can be repesented as:

$$y\;=\;V\cdot{f(U\cdot{x}\;+b)}\;+\;M\cdot{x}\;+\;d$$

Where, $$U\in\mathbb{R}^{n_h\times{n_i}}$$,  $$V\in\mathbb{R}^{n_o\times{n_h}}$$ are weight matrixes, $$n_h$$ is the number of nodes in hidden layer, $$n_o=V$$ is the size of output layer, weight matrix $$M\in\mathbb{R}^{n_o\times{n_i}}$$ is for the direct connection between input layer and output layer, $$b\in\mathbb{R}^{n_h}$$ and $$d\in\mathbb{R}^{n_o}$$ are biase vectors in hidden layer and output layer respectively, $$y\in\mathbb{R}^{n_o}$$ is the output vector, and $$f(\cdot)$$ is activation function.

The $$i$$th element of output vector $$y$$ is the unnormalized probability $$P(w_i|w_{t-n+1}...w_{t-1})$$, where $$w_i$$ is the $$i$$th word in the vocabulaury. In order to guarantee all the output probailities are positive and summing to 1, a softmax layer is always adopted following the output layer of feedforward nerual network:

$$f(y_i) = \frac{e^{y_i}}{\sum_{j=1}^{V}e^{y_j}}, i = 1, 2, ..., V$$

where $$y_i\;(i\;=\; 1, 2, ..., V)$$ is $i$th element of the ouput vecotor $$y$$.

<div style="text-align: center;">
<img width="500" src="/images/nnlms/fnnlm.png">
</div>

#### 2.2 Recurrent Neural Network Language Model, RNNLM
The idea of applying recurrent neural network (RNN) model into language modeling was proposaled much earlier \citep{bengio_2003a, castro_2003}, but fisrt attempt to build recurrent nerual network based language model was made by \citet{mikolov_2010, mikolov_2011}. Recurrent neural networks are fundamentally different from feedforward architectures in the sense that they not only operate on an input space but also on an internal state space, and the state space enables the representation of sequentially extended dependencies. Arbitrary length of history . The general architecture of recurrent neural network language model (RNNLM) can be represented as Figure 2. The representation of words in RNNLM is the same as that of FNNLM, but the input of recurrent neural network at each step is the feature vector of a direct preivous word instead of the concatenation of the $n-1$ previous words' feature vectors and all other previous words are taken into account by the internal state of previous step. At step $$t$$, recurrent neural network is represented as:

$$s_t\;=\;f(Ux_t\;+\;Ws_{t-1}\;+\;b)$$

$$y_t\;=\;Vs_t\;+\;Mx_t\;+\;d$$

where, weight matrix $$W\in\mathbb{R}^{n_h\times{n_h}}$$. The output of recurrent neural network are also unnormalized probabilities and should be regularized by a softmax layer.

<div style="text-align: center;">
<img src="/images/nnlms/rnnlm.png">
</div>

Although above RNN can deal with all of the predecessor words to predict next word in a word sequence, but it is quit difficult to train over long term dependencies because of the vanishing problem. Long-Short Term Memoery (LSTM) RNN was raised to solve this problem, and LSTM-RNN was introduce to language modeling by \citet{sunder_2012}. LSTM-RNN was introduced by \citet{hochreiter_1997} and were refined and popularized by many people in following work \citep{gers_2000, cho_2014b}. The general :

$$i_t\;=\;\sigma(U_ix_t\;+\;W_{i}s_{t-1}\;+\;V_{i}c_{t-1}\;+\;b_i)$$

$$f_t\;=\;\sigma(U_{f}x_t\;+\;W_{f}s_{t-1}\;+\;V_{f}c_{t-1}\;+\;b_f)$$

$$o_t\;=\;\sigma(U_{o}x_t\;+\;W_{o}s_{t-1}\;+\;V_{o}c_{t-1}\;+\;b_o)$$

$$g_t\;=\;f(Ux_t\;+\;Ws_{t-1}\;+\;Vc_{t-1}\;+\;b)$$

$$c_t\;=\;f_t\;*\;c_{t-1}\;+\;i_t\;*\;g_t$$

$$s_t\;=\;o_t\;*\;f(c_t)$$

$$y_t\;=\;Vs_t + Mx_t + d$$

Where, $$i_t\in\mathbb{R}^{n_h}$$, $$f_t\in\mathbb{R}^{n_h}$$, $$o_t\in\mathbb{R}^{n_h}$$ are input gate, forget gate and output gate, respectively. $$c_t\in\mathbb{R}^{n_h}$$ is previous state of nodes. $$U_i$$, $$U_f$$, $$U_o$$, $$U\in\mathbb{R}^{n_h\times{n_i}}$$, $$W_i$$, $$W_f$$, $$W_o$$, $$W\in\mathbb{R}^{n_h\times{n_i}}$$, $$V_i$$, $$V_f$$, $$V_o$$, $$V\in\mathbb{R}^{n_h\times{n_i}}$$ are all weight matrixes. $$b_i$$, $$b_f$$, $$b_o$$, $$b\in\mathbb{R}^{n_h}$$, and $$d\in\mathbb{R}^{n_o}$$ are bias vectors. $$f(\cdot)$$ is the activation function in hidden layer and $$\sigma(\cdot)$$ is the activation function for gates.

### 3. Training
Training of NNLMs is achieved by maximizing the log-likelihood of the training data or a regularized criterion, e.g. by adding a weight decay penalty, and the object function is:

$$L = \frac{1}{T}\sum_{t=1}^{T}\textrm{log}(P(w_t, w_{t-1}, ..., w_{t-n+1}; \theta))\;+\;R(\theta)$$

where, $$\theta$$ is the set of parameters of NNLMs, $$R(\theta)$$ is a regularization term.

Cross entropy criterion is used

$$L = \frac{1}{T}\sum_{i=1}^{T}o^{^}log{o}$$

For word $$w_t$$, the cross entropy is $$L_t = o^{^}log{o}$$ and the except output $$o^{^}$$ is 1 of $$V$$ vector, and only the  the gradient for the output of neural network is

$$\frac{\partial{L_t}}{\partial{y_i}} = \frac{\partial{o^{^}log{o}}}{\partial{y_i}} = 1 - o_i$$

Stochastic gradient descent (SGD) method is a proper learning algorithm for training all NNLMs, and the gradients of errors for parameters $\theta$ are computed using backpropagation (BP) algorithm. For RNNLMs, backpropagation through time (BPTT) \citep{rumelhart_1986} should be used for better performance, and \citep{mikolov_2012} reported that error gradients were computed through 5 time steps is enough, at least for simple RNNLM on small corpus. The parameters of NNLMs are updated as:

$$\theta = \theta + \alpha\frac{\partial{L}}{\partial{\theta}} - \beta\theta$$

where, $$\alpha$$ is learning rate and $$\beta$$ is regularization parameter.

As metioned above, two corpora are chosen for experiments in this paper, Brown corpus and One Billion Word Benchmark. They are all frequenctly used corpora for studies on language modeling and avaliable for everyone freely. Experimental setup for Brown corpus is the same as that in \citep{bengio_2003a}, the first 800000 words (ca01$$\sim$$cj54) were used for training, the following 200000 words (cj55$$\sim$$cm06) for validation and the rest (cn01$$\sim$$cr09) for testing.

### 4. Evaluation
The performance of neural network language models are usually measured by perflexity (PPL) which corresponds to the cross-entropy between the language model and test data. The perflexity of word sequence $$\textbf{w}=[w_0, w_1, ..., w_K]$$ is defined as:

$$PPL\;=\;\sqrt[K]{\prod^{K}_{i=0}\frac{1}{P(w_i|w_0...w_{i-1})}}\;=\;2^{-\frac{1}{K}\sum^{K}_{i=0}log_2P(w_i|w_0...w_i{i-1})}$$

Perflexity can be defined as the exponential of the average number of bits required to encode the test data using a language model and lower perflexity indicates that the language model is closer to the true model which generates the test data.

### 5. Comparison of NNLMs with Different Architectures
Comparision among diffrent types of NNLMs have already been made on both small corpus and large ones \citep{mikolov_2012, sunder_2013}. The results show that RNNLMs outperformed FNNLMs and the best performance is achieved using LSTM-RNNLMs. However, one or more optimization techniques are adopted for the NNLMs used in these comparision, which makes the result not suitable for further analysis. So the standard version of the three types of NNLMs are tested on Brown Corpus and One Billion Word Benchmark (OBWB) \citep{chelba_2014} here, and a class based speed-up technique is used and the algorithm, the detail about this technique will be discussed later. The reuslts are showed in Table 1 and will be used as the baseline in this paper.

LM     | n |   m   | n_h     | Direct | Bias | Brown | OBWB 
-------|---|-------|---------|--------|------|-------|-------
FNNLM  | 5 |  100  |   50    |   No   |  No  | 73.2  |  1
RNNLM  | - |  100  |   50    |   No   |  No  | 73.2  |  2
LSTMLM | - |  100  |   50    |   No   |  No  | 73.2  |  3

The NNLMs performed in above experiments are all without direct connections from input layer to output layer and biases in both hidden and output layer. \citet{bengio_2003a} suggests that better generalization and lower perplexity with no direct connections but longer training. A reasonable interpretation is that direct input-to-output connections provide a bit more capacity and faster learning of the "linear" part of mapping from word features to log-probabilities. On the other hand, without those connections the hidden units form a tight bottleneck which might force better generalization. For the biases, \citet{mikolov_2012} reported that no significant improvement of performance was gained with biases. So no direct connections and biases will be used in the following experiments neither.

### 6. Conclusion

### References