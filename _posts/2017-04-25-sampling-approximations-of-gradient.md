---
layout: post
title: Sampling approximation of gradient.
abstract: Sampling approximation of gradient is a speed-up technique for trainging neural netowrk language models, and is proposaled by Bengio and Senecal. Three algorithms are represented by Bengio and Senecal, but only the importance sampling method worked finely for neural network language models. This post mainly focuses on improtance sampling, and converys it in a simpler and easier way.
---

#### 0. INTRODUCTION
Inspired by the contrastive divergence model ([Hinton, 2002](http://www.cs.toronto.edu/~fritz/absps/nccd.pdf)), [Bengio and Senecal (2003)](http://www.iro.umontreal.ca/~lisa/bib/pub_subject/language/pointeurs/submit_aistats2003.pdf) proposed a sampling-based method to speed up the training of NNLMs. The parameters of neural network and feature vectors for words are learned using stochastic gradient descent method, and the computation of the partition function is quite expensive. The idea of sampling based method is to approximate the gradient of the log-likelihood with repect to the parameters set $\theta$ based on sampling method rather than computing the gradient explicitly.

#### 1. PRELIMINARIES
In all NNLMs, a softmax layer is used to regularize the outputs, and NNLMs can be treated as a special case of energy-based probability models:

$$
P(w_t{\mid}h_t)\;=\;\frac{e^{-y(w_t,\;h_t)}}{\sum_{i=1}^{N_v}e^{-y(w_i,\;h_t)}}
$$

where, $$w_t$$ is the target word, $$h_t$$ is the previous context of target word, $$N_v$$ is the size of vocabulary, and $$y(w_i,\;h_t)$$, $$i=1, 2, \dots, N_v$$ is the output of neural network model.

NNLMs are commonly trained using stochastic gradient descent (SGD) to maximum the log-likelihood of training data set, and the log-likelihood gradient for the parameters set $$\theta$$ should be calculated during training which can be generally represented as the sum of two parts: positive reinforcement for word $$w_t$$ and negative reinforcement for all word $$w_i$$, weighted by $$P(w_i{\mid}h_t)$$, $$i = 1, 2, \dots, N_v$$:

$$
\frac{\partial{\textrm{log}P(w_t{\mid}h_t)}}{\partial{\theta}}\;=\;-\frac{\partial{y(w_t,\;h_t)}}{\partial{\theta}}\;+\;\sum_{i=1}^{N_v}P(w_i{\mid}h_t)\frac{\partial{y(w_i,\;h_t)}}{\partial{\theta}}
$$

The detail inference process to get the above equation is as follows:

$$
\begin{align*}
\textrm{log}P(w_t{\mid}h_t)&=-y(w_t,\;h_t)\;-\;\textrm{log}\sum_{i=1}^{N_v}e^{-y(w_i,\;h_t)}\\
\frac{\partial{\textrm{log}P(w_t{\mid}h_t)}}{\partial{\theta}}&=-\frac{\partial{y(w_t,\;h_t)}}{\partial{\theta}}\;-\;\frac{\partial{\textrm{log}\sum_{i=1}^{N_v}e^{-y(w_i,\;h_t)}}}{\partial{\theta}}\\
&=-\frac{\partial{y(w_t,\;h_t)}}{\partial{\theta}}\;-\;\frac{1}{\sum_{j=1}^{N_v}e^{-y(w_j,\;h_t)}}\frac{\partial{\sum_{i=1}^{N_v}e^{-y(w_i,\;h_t)}}}{\partial{\theta}}\\
&=-\frac{\partial{y(w_t,\;h_t)}}{\partial{\theta}}\;+\;\frac{1}{\sum_{j=1}^{N_v}e^{-y(w_j,\;h_t)}}\sum_{i=1}^{N_v}e^{-y(w_i,\;h_t)}\frac{\partial{y(w_i,\;h_t)}}{\partial{\theta}}\\
&=-\frac{\partial{y(w_t,\;h_t)}}{\partial{\theta}}\;+\;\sum_{i=1}^{N_v}\frac{e^{-y(w_i,\;h_t)}}{\sum_{j=1}^{N_v}e^{-y(w_j,\;h_t)}}\frac{\partial{y(w_i,\;h_t)}}{\partial{\theta}}\\
&=-\frac{\partial{y(w_t,\;h_t)}}{\partial{\theta}}\;+\;\sum_{i=1}^{N_v}P(w_i{\mid}h_t)\frac{\partial{y(w_i,\;h_t)}}{\partial{\theta}}
\end{align*}
$$

With sampling method, the second part of above log-likelihood gradient, negative reinforcement for words, will be approxiamted, and three approximation algorithms are invested by [Bengio and Senecal (2003)](http://www.iro.umontreal.ca/~lisa/bib/pub_subject/language/pointeurs/submit_aistats2003.pdf): <strong>Monte-Carlo Algorithm, Independent Metropolis-Hastings Algorithm, Importance Sampling Algorithm</strong>. Monte-Carlo algorithm is not feasible beacuse Monte-Carlo approximations cannot be cheaply performed with training samples. One solution is to using a proposal distribution $Q$, like n-grams or markov chain. When a Monte-Carlo Markov Chain (MCMC) is adopted as a proposal distribution, it will be the Independent Metropolis-Hastings Algorithm. But the convergence of the Markov chain is is a problem and performance of neural network language models using Metropolis-Hastings algorithm is very poor. With importance sampling, 19-fold speed-up is achieved while the perplexities on training data set and test data set is almost the same as the nerual network language model trained with exact and complete gradient ([Bengio and Senecal, 2003](http://www.iro.umontreal.ca/~lisa/bib/pub_subject/language/pointeurs/submit_aistats2003.pdf)). Therefore, only the detail procedure of importance sampling will be introduced here.

#### 2. IMPORTANCE SAMPLING ALGORITHM
Importance sampling is a Monte-Carlo scheme, which can be used when an existing distribution is adopted to approximate gradient. For the following gradient:

$$
G(Y)\;=\;\sum_{y}P(y)g(\hat{y})
$$

its improtance sampling estimator is:

$$
E_p(G(Y))\;=\;\frac{1}{N_s}\sum_{\hat{y}}\frac{P(\hat{y})}{Q(\hat{y})}g(\hat{y})
$$

where $$N_s$$ is the size of samples.

Appling importance sampling estimator to the gradient of negative samples in NNLMs:

$$
\sum_{i=1}^{N_v}P(w_i{\mid}h_t)\frac{\partial{y(w_i,\;h_t)}}{\partial{\theta}}\;=\;\frac{1}{N_s}\sum_{i=1}^{N_v}\frac{P(w_i{\mid}h_t)}{Q(w_i{\mid}h_t)}\frac{\partial{y(w_i,\;h_t)}}{\partial{\theta}}
$$

However, in order to speed up traing of NNLMs, the calcualtion of $$P(w_i{\mid}h_t)$$ should be avoided. Covert $$P(w_i{\mid}h_t)$$ into：

$$
P(w_i{\mid}h_t)\;=\;\frac{e^{-y(w_t,\;h_t)}}{\sum_{i=1}^{N_v}e^{-y(w_i,\;h_t)}}=\frac{e^{-y(w_t,\;h_t)}}{Z}
$$

and $$Z$$ is itself is a average with uniform distribution $$1/N_v$$:

$$
Z = \sum_{i=1}^{N_v}e^{-y(w_i,\;h_t)} = N_v\sum_{i=1}^{N_v}\frac{1}{N_v}e^{-y(w_i,\;h_t)}
$$

so $$Z$$ can be estiamted by importence sampling using a proposal distribution $$Q$$ too:

$$
Z\;=\;\frac{N_v}{N_s}\sum_{i=1}^{N_v}\frac{e^{-y(w_i,\;h_t)}}{N_vQ(w_i{\mid}h_t)}\;=\;\frac{1}{N_s}\sum_{i=1}^{N_v}\frac{e^{-y(w_i,\;h_t)}}{Q(w_i{\mid}h_t)}
$$

then the overall gradient estimator for example $$(w_t, h_t)$$ is:

$$
\frac{\partial{\textrm{log}P(w_t|h_t)}}{\partial{\theta}}\;=\;-\frac{\partial{y_t}}{\partial{\theta}}\;+\;\frac{\sum_{\hat{w}\in{W}}\frac{\partial{y(w_t|h_t)}}{\partial{\theta}}e^{-y(\hat{w}|h_t)}/Q(\hat{w}|h_t)}{\sum_{\hat{w}\in{W}}e^{-y(\hat{w}|H-t)}/Q(\hat{w}|h_t)}
$$

The sample size should increase as training progresses to avoid divergence, the adaptive sample size algorithm:

$$
S = \frac{(\sum_{j=1}^{N}r_j)^2}{\sum_{j=1}^{N}r_j^2}
$$

where $$r_j\;=\;P(w_j)/Q(w_j)$$ is the importance sampling ratio for the j-th sampling, and is estimated as:

$$
r_j\;\approx\;\frac{e^{y()}/Q()}{\sum_{}^{}e^{-y()}/Q()}
$$

sampling by 'blocks' with sie $N_b$ until the effective sample size $S$ becomes greater than a minimum value $N_0$, and do full back-progration when the samping size is greater than a threshold.

The detail procedure of importance sampling are as following:
calculate the positive contribution of the gradient;
$N$ samples from the proposal distribution $Q$;

As more efficent and easy speed up techiques have been proposed now, just posted here for completeness and no futher studies will be performanced.

<img src="/images/signature.png" align="right">