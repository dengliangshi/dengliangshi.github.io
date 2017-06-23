---
layout: post
title: Sampling approximation of gradient.
abstract: Sampling approximation of gradient is a speed-up technique for trainging neural netowrk language models, and is proposaled by Bengio and Senecal. Three algorithms are represented by Bengio and Senecal, but only the importance sampling method worked finely for neural network language models. This post mainly focuses on improtance sampling, and converys it in a simpler and easier way.
---

#### 0. INTRODUCTION

Inspired by the contrastive divergence model ([Hinton, 2002](http://www.cs.toronto.edu/~fritz/absps/nccd.pdf)), [Bengio and Senecal (2003)](http://www.iro.umontreal.ca/~lisa/bib/pub_subject/language/pointeurs/submit_aistats2003.pdf) proposed a sampling-based method to speed up the training of NNLMs. 

#### 1. PRELIMINARIES

NNLMs can be treated as a special case of energy-based probability models:

$$
P(w_t{\mid}h_t)\;=\;\frac{e^{-y(w_t,\;h_t)}}{\sum_{i=1}^{V}e^{-y(w_i,\;h_i)}}
$$

where, $$w_t$$ is the target word, $$h_t$$ is the previous context of word $$w_t$$, $$y(w_i,\;h_i)$$ is the output of neural network model, 

The parameters of neural network and feature vectors for words are learned using stochastic gradient descent method, and the computation of the partition function is quite expensive. The idea of sampling based method is to approximate the gradient of the log-likelihood with repect to the parameters $\theta$ based on sampling method rather than computing the gradient explicitly. The log-likelihood gradient for the parameters set $$\theta$$ can be generally represented as the sum of two parts: positive reinforcement for word $$w_t$$ and negative reinforcement for all word $$w_i$$, weighted by $$P(w_i{\mid}h_t)$$, $$i = 1, 2, ..., V$$:

$$
\frac{\partial{\textrm{log}P(w_t{\mid}h_t)}}{\partial{\theta}}\;=\;-\frac{\partial{y_t}}{\partial{\theta}}\;+\;\sum_{i=1}^{V}P(w_i{\mid}h_i)\frac{\partial{y_i}}{\partial{\theta}}
$$

The detail inference process to get the above equation is as follows:

$$
\begin{align*}
\textrm{log}P(w_t{\mid}h_t)&=-y(w_t,\;h_t)\;-\;\textrm{log}\sum_{i=1}^{V}e^{-y(w_i,\;h_i)}\\
\frac{\partial{\textrm{log}P(w_t{\mid}h_t)}}{\partial{\theta}}&=-\frac{\partial{y(w_t,\;h_t)}}{\partial{\theta}}\;-\;\frac{\partial{\textrm{log}\sum_{i=1}^{V}e^{-y(w_i,\;h_i)}}}{\partial{\theta}}\\
&=-\frac{\partial{y(w_t,\;h_t)}}{\partial{\theta}}\;-\;\frac{1}{\sum_{j=1}^{V}e^{-y(w_j,\;h_j)}}\frac{\partial{\sum_{i=1}^{V}e^{-y(w_i,\;h_i)}}}{\partial{\theta}}\\
&=-\frac{\partial{y(w_t,\;h_t)}}{\partial{\theta}}\;+\;\frac{1}{\sum_{j=1}^{V}e^{-y(w_j,\;h_j)}}\sum_{i=1}^{V}e^{-y(w_i,\;h_i)}\frac{\partial{y(w_i,\;h_i)}}{\partial{\theta}}\\
&=-\frac{\partial{y(w_t,\;h_t)}}{\partial{\theta}}\;+\;\sum_{i=1}^{V}\frac{e^{-y(w_i,\;h_i)}}{\sum_{j=1}^{V}e^{-y(w_j,\;h_j)}}
\frac{\partial{y_i}}{\partial{\theta}}\\
&=-\frac{\partial{y(w_t,\;h_t)}}{\partial{\theta}}\;+\;\sum_{i=1}^{V}P(w_i{\mid}h_t)\frac{\partial{y_i}}{\partial{\theta}}
\end{align*}
$$

#### 2. SAMPLING APPROXIMATIONS OF THE LOG-LIKELIHOOD GRADIENT

the weighted negative part is replaced by sampling approximations, and three algorithms for sampling approximations of the log-likelihood gradient are presented by \citet{bengio_2003b}: <strong>Monte-Carlo Algorithm, Independent Metropolis-Hastings Algorithm, Importance Sampling Algorithm</strong>. Monte-Carlo algorithm is not feasible beacuse Monte-Carlo approximations cannot be cheaply performed with training samples. One solution is to using a proposal distribution $Q$, like n-grams or markov chain. When a Monte-Carlo Markov Chain (MCMC) is adopted as a proposal distribution, it will be the Independent Metropolis-Hastings Algorithm. But the convergence of the Markov chain is is a problem and performance of neural network language models using Metropolis-Hastings algorithm is very poor. With importance sampling, 19-fold speed-up is achieved while the perplexities on training data set and test data set is almost the same as the nerual network language model trained with exact and complete gradient ([Bengio and Senecal, 2003](http://www.iro.umontreal.ca/~lisa/bib/pub_subject/language/pointeurs/submit_aistats2003.pdf)). So, only the detail procedure of importance sampling will be introduced here.

Importance sampling is a Monte-Carlo scheme which uses an existing proposal distribution to approximate the gradient. 


then the gradient estimator becomes:

$$
\frac{\partial{\textrm{log}P(w_t|h_t)}}{\partial{\theta}}\;=\;-\frac{\partial{y_t}}{\partial{\theta}}\;+\;\frac{\sum_{\hat{w}\in{W}}\frac{\partial{y(w_t|h_t)}}{\partial{\theta}}e^{-y(\hat{w}|h_t)}/Q(\hat{w}|h_t)}{\sum_{\hat{w}\in{W}}e^{-y(\hat{w}|H-t)}/Q(\hat{w}|h_t)}
$$

Calculate the positive contribution of the gradient;
$N$ samples from the proposal distribution $Q$, 

The sample size should increase as training progresses to avoid divergence, the adaptive sample size algorithm:

$$
S = \frac{(\sum_{j=1}^{N}r_j)^2}{\sum_{j=1}^{N}r_j^2}
$$

As more efficent and easy speed up techique has been proposed now, just posted here for completeness and no futher studies will be performanced.
