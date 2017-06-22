---
layout: post
title: An Optimization Approach to Back-propagate Errors
abstract: A new implemetation of BPTT algorithm is proposed in this post.
---

### 1. Introduction
Back-propagation through time (BPTT) algorithm is a gradients training recurrent neural network was first proposed by Rumelhart et al. (1986), and 

### 2. Optimization Approach 
For stardand recurrent neural network (RNN) model (Figure 1), it can be represented as:

$$y_t = Vs_t,\;s_t = f(Ux_t + Ws_{t-1})$$

<div style="text-align: center;">
<img src="/images/bptt/rnn.png">
</div>

At time step $$t$$, let the target output is $$y_{t}^{'}$$, then the error is:

$$e_t = y_{t}^{'} - y_t$$

Using BPTT algorithm, the error $$e_t$$ will be backpropagation throgh all the previous step. Take $$x_i$$ ($$0\leq{i}\leq{t}$$) as an example, the error gradient is:

$$\frac{\partial{e_t}}{\partial{x_i}}=\frac{\partial{e_t}}{\partial{s_i}}\frac{\partial{s_i}}{\partial{x_i}}$$

<div style="text-align: center;">
<img src="/images/bptt/error.png">
</div>

the final error gradient for $$x_i$$ is

$$\frac{\partial{E_i}}{\partial{x_i}}=\sum_{t=i}^{n}\frac{\partial{e_t}}{\partial{x_i}}$$

At each time step, the error should be backpropagation throngh all the previous time step. If the input sequence is long, the compution will be very expensive. But, with a careful observe, if 

$$\frac{\partial{E_i}}{\partial{x_i}}=\sum_{t=i}^{n}\frac{\partial{e_t}}{\partial{x_i}}=\frac{\partial{s_i}}{\partial{x_i}}\sum_{t=i}^{n}\frac{\partial{e_t}}{\partial{s_i}}$$

which can be rewrited as:

$$\frac{\partial{E_i}}{\partial{x_i}} = U^T\frac{\partial{E_i}}{\partial{s_i}} =  U^T(\frac{\partial{e_i}}{\partial{s_i}} + W^T\frac{\partial{E_{i+1}}}{\partial{s_i}})$$

it indicates that the error two part: the error from current output and the error from all late time step. Instead of propagation the error to all the previous time step, just accumulate the error gradient to the hidden state of directly previous step. 

The 

### 3. Comarsion
The computational complexity of previous implemetation of BPTT without truncation is $$O(n^2)$$, and $$n$$ is the length of input sequence. the proposed approach for BPTT is $$O(n)$$, which mean the new approach will be n times faster than preivous one. For truncated BPTT, like m steps the computational complexity will be $$O(nm)$$, and slower than the optimization method.

the performance, long-short term memeory (LSTM) long dependencies, but with truncation BPTT, the advantage of LSTM is limited. In this paper, 

Model | BPTT | PPL
------|------|----
RNN   | old  | 252

### 4. Conclusion
The proposed approach to implementate BPTT algorithm show the same performance as previous implementation without truncation, and is proportional to the length of input sequence or the number of steps errors are back-propagate for truncated one. The results of the experiments on show that the , and LSTM neural network learn the long dependencies.

### References