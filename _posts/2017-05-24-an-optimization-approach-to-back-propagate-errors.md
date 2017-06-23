---
layout: post
title: An optimization approach to back-propagate errors
abstract: It is easy to get confused about back-propagation through time (BPTT) algorithm when starting to implement it in some applications, at least I did. In this post, if BPTT should always be implemented with truncation will be discussed, and a common mistake in impementation of BPTT will be explained.
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

```
def bptt(self):
    """A naive implementation of BPTT.
    """
    dLdb = np.zeros(self.hidden_size)
    dLdx = np.zeros((self.T, self.input_size))
    dLdU = np.zeros((self.hidden_size, self.input_size))
    dLdW = np.zeros((self.hidden_size, self.hidden_size))
    for t in xrange(self.T-1, -1, -1):
        dLdp = self.dLds[t] * (1.0 - (self.s[t] ** 2))
        for step in xrange(t, -1, -1):
            dLdU += np.outer(dLdp, self.x[step])
            dLdW += np.outer(dLdp, self.s[step-1])
            dLdx[step] += np.dot(self.U.T, dLdp)
            dLdb += dLdp
            dLdp = np.dot(self.W.T, dLdp) * (1.0 - (self.s[step-1] ** 2))
    return dLdx, dLdU, dLdW, dLdb
```

At each time step, the error should be backpropagation throngh all the previous time step. If the input sequence is long, the compution will be very expensive. But, with a careful observe, if 

$$\frac{\partial{E_i}}{\partial{x_i}}=\sum_{t=i}^{n}\frac{\partial{e_t}}{\partial{x_i}}=\frac{\partial{s_i}}{\partial{x_i}}\sum_{t=i}^{n}\frac{\partial{e_t}}{\partial{s_i}}$$

which can be rewrited as:

$$\frac{\partial{E_i}}{\partial{x_i}} = U^T\frac{\partial{E_i}}{\partial{s_i}} =  U^T(\frac{\partial{e_i}}{\partial{s_i}} + W^T\frac{\partial{E_{i+1}}}{\partial{s_i}})$$

it indicates that the error two part: the error from current output and the error from all late time step. Instead of propagation the error to all the previous time step, just accumulate the error gradient to the hidden state of directly previous step. 

The 

```
def new_bptt(self):
    """A optimized implementation of BPTT.
    """
    dLdb = np.zeros(self.hidden_size)
    dLdx = np.zeros((self.T, self.input_size))
    dLdU = np.zeros((self.hidden_size, self.input_size))
    dLdW = np.zeros((self.hidden_size, self.hidden_size))
    for t in xrange(self.T-1, -1, -1):
        dLdp = self.dLds[t] * (1.0 - (self.s[t] ** 2))
        dLdU += np.outer(dLdp, self.x[t])
        dLdW += np.outer(dLdp, self.s[t-1])
        dLdx[t] += np.dot(self.U.T, dLdp)
        dLdb += dLdp
        self.dLds[t-1] += np.dot(self.W.T, dLdp)
    return dLdx, dLdU, dLdW, dLdb
```

### 3. Comarsion
The computational complexity of previous implemetation of BPTT without truncation is $$O(n^2)$$, and $$n$$ is the length of input sequence. the proposed approach for BPTT is $$O(n)$$, which mean the new approach will be n times faster than preivous one. For truncated BPTT, like m steps the computational complexity will be $$O(nm)$$, and slower than the optimization method.

the performance, long-short term memeory (LSTM) long dependencies, but with truncation BPTT, the advantage of LSTM is limited. In this paper, 

Model | BPTT | PPL
------|------|----
RNN   | old  | 252

### 4. Conclusion
The proposed approach to implementate BPTT algorithm show the same performance as previous implementation without truncation, and is proportional to the length of input sequence or the number of steps errors are back-propagate for truncated one. The results of the experiments on show that the , and LSTM neural network learn the long dependencies.

### References