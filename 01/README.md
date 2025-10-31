# Lesson 1

We will learn how to build [Micrograd](https://github.com/karpathy/micrograd).

## Backpropagation

Backpropagation is an algorithm that allows you to efficiently
evaluate the gradient of some kind of a loss function, with respect
to the weights of a model.

We can then iteratively tune the weights of that Neural Network to
minimize the loss function, and therefore improve the accuracy
of the network.

It is at the core of every modern neural network library like PyTorch,
Tensorflow, etc.

Micrograd allows the building of mathemtical expressions.

## Relu

Relu (short for `Rectified Linear Unit`) is a **simple, non-linear activation function** used in Neural Nets.

### Mathemtically:

$$f(x) = max(0, x)$$

So:

- If $$x \ge 0$$, it outputs $$x$$
- If $$x \le 0$$, it outputs $$0$$.

### Examples:

| Input | Output | Reason                   |
| ----- | ------ | ------------------------ |
| 2     | 2      | 2 is larger than 0.      |
| \-5   | 0      | \-5 is smaller than 0.   |
| 0     | 0      | 0 is less or equal to 0. |
