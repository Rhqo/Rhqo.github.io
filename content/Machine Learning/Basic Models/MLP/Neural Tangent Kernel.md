---
title: Neural Tangent Kernel
---
[Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572)

> At initialization, artificial neural networks (ANNs) are equivalent to Gaussian processes in the infinite-width limit, thus connecting them to kernel methods. We prove that the evolution of an ANN during training can also be described by a kernel: during gradient descent on the parameters of an ANN, the network function $f_\theta$ (which maps input vectors to output vectors) follows the kernel gradient of the functional cost (which is convex, in contrast to the parameter cost) w.r.t. a new kernel: the **Neural Tangent Kernel (NTK)**.

Neural Net이 infinite의 width 또는 데이터에 관해 sufficiently large width를 지니고 있으면, \
Neural Net이 estimated function을 학습하는 과정은 결과적으로 kernel regression과 같다. \
이를 Neural Tangent Kernel(NTK)라고 한다. 


> [!Tips] [[What is Kernel Regression?]]
> NN이 infinite width일 때 어떻게 kernel regression, kernel function으로 되는지 알기 위해서는, \
> kernel regression이 뭔지 알아야 할 필요가 있다.

