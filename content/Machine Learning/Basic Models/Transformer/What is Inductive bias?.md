---
title: What is Inductive bias?
---

> Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amounts of data. \
> Transformer는 CNN에 고유한 inductive bias가 부족하므로 충분하지 못한 양의 데이터으로 학습할 때 일반화가 잘 되지 않는다.

Inductive bias, 귀납적 편향이란 모델이 구조적으로 내재하고 있는 가정이나 편견을 의미하며, \
모델이 학습 과정에서 특정한 패턴이나 관계를 더 쉽게 학습할 수 있도록 돕는 구조적 특성이다.

CNN이 왜 inductive bias가 큰 것인지에 대한 것은 명확하게 밝혀진 바는 없지만, 이론적인 분석은 다음 논문에 있다. \
([Theoretical Analysis of Inductive Biases in Deep Convolutional Networks](https://arxiv.org/abs/2305.08404))

간략히 논문 내용을 소개하자면, CNN 구조는 multi-channeling & downsampling, weight sharing & locality의 특징을 가지고 있다.

Down sampling은 네트워크가 점차 넓은 정보를 볼 수 있게 receptive field를 확장해주면서, 필요한 depth를 log-scale로 줄이는 것이고, multi-channeling은 downsampling 과정에서 손실되는 정보를 보완해주는 것이다. \
Weight sharing은 특정 kernel의 weight이 입력 feature map의 서로 다른 공간 위치에 걸쳐 동일하게 적용된다는 것이고, locality는 작은 크기의 kernel을 사용하는 것이다. \
이러한 특징들이 모델 어떠한 constraint를 가하게 되고, 이것이 inductive bias이며, CNN이 적은 양의 데이터로 학습이 되도록 돕는 구조적인 특징이 될 것이다.

CNN이 이러한 inductive bias를 가지고 있기 때문에, transformer에 비해 적은 양의 데이터로도 학습이 가능한 것이다.

머신러닝은 특정 문제를 풀기 위해 학습 데이터에 대해서 가장 loss가 작은 function를 찾게 된다.

하지만 function의 제한이 없다면 overfitting이 일어나므로 제한을 걸어주는데, 이 제한이 바로 Inductive bias이다. 

하지만 Inductive bias가 적절하지 못하거나 지나치게 강하면 학습을 통해 얻은 function의 성능이 좋지 않을 수 있다.

Inductive bias가 강하면 오히려 generalization(variance)이 떨어져 오히려 학습을 방해하여 성능을 저해할 요소가 될 수 있으므로 Inductive Bias과 generalization은 trade-off가 있다.

결국, ViT 논문에서 나오는 것과 같이 "CNN보다 Inductive Bias가 부족하다"라는 것은 bias-variance의 trade-off로 설명이 가능하다.

다음 표는 transformer 이전의 논문의 표라 transformer에 대한 건 없지만, \
FCNN, CNN, RNN, GNN의 inductive bias의 비교이다. \
![[What is Inductive bias?_0.png]]
[Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261) \
각 구조에서 entity들 간의 relation에 의해 생기는 relational inductive bias가 생기게 된다는 내용이다. \
각각이 가지고 있는 inductive bias 덕분에 적은 데이터로 쉽게 generalize하는 것이 가능한 동시에, inductive bias 때문에 모델의 성능이 저해될 수 있다.


> [!Tips] [[What is Bias-Variance Tradeoff?]]
