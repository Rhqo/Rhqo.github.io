---
title: Why Patch Embedding?
---
기존의 vanilla neural net에서는 이미지를 처리하기 위해 neuron을 사용하며, 이미지를 'neuronize'하는 과정을 거쳐야 했다.

Neuronize는 이미지의 각 pixel을 3개의 neuron(R, G, B)로 나누는 방법이다.

이와 달리 transformer 및 attention 구조에서는, 이미지를 처리하기 위해 token을 사용하며, 이미지를 'tokenize'하는 과정을 거쳐야 한다.

Token은 'encapsulated groups of neruon'이며, Tokenization은 이미지를 vector의 set으로 바꾸는 과정이다.

![[Why Patch Embedding?_0.png]]
Transformer에서는 모든 연산을 token 단위로 수행하게 된다.

Linear combination :
![[Why Patch Embedding?_1.png]]

Activation function $F_\theta$ :
![[Why Patch Embedding?_2.png]]

Token network :
![[Why Patch Embedding?_3.png]]

Pixel 단위가 아닌 token단위의 연산은 계산 효율 때문일 것이라는 건 너무 당연하다.

Fully connected layer로 이미지를 분석하기보다는 convolution이 계산 효율적이라는 점을 보면 알 수 있을 것이다.

