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

Token network(Token-wise MLP) :
![[Why Patch Embedding?_3.png]]

Pixel 단위가 아닌 token단위의 연산은 계산 효율 때문일 것이라는 건 너무 당연하다.

Fully connected layer로 이미지를 분석하기보다는 convolution이 계산 효율적이라는 점을 보면 알 수 있을 것이다.

내 생각에는, 계산 효율성 이외에도, pixel에는 context 정보를 포함하기 어려운 것도 한가지 이유가 될 수 있을 것 같다.

Patch와 그들간의 관계들을 사용하여 feature를 추출할 수 있다는 것은 이미 CNN에서 증명되었다. \
(Vision 정보는 인접 픽셀간의 locality가 존재한다는 것을 미리 알고 있기 때문에 Conv layer는 인접 픽셀간의 정보를 추출하기 위한 목적으로 설계)

이는 patch가 의미 정보 혹은 context를 가질 수 있는 **단위**가 될 수 있음을 의미한다고 생각한다.

우리가 문장을 분석할 때, 알파벳들만 보고 문장을 파악하기 보다 단어를 보고 문장을 파악하는 것이 유리하듯이, vision transformer 또한 token 혹은 patch로 나눠서 보는 것이 유리할 것으로 생각된다.

어쩌면 ViT의 논문 제목이 "An Image is worth 16X16 **words**"인 것도 그러한 생각이 내재되어 있기 때문이 아닐까 생각된다.

